import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
from sklearn.model_selection import train_test_split

# Assume you have your states and actions defined
states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
actions = ['up', 'down', 'right', 'left']
num_states = len(states)
num_actions = len(actions)
# listOfHoles = np.array(['F', 'H', 'L', 'M'])

def create_stratified_balanced_dataset(dataset, samples_per_state_action=50):
    """
    Create a more carefully balanced dataset by ensuring coverage 
    of all state-action combinations
    """
    input_features = dataset['input_features']
    target_features = dataset['target_features']
    rewards = dataset['rewards']
    done_flags = dataset['done_flags']
    
    # Extract state and action features
    state_features = input_features[:, :32]
    action_features = input_features[:, 32:]
    
    # Group by state-action pairs
    state_action_groups = {}
    
    for i in range(len(input_features)):
        # Convert state features to a hashable key
        state_key = tuple(torch.round(state_features[i], decimals=3).numpy())
        action_idx = torch.argmax(action_features[i]).item()
        
        key = (state_key, action_idx)
        if key not in state_action_groups:
            state_action_groups[key] = []
        state_action_groups[key].append(i)
    
    print(f"Found {len(state_action_groups)} unique state-action combinations")
    
    # Sample from each group
    balanced_indices = []
    state_action_coverage = {}
    
    for (state_key, action_idx), indices in state_action_groups.items():
        n_samples = min(samples_per_state_action, len(indices))
        if n_samples > 0:
            sampled = torch.tensor(indices)[torch.randperm(len(indices))[:n_samples]]
            balanced_indices.extend(sampled.tolist())
            
            if action_idx not in state_action_coverage:
                state_action_coverage[action_idx] = 0
            state_action_coverage[action_idx] += n_samples
    
    print("Samples per action after stratification:")
    actions = ['up', 'down', 'right', 'left']
    for action_idx, count in state_action_coverage.items():
        print(f"  {actions[action_idx]}: {count}")
    
    balanced_indices = torch.tensor(balanced_indices)
    
    return {
        'input_features': input_features[balanced_indices],
        'target_features': target_features[balanced_indices],
        'rewards': rewards[balanced_indices],
        'done_flags': done_flags[balanced_indices]
    }

def one_hot_state(state):
    vec = np.zeros(num_states)
    vec[states.index(state)] = 1
    return vec

def find_most_similar_state(target_features, v_network, states):
    """Helper function to find the most similar state"""
    best_match = None
    best_sim = -1
    
    for test_state in states:
        test_vec = one_hot_state(test_state)
        test_features = v_network.extract_features(test_vec)
        test_binary = torch.where(test_features > 0, 1.0, -1.0).float()
        
        sim = torch.nn.functional.cosine_similarity(
            target_features.unsqueeze(0), 
            test_binary.unsqueeze(0)
        ).item()
        
        if sim > best_sim:
            best_sim = sim
            best_match = test_state
    
    return best_match

def state_to_idx(state):
    return states.index(state)

# Helper functions for reward conversion
def reward_to_class(reward):
    """Convert reward value to class index"""
    if reward == -1:
        return 0
    elif reward == 0:
        return 1
    elif reward == 1:
        return 2
    else:
        raise ValueError(f"Unexpected reward value: {reward}")

def class_to_reward(class_idx):
    """Convert class index back to reward value"""
    reward_map = {0: -1, 1: 0, 2: 1}
    return reward_map[class_idx]

#-------------------Network and Dataset Definitions-------------------
# Dataset class for feature-based transitions with discrete rewards
class FeatureTransitionDataset(Dataset):
    def __init__(self, dataset_dict):
        """
        dataset_dict should contain:
        - 'input_features': array of shape (N, feature_dim) - current state features
        - 'target_features': array of shape (N, feature_dim) - next state features  
        - 'rewards': array of shape (N,) - rewards (values: -1, 0, 1)
        - 'done_flags': array of shape (N,) - done flags
        """
        self.input_features = torch.FloatTensor(dataset_dict['input_features'])
        self.target_features = torch.FloatTensor(dataset_dict['target_features'])
        self.done_flags = torch.FloatTensor(dataset_dict['done_flags'])
        
        # Convert rewards to class indices
        reward_classes = []
        for reward in dataset_dict['rewards']:
            reward_classes.append(reward_to_class(reward))
        self.reward_classes = torch.LongTensor(reward_classes)
    
    def __len__(self):
        return len(self.input_features)
    
    def __getitem__(self, idx):
        return (
            self.input_features[idx],      # Input: state features (+ action if included)
            self.target_features[idx],     # Target: next state features
            self.reward_classes[idx],      # Reward class (0, 1, or 2)
            self.done_flags[idx]           # Done flag
        )

class ImprovedMDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures=8):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        # Separate the action and state processing
        self.action_processor = nn.Linear(4, 16)  # Process action separately
        self.state_processor = nn.Linear(32, 48)  # Process state separately
        
        # Main network processes the combined features (48 + 16 = 64)
        self.fc1 = nn.Linear(64, 128)  # 64 = 48 (state) + 16 (action)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        
        self.pi = nn.Linear(64, num_mixtures)
        self.sigma = nn.Linear(64, num_mixtures * output_dim)
        self.mu = nn.Linear(64, num_mixtures * output_dim)
        
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        # Initialize mixture weights to be uniform
        nn.init.constant_(self.pi.bias, 0.0)
        nn.init.normal_(self.pi.weight, 0, 0.01)
        
        # Initialize sigma to reasonable values (not too small)
        nn.init.constant_(self.sigma.bias, -0.5)
        nn.init.normal_(self.sigma.weight, 0, 0.01)
    
    def forward(self, x):
        # Split input into state and action
        state_part = x[:, :32]
        action_part = x[:, 32:]
        
        # Process separately then combine
        state_processed = self.relu(self.state_processor(state_part))
        action_processed = self.relu(self.action_processor(action_part))
        
        combined = torch.cat([state_processed, action_processed], dim=1)
        
        # Main processing
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Output heads
        pi = nn.functional.softmax(self.pi(x), dim=1)
        sigma = torch.clamp(torch.exp(self.sigma(x)), min=1e-3, max=5.0).view(-1, self.num_mixtures, self.output_dim)
        mu = self.mu(x).view(-1, self.num_mixtures, self.output_dim)
        
        return pi, sigma, mu

# Environment Model Network with MDN for Features
class FeatureEnvironmentModelWithMDN(nn.Module):
    def __init__(self, input_dim, feature_dim, num_mixtures=8):
        super().__init__()
        
        # Shared layers for reward and done prediction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # MDN for next state features prediction
        self.mdn_head = ImprovedMDN(input_dim, feature_dim, num_mixtures)
        
        # Keep the original reward and done heads
        self.reward_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 classes: -1, 0, 1
        )
        
        self.done_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single value for done probability
        )
    
    def forward(self, x):
        # Get MDN predictions for next features
        pi, sigma, mu = self.mdn_head(x)
        
        # Get reward and done predictions using shared layers
        shared_features = self.shared(x)
        reward_logits = self.reward_head(shared_features)
        done = torch.sigmoid(self.done_head(shared_features))
        
        return pi, sigma, mu, reward_logits, done
    
    def predict_next_features(self, x):
        """
        Get the most likely next features from MDN
        """
        with torch.no_grad():
            pi, sigma, mu = self.mdn_head(x)
            # Use the mixture with highest probability
            best_mixture_idx = torch.argmax(pi, dim=1)
            batch_size = x.shape[0]
            
            next_features = mu[range(batch_size), best_mixture_idx]
            return next_features

#--------------------------Training Function-------------------------
# MDN Components
def mdn_loss(target, pi, sigma, mu):
    """
    MDN loss function (negative log likelihood)
    """
    # Expand target to match mixture dimensions
    target = target.unsqueeze(1).expand_as(mu)
    
    # Calculate probability for each mixture
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target).sum(dim=2)  # Sum over output dimensions
    
    # Weight by mixture probabilities
    weighted_log_prob = log_prob + torch.log(pi + 1e-10)
    
    # Log-sum-exp for numerical stability
    max_log_prob = torch.max(weighted_log_prob, dim=1, keepdim=True)[0]
    log_sum_exp = max_log_prob + torch.log(
        torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=1, keepdim=True)
    )
    
    return -torch.mean(log_sum_exp)

# Training function for MDN-based model with curriculum learning
def train_mdn_feature_model_curriculum(model, dataset_dict, num_epochs=1000):
    """
    Train MDN model with curriculum learning - start with easy examples, gradually add harder ones
    """
    print("Starting MDN training with curriculum learning...")
    
    # Extract data from dataset dict
    input_features = torch.FloatTensor(dataset_dict['input_features'])
    target_features = torch.FloatTensor(dataset_dict['target_features'])
    
    # Convert rewards to class indices
    reward_classes = []
    for reward in dataset_dict['rewards']:
        reward_classes.append(reward_to_class(reward))
    reward_classes = torch.LongTensor(reward_classes)
    
    #done_flags = torch.FloatTensor(dataset_dict['done_flags'])
    # Handle done flags - convert boolean to float if needed
    done_flags = dataset_dict['done_flags']
    if isinstance(done_flags, torch.Tensor):
        done_flags = done_flags.float()  # Convert boolean tensor to float
    else:
        done_flags = torch.FloatTensor(done_flags)  # Convert numpy array to float tensor
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8)
    
    # Start with smaller batches, increase gradually
    initial_batch_size = 16
    max_batch_size = 64
    
    # Loss functions
    reward_criterion = nn.CrossEntropyLoss()   # For reward prediction (classification)
    done_criterion = nn.BCELoss()              # For done prediction
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Curriculum: gradually increase batch size
        current_batch_size = min(max_batch_size, initial_batch_size + epoch // 100)
        
        # Create dataloader for this epoch
        dataset_loader = DataLoader(
            TensorDataset(input_features, target_features, reward_classes, done_flags),
            batch_size=current_batch_size,
            shuffle=True
        )
        
        total_loss = 0.0
        total_reward_acc = 0.0
        num_batches = 0
        
        model.train()
        for batch_input, batch_target, batch_rewards, batch_dones in dataset_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pi, sigma, mu, reward_logits, pred_dones = model(batch_input)
            
            # Main MDN loss for features
            mdn_loss_val = mdn_loss(batch_target, pi, sigma, mu)
            
            # Reward and done losses
            reward_loss = reward_criterion(reward_logits, batch_rewards)
            done_loss = done_criterion(pred_dones.squeeze(), batch_dones)
            
            # Diversity regularization (stronger)
            pi_entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1).mean()
            diversity_loss = -0.05 * pi_entropy  # Encourage uniform mixtures
            
            # Mixture separation regularization
            separation_loss = 0.0
            for i in range(mu.shape[1]):
                for j in range(i+1, mu.shape[1]):
                    # Encourage different mixtures to be far apart
                    distance = torch.norm(mu[:, i] - mu[:, j], dim=1).mean()
                    separation_loss += torch.exp(-distance)  # Penalty for being too close
            separation_loss *= 0.01
            
            # Combined loss
            total_loss_batch = mdn_loss_val + reward_loss + done_loss + diversity_loss + separation_loss
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Calculate reward accuracy
            pred_reward_classes = torch.argmax(reward_logits, dim=1)
            total_reward_acc += (pred_reward_classes == batch_rewards).float().mean().item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_reward_acc = total_reward_acc / num_batches
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Reward Acc: {avg_reward_acc:.3f}, "
                  f"Batch Size: {current_batch_size}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Test diversity every 100 epochs
            #test_mdn_diversity_quick(model, v_network)
        
        # Early stopping
        if patience_counter > 200:  # Stop if no improvement for 200 epochs
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model


#-------------------------Main Execution-------------------------
# Main execution for feature-based data
if __name__ == "__main__":
    # Load your actual dataset
    print("Loading dataset from 'transition_features_with_rewards.pt'...")
    dataset = torch.load('transition_features_with_rewards.pt')

    print("=== Creating Stratified Balanced Dataset ===")
    stratified_dataset = create_stratified_balanced_dataset(dataset, samples_per_state_action=30)
    
    # Extract the data
    input_features = dataset['input_features']
    target_features = dataset['target_features'] 
    rewards = dataset['rewards']
    done_flags = dataset['done_flags']
    
    print(f"Dataset loaded successfully!")
    print(f"Input features shape: {input_features.shape}")
    print(f"Target features shape: {target_features.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Done flags shape: {done_flags.shape}")
    
    # Convert to numpy for the dataset class
    dataset_dict = {
        'input_features': input_features.numpy() if isinstance(input_features, torch.Tensor) else input_features,
        'target_features': target_features.numpy() if isinstance(target_features, torch.Tensor) else target_features,
        'rewards': rewards.numpy() if isinstance(rewards, torch.Tensor) else rewards,
        'done_flags': done_flags.numpy() if isinstance(done_flags, torch.Tensor) else done_flags
    }
    
    # Check unique reward values
    unique_rewards = np.unique(dataset_dict['rewards'])
    print(f"Unique reward values: {unique_rewards}")
    
    # Verify rewards are in expected format
    expected_rewards = {-1, 0, 1}
    actual_rewards = set(unique_rewards)
    if not actual_rewards.issubset(expected_rewards):
        print(f"Warning: Found unexpected reward values: {actual_rewards - expected_rewards}")
    
    print(f"Dataset size: {len(dataset['input_features'])}")
    
    # Choose which model to train
    USE_MDN_MODEL = True
    
    input_dim = dataset_dict['input_features'].shape[1]  # Should be 36 (32 state + 4 action)
    feature_dim = dataset_dict['target_features'].shape[1]  # Should be 32 for your data
    
    if USE_MDN_MODEL:
        print("\n🔄 Training with MDN-based model...")
        
        model = FeatureEnvironmentModelWithMDN(input_dim, feature_dim, num_mixtures=5)
        print(f"Model architecture: {input_dim} -> MDN({feature_dim}) + reward + done")
        
        # Train the model with MDN-specific function
        model = train_mdn_feature_model_curriculum(model, stratified_dataset, num_epochs=1000)
        
        # Save the model
        torch.save(model.state_dict(), 'mdn_environment_model.pth')
        print("MDN Model saved as 'mdn_environment_model.pth'")
    else:
        print("Error: Please set USE_MDN_MODEL to True to train the MDN-based model.")

    print(f"\nTraining completed!")