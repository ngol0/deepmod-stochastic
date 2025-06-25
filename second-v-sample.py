# V-Prime Training Using Weighted Sampling from MDN

import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset

# Environment setup
states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
num_states = len(states)
actions = ['up', 'down', 'right', 'left']
num_actions = len(actions)

# Helper functions for reward conversion
def class_to_reward(class_idx):
    reward_map = {0: -1, 1: 0, 2: 1}
    return reward_map[class_idx]

# One-hot encoding
def one_hot(state):
    vec = np.zeros(num_states)
    vec[states.index(state)] = 1
    return torch.tensor(vec, dtype=torch.float32)

# Load the complete dataset with rewards
def load_transition_dataset(filename='transition_features_with_rewards.pt'):
    return torch.load(filename)

def load_current_reward(filename='unique_rewards.pt'):
    """
    Load the current reward dataset
    """
    try:
        dataset = torch.load(filename)
        print(f"Loaded dataset with {len(dataset['input_features'])} data")
        return dataset
    except FileNotFoundError:
        print(f"File {filename} not found. Please check the path.")
        return None
    
def find_most_similar_state_to_letters(target_features, v_network):
    """
    Find the most similar state using V-network features like in the reference
    """
    best_match = None
    best_similarity = -1
    
    if isinstance(target_features, torch.Tensor):
        target_features = target_features.numpy()
    
    target_norm = np.linalg.norm(target_features)
    if target_norm == 0:
        return "ZERO_VECTOR", 0.0
    
    for test_state in states:
        test_vec = one_hot(test_state)
        test_features = v_network.extract_features(test_vec).numpy()
        
        test_norm = np.linalg.norm(test_features)
        if test_norm == 0:
            continue
            
        # Cosine similarity
        similarity = np.dot(target_features, test_features) / (target_norm * test_norm)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = test_state
    
    return best_match, best_similarity

#------------------------------Networks----------------------------------------------------
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
        
        self.num_mixtures = num_mixtures
        self.feature_dim = feature_dim
    
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
    
    def sample_next_features(self, x, num_samples=1):
        """
        Sample next features from the MDN distribution
        """
        with torch.no_grad():
            pi, sigma, mu = self.mdn_head(x)
            batch_size = x.shape[0]
            
            samples = []
            for _ in range(num_samples):
                # Sample which mixture component to use based on pi
                mixture_idx = torch.multinomial(pi, 1).squeeze()
                
                # Get the mean and std for selected mixture
                selected_mu = mu[range(batch_size), mixture_idx]
                selected_sigma = sigma[range(batch_size), mixture_idx]
                
                # Sample from the Gaussian
                eps = torch.randn_like(selected_mu)
                sample = selected_mu + selected_sigma * eps
                
                samples.append(sample)
            
            if num_samples == 1:
                return samples[0]
            return torch.stack(samples)
    
    def sample_with_mixture_weights(self, x, num_samples_per_mixture=3):
        """
        Sample from each mixture component weighted by pi
        Returns samples and their corresponding weights
        """
        with torch.no_grad():
            pi, sigma, mu = self.mdn_head(x)
            batch_size = x.shape[0]
            
            all_samples = []
            all_weights = []
            
            # Sample from each mixture
            for mix_idx in range(self.num_mixtures):
                mix_weight = pi[:, mix_idx]
                mix_mu = mu[:, mix_idx]
                mix_sigma = sigma[:, mix_idx]
                
                # Generate samples from this mixture
                for _ in range(num_samples_per_mixture):
                    eps = torch.randn_like(mix_mu)
                    sample = mix_mu + mix_sigma * eps
                    all_samples.append(sample)
                    all_weights.append(mix_weight)
            
            # Stack and return
            samples = torch.stack(all_samples, dim=1)  # [batch, n_samples, feature_dim]
            weights = torch.stack(all_weights, dim=1)  # [batch, n_samples]
            
            return samples, weights
        
# Define V-network (same architecture used during training)
class VNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # Feature layer
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

    def extract_features(self, x):
        with torch.no_grad():
            return self.tanh(self.fc1(x))

#-----------------------------Loading and Preparing Data----------------------------------
def load_everything():
    # Load dataset
    try:
        print("Loading dataset...")
        dataset = load_transition_dataset()
        print("✅ Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset['input_features'])}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load trained MDN model
    try:
        print("Loading trained MDN model...")
        input_dim = dataset['input_features'].shape[1]  # Should be 36
        feature_dim = dataset['target_features'].shape[1]  # Should be 32
        
        model = FeatureEnvironmentModelWithMDN(input_dim, feature_dim, num_mixtures=5)
        model.load_state_dict(torch.load('D:/City-learning/virtual-env/deepmod/later_models/mdn_environment_model3.pth'))
        model.eval()
        print("✅ MDN Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load V-network for state feature extraction
    try:
        print("Loading V-network for state mapping...")
        v_network = VNetwork(num_states)
        v_network.load_state_dict(torch.load('D:/City-learning/virtual-env/deepmod/later_models/v_network.pth'))
        v_network.eval()
        print("✅ V-network loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading V-network: {e}. Will skip state letter mapping tests")
        v_network = None
    
    return dataset, model, v_network

#---------------------------------New network------------------------------------------
class VNetworkFeature(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 32)
        self.fc2 = nn.Linear(32, 32) 
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

###----------------------------------------------------------------------------------------------------------------
def train_vprime_with_weighted_sampling(current_reward, v_prime, mdn_improved, v_network, 
                                      num_epochs=4000, gamma=0.9, num_samples_per_mixture=3):
    """
    Train V-prime using weighted sampling from MDN distribution
    """
    unique_states = current_reward['input_features']
    
    print(f"Training V-prime with weighted MDN sampling for {num_epochs} epochs...")
    print(f"Found {len(unique_states)} unique states")
    print(f"Using {num_samples_per_mixture} samples per mixture component")
    
    optimizer = optim.Adam(v_prime.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Create target network for stability
    target_v_prime = VNetworkFeature(feature_dim=32)
    target_v_prime.load_state_dict(v_prime.state_dict())
    target_v_prime.eval()
    
    # Define actions (one-hot encoded)
    actions = [
        torch.tensor([1, 0, 0, 0], dtype=torch.float32),  # up
        torch.tensor([0, 1, 0, 0], dtype=torch.float32),  # down  
        torch.tensor([0, 0, 1, 0], dtype=torch.float32),  # right
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)   # left
    ]
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Process each unique state individually
        for state_idx, features in enumerate(unique_states):
            with torch.no_grad():
                mdn_improved.eval()
                target_v_prime.eval()
                
                target_value = -float('inf')
                
                # Evaluate all 4 actions for this state
                for action in actions:
                    # Create input: state + action
                    state_action_input = torch.cat([features, action]).unsqueeze(0)
                    
                    # Get reward prediction
                    _, _, _, reward_logits, pred_done = mdn_improved(state_action_input)
                    pred_reward_class = torch.argmax(reward_logits, dim=1).item()
                    pred_reward = class_to_reward(pred_reward_class)
                    
                    # Get current state reward from dataset
                    current_state_reward = current_reward['rewards'][state_idx].item()
                    
                    # Sample from MDN with weights
                    samples, weights = mdn_improved.sample_with_mixture_weights(
                        state_action_input, 
                        num_samples_per_mixture=num_samples_per_mixture
                    )
                    
                    # Compute values for all samples
                    sample_values = []
                    for i in range(samples.shape[1]):
                        next_features = samples[:, i].squeeze()
                        next_value = target_v_prime(next_features.unsqueeze(0)).squeeze().item()
                        #next_value = np.clip(next_value, -10.0, 10.0)
                        sample_values.append(next_value)
                    
                    # Compute weighted average (expected value)
                    sample_values = torch.tensor(sample_values)
                    weights_squeezed = weights.squeeze()
                    normalized_weights = weights_squeezed / weights_squeezed.sum()
                    expected_next_value = (normalized_weights * sample_values).sum().item()
                    
                    # Bellman update with expected value
                    q_value = current_state_reward + gamma * expected_next_value
                    target_value = max(target_value, q_value)
                
                # Clamp target value
                #target_value = np.clip(target_value, -10.0, 10.0)
            
            # Train on this single state
            v_prime.train()
            optimizer.zero_grad()
            
            # Forward pass for current state
            current_prediction = v_prime(features.unsqueeze(0)).squeeze()
            target_tensor = torch.tensor([[target_value]], dtype=torch.float32)
            
            # Compute loss
            loss = loss_fn(current_prediction, target_tensor)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v_prime.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Update target network periodically
        if epoch % 100 == 0:
            target_v_prime.load_state_dict(v_prime.state_dict())
            target_v_prime.eval()
        
        # Print progress
        if epoch % 100 == 0:
            avg_loss = epoch_loss / len(unique_states)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
            
            # Early stopping if loss explodes
            if avg_loss > 1000:
                print(f"Loss exploded! Reducing learning rate...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            # Test current policy quality
            if epoch % 500 == 0:
                test_policy(v_prime, v_network, mdn_improved, 'A')

    return v_prime

#------------------------------Testing Current Policy Quality----------------------------------

def test_policy(v_prime, v_network, mdn_improved, test_state='A'):
    """Test the quality of current policy"""
    print("  Testing current policy quality...")
    time_step = 0
    total_rewards = 0
    
    actions = [
        torch.tensor([1, 0, 0, 0], dtype=torch.float32),  # up
        torch.tensor([0, 1, 0, 0], dtype=torch.float32),  # down  
        torch.tensor([0, 0, 1, 0], dtype=torch.float32),  # right
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)   # left
    ]
    action_names = ['up', 'down', 'right', 'left']
    
    v_prime.eval()
    mdn_improved.eval()
    
    with torch.no_grad():
        # state_features = random.choice(test_states)
        # state, sim = find_most_similar_state_to_letters(state_features, v_network)
        state = test_state
        state_vec = one_hot(state)
        state_features = v_network.extract_features(state_vec)
        print(f"Testing policy starting from state: {state}")
        print(f"\n    Starting state: {state}")

        while time_step < 99:
            #start random among unique state
                
            action_values = []
            all_next_features = []
            all_rewards = []
            all_done = []
                
            for j, (action, action_name) in enumerate(zip(actions, action_names)):
                # Create state-action input
                state_action = torch.cat([state_features, action]).unsqueeze(0)
                    
                # Get MDN prediction
                pi, sigma, mu, reward_logits, pred_done = mdn_improved(state_action)
                    
                # Get most likely next state
                best_mixture = torch.argmax(pi, dim=1)
                next_features = mu[0, best_mixture[0]]
                    
                next_value = v_prime(next_features.unsqueeze(0)).item()
                    
                # Get reward prediction
                pred_reward_class = torch.argmax(reward_logits, dim=1).item()
                pred_reward = class_to_reward(pred_reward_class)
                    
                # Calculate Q-value
                #q_value = pred_reward + 0.99 * next_value
                action_values.append(next_value)
                all_next_features.append(next_features)
                all_done.append(pred_done.item())
                all_rewards.append(pred_reward)
                    
                #print(f"      {action_name}: R={pred_reward}, V_next={next_value:.3f}")
                
            # Show best action
            best_action_idx = np.argmax(action_values)
            best_action = action_names[best_action_idx]
            #print(f"      Best action: {best_action}")

            next_state_features = all_next_features[best_action_idx]  # Update state for next step
            next_state, _ = find_most_similar_state_to_letters(next_state_features, v_network)
            print(f"     (Action: {best_action}) -> Next state: {next_state}")
            
            time_step += 1
            total_rewards += all_rewards[best_action_idx]  # Use the reward from the best action

            if (all_done[best_action_idx] > 0.5) :
                break
            
            state_features = next_state_features  # Update state for next iteration
    print(f"\nTotal steps taken: {time_step}, Total rewards: {total_rewards}")
    return time_step, total_rewards

def comprehensive_policy_evaluation(v_prime, v_network, mdn_improved, unique_states, state_to_letter_map=None):
    """
    Comprehensive evaluation of the final policy using feature-based states
    """
    print("\n" + "="*50)
    print("COMPREHENSIVE POLICY EVALUATION")
    print("="*50)
    
    actions = [
        torch.tensor([1, 0, 0, 0], dtype=torch.float32),  # up
        torch.tensor([0, 1, 0, 0], dtype=torch.float32),  # down  
        torch.tensor([0, 0, 1, 0], dtype=torch.float32),  # right
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)   # left
    ]
    action_names = ['up', 'down', 'right', 'left']
    
    policy_summary = {}
    
    v_prime.eval()
    mdn_improved.eval()
    
    # Test first 10 unique states for efficiency
    test_states = unique_states[:min(16, len(unique_states))]
    
    with torch.no_grad():
        for state_idx, state_features in enumerate(test_states):
            
            # Get state identifier (use index if no mapping provided)
            name, sim = find_most_similar_state_to_letters(state_features, v_network)
            state_name = f"STATE_{name}"
            
            action_values = []
            
            # Evaluate all actions for this state
            for action in actions:
                # Create state-action input
                state_action = torch.cat([state_features, action]).unsqueeze(0)
                
                # Get MDN prediction
                pi, sigma, mu, reward_logits, pred_done = mdn_improved(state_action)
                
                # Get most likely next state
                best_mixture = torch.argmax(pi, dim=1)
                next_features = mu[0, best_mixture[0]]
                
                # Get value of next state
                next_value = v_prime(next_features.unsqueeze(0)).item()
                
                action_values.append(next_value)
            
            # Determine best action
            best_action_idx = np.argmax(action_values)
            best_action = action_names[best_action_idx]
            
            # Store policy summary
            policy_summary[state_name] = {
                'state_features': state_features,
                'best_action': best_action,
                'all_q_values': action_values,
            }
            
            #print(f"{state_name}: {best_action} (Q={best_q_value:.3f}, R={best_reward}, V_next={best_next_value:.3f})")
    
    return policy_summary

def compare_with_ground_truth_policy(policy_summary):
    """
    Compare learned policy with expected optimal policy for FrozenLake
    """
    print("\n" + "="*50)
    print("POLICY COMPARISON WITH EXPECTED OPTIMAL")
    print("="*50)
    
    # Define expected optimal policy for FrozenLake 4x4
    # This is a rough approximation - actual optimal may vary
    expected_policy = {
        'STATE_A': 'right',  # Usually should move toward goal
        'STATE_B': 'right',
        'STATE_C': 'down',
        'STATE_D': 'left',
        'STATE_E': 'down',
        'STATE_F': 'right',  # Holes
        'STATE_G': 'down',
        'STATE_H': 'left', # Holes
        'STATE_I': 'right',
        'STATE_J': 'right',
        'STATE_K': 'down',
        'STATE_L': 'down', # Holes
        'STATE_M': 'right', # Holes
        'STATE_N': 'right',
        'STATE_O': 'right',
        'STATE_P': 'right'   # Goal state
    }
    
    matches = 0
    total = 0
    
    print("State | Learned | Expected | Match")
    print("------|---------|----------|------")
    
    for state_name, policy_info in policy_summary.items():
        learned_action = policy_info['best_action']
        expected_action = expected_policy.get(state_name, 'unknown')
        
        if expected_action != 'unknown':
            is_match = learned_action == expected_action
            matches += int(is_match)
            total += 1
            match_symbol = "✅" if is_match else "❌"
        else:
            match_symbol = "?"
        
        print(f"{state_name:>5} | {learned_action:>7} | {expected_action:>8} | {match_symbol}")
    
    if total > 0:
        accuracy = matches / total * 100
        print(f"\nPolicy match accuracy: {matches}/{total} ({accuracy:.1f}%)")

def main_evaluation(v_prime, v_network, mdn_improved, dataset):
    """
    Main evaluation function that runs all policy tests
    """
    # Get unique states from dataset
    input_features = dataset['input_features']
    state_features = input_features[:, :32]
    unique_states = torch.unique(state_features, dim=0)
    
    print(f"Evaluating policy on {len(unique_states)} unique states...")
    
    policy_summary = comprehensive_policy_evaluation(v_prime, v_network, mdn_improved, unique_states)
    
    # # Compare with expected policy
    compare_with_ground_truth_policy(policy_summary)
    
    test_policy(v_prime, v_network, mdn_improved, 'A')
    test_policy(v_prime, v_network, mdn_improved, 'B')
    test_policy(v_prime, v_network, mdn_improved, 'D')


def main():
    dataset, mdn, v_network = load_everything()
    current_reward = load_current_reward()

    # Execute the training
    print("="*60)
    print("RETRAINING V-PRIME WITH IMPROVED MDN")
    print("="*60)

    v_prime_improved = VNetworkFeature(feature_dim=32)  # 32-dimensional features

    v_prime_final = train_vprime_with_weighted_sampling(
        current_reward,
        v_prime_improved, 
        mdn,  # Use the improved MDN we just trained
        v_network
    )


    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)

    # Save the trained V-prime
    torch.save(v_prime_final.state_dict(), 'v_prime_improved_final.pth')
    print("Improved V-prime saved to 'v_prime_improved_final.pth'")

    # Comprehensive evaluation
    policy_summary = main_evaluation(v_prime_final, v_network, mdn, dataset)

    # Compare with original V-network values
    print("\n" + "="*50)
    print("COMPARISON: V-Prime vs Original V-Network")
    print("="*50)
    print("State | V-Prime Value | Original V | Difference")
    print("------|---------------|------------|------------")

    for state in states:
        state_vec = one_hot(state)
        
        # V-prime value (using learned features)
        with torch.no_grad():
            state_features = v_network.extract_features(state_vec)
            #state_features_binary = torch.where(state_features > 0, 1.0, -1.0)
            vprime_value = v_prime_final(state_features.unsqueeze(0)).item()
        
        # Original V-network value
        with torch.no_grad():
            original_value = v_network(state_vec).item()
        
        difference = vprime_value - original_value
        print(f"  {state}   | {vprime_value:11.3f} | {original_value:8.3f} | {difference:9.3f}")

    print("\n🎉 V-Prime training with improved MDN completed!")

if __name__ == "__main__":
    main()