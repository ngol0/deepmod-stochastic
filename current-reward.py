import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random

# Environment setup
states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
num_states = len(states)
actions = ['up', 'down', 'right', 'left']
num_actions = len(actions)

R = np.array([
    ['A', 'E', 'B', 'A'], ['B', 'F', 'C', 'A'], ['C', 'G', 'D', 'B'], ['D', 'H', 'D', 'C'],
    ['A', 'I', 'F', 'E'], ['B', 'J', 'G', 'E'], ['C', 'K', 'H', 'F'], ['D', 'L', 'H', 'G'],
    ['E', 'M', 'J', 'I'], ['F', 'N', 'K', 'I'], ['G', 'O', 'L', 'J'], ['H', 'P', 'L', 'K'],
    ['I', 'M', 'N', 'M'], ['J', 'N', 'O', 'M'], ['K', 'O', 'P', 'N'], ['L', 'P', 'P', 'O']
])

listOfHoles = ['F', 'H', 'L', 'M']
maxSteps = 99

# One-hot encoding
def one_hot(state):
    vec = np.zeros(num_states)
    vec[states.index(state)] = 1
    return torch.tensor(vec, dtype=torch.float32)

# Q-network definition
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
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

# Transition probabilities
P = {}
for i, state in enumerate(states):
    P[state] = {}
    for a in range(num_actions):
        intended_state = R[i][a]
        transitions = [(intended_state, 0.8)]  # Intended move
        for random_a in range(num_actions):  # Random moves
            random_state = R[i][random_a]
            prob = 0.2 / num_actions
            transitions.append((random_state, prob))

        # Combine same next states
        combined = {}
        for s_prime, p in transitions:
            combined[s_prime] = combined.get(s_prime, 0) + p
        P[state][actions[a]] = list(combined.items())

# Environment step function
def step(state, action):
    transitions = P[state][action]
    next_states, probs = zip(*transitions)
    next_state = np.random.choice(next_states, p=probs)
    return next_state

# Reward function
def get_reward(next_state):
    """Calculate reward based on state transition"""
    if next_state == 'P':
        return 1.0  # Goal reached
    elif next_state in listOfHoles:
        return -1.0  # Fell into hole
    else:
        return 0.0  # Normal step

# Terminal state check
def is_terminal(state):
    """Check if state is terminal (goal or hole)"""
    return state == 'P'

# Load saved networks
v_network = VNetwork(num_states)
v_network.load_state_dict(torch.load('D:/City-learning/virtual-env/deepmod/later_models/v_network.pth'))
v_network.eval()

policy_net = DQN(num_states, num_actions)
policy_net.load_state_dict(torch.load('D:/City-learning/virtual-env/deepmod/later_models/policy_net.pth'))

print("Load successfully")

# Action selection for exploration
def choose_action_exploration(state_index, epsilon=0.3):
    """Choose action with epsilon-greedy exploration"""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return torch.argmax(policy_net(one_hot(states[state_index]))).item()

# MODIFIED: Containers for transitions (now including rewards and done flags)
input_features = []     # F
rewards = []           # R
transition_table = {}   # Store unique transitions
visited = set()         # Track (state, action) coverage

# Exploration parameters
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.001
epsilon_decay = 0.995

print("Generating feature transitions with rewards through environment interaction...")

# === Exploration & Feature Transition Building ===
for episode in range(1000): 
    state = random.choice(states)
    state_index = states.index(state)
    #print(f"\nEpisode {episode + 1}, Starting state: {state}")

    for step_num in range(maxSteps):
        # Choose action for exploration
        action = choose_action_exploration(state_index, epsilon)
        action_str = actions[action]

        # Take step in environment (with stochasticity)
        next_state = step(state, action_str)
        next_state_index = states.index(next_state)

        #print("Next state:", next_state)

        # MODIFIED: Calculate reward and terminal flag
        reward = get_reward(next_state)
        current_reward = get_reward(state)
        done = is_terminal(next_state)

        # Extract features from V network
        state_vec = one_hot(state)
        next_state_vec = one_hot(next_state)
        
        state_features = v_network.extract_features(state_vec).detach().numpy()
        next_state_features = v_network.extract_features(next_state_vec).detach().numpy()

        # One-hot encode the action
        action_onehot = np.zeros(num_actions)
        action_onehot[action] = 1

        # Store transition data with rewards and done flags
        input_feature = np.concatenate([state_features, action_onehot])
        input_features.append(state_features)
        rewards.append(current_reward)

        # Store in transition table (for coverage tracking)
        transition_table[(tuple(state_features), action)] = (tuple(next_state_features), reward, done)

        # Track coverage
        visited.add((state, action))

        # Update current state
        state = next_state
        state_index = next_state_index

        # Early termination conditions
        if done:  # Stop episode if reached terminal state
            break

    # Decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print progress
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode+1}, Epsilon: {epsilon:.3f}, "
              f"Transitions collected: {len(input_features)}, "
              f"Unique (s,a): {len(visited)} / {num_states * num_actions}")

# MODIFIED: Convert to tensors including rewards and done flags
input_features = torch.tensor(np.array(input_features), dtype=torch.float32)
rewards = torch.tensor(np.array(rewards), dtype=torch.float32)

dataset = {
    'input_features': input_features,
    'rewards': rewards,
}

combined = torch.cat([input_features, rewards.unsqueeze(1)], dim=1)  # [N, F+1]
unique_rows = torch.unique(combined, dim=0)

torch.save(unique_rows, 'unique_data.pt')

torch.save(dataset, 'current_rewards.pt')
print("Complete dataset saved to 'current_rewards.pt'")

loaded_data = torch.load('unique_data.pt')
# Check if the saved unique data is loaded correctly
print(f"Unique data shape: {loaded_data.shape}")
print(f"Sample unique row: {loaded_data[0]}")

unique_dataset = {
    'input_features': loaded_data[:, :-1],  # All but last column
    'rewards': loaded_data[:, -1]            # Last column is rewards
}

torch.save(unique_dataset, 'unique_rewards.pt')

# Function to load the complete dataset
def load_transition_dataset(filename='unique_rewards.pt'):
    """
    Load the complete transition dataset
    
    Returns:
        dict with keys: 'input_features', 'rewards'
    """
    return torch.load(filename)

# Example usage:
print("\n=== Dataset Loading Example ===")
loaded_dataset = load_transition_dataset()
print(f"Loaded dataset keys: {list(loaded_dataset.keys())}")
print(f"Input features shape: {loaded_dataset['input_features'].shape}")
print(f"Rewards shape: {loaded_dataset['rewards'].shape}")
print(f"Sample reward: {loaded_dataset['rewards'][0].item()}")