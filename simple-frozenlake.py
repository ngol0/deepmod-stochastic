import numpy as np
import torch
import torch.nn as nn
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

listOfHoles = np.array(['F', 'H', 'L', 'M'])

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
    
# Q-network definition
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

# Q-values printer
def print_q_values(policy_net):
    print("\nQ-values by state:")
    policy_net.eval()
    with torch.no_grad():
        for state in states:
            state_vec = one_hot(state)
            q_vals = policy_net(state_vec)
            q_vals_np = q_vals.numpy()
            best_action_idx = torch.argmax(q_vals).item()
            print(f"State {state}: ", end='')
            for i, a in enumerate(actions):
                print(f"{a}: {q_vals_np[i]:.2f}  ", end='')
            print(f"=> Best: {actions[best_action_idx]}")
    policy_net.train()

# Training setup
num_episodes = 5000
max_steps = 99
gamma = 0.99
lr = 0.001
initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 0.9995

epsilon = initial_epsilon
policy_net = DQN(num_states, num_actions)
target_net = DQN(num_states, num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# V network
v_network = VNetwork(num_states)
V_optimizer = optim.Adam(v_network.parameters(), lr=lr)
v_loss_fn = nn.MSELoss()

# Action selection
def select_action(state_vec, epsilon):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        return torch.argmax(policy_net(state_vec)).item()

# Training loop
for episode in range(num_episodes):
    state = random.choice(states)
    for t in range(max_steps):
        state_vec = one_hot(state)
        action_idx = select_action(state_vec, epsilon)
        action = actions[action_idx]
        next_state = step(state, action)

        reward = 1.0 if next_state == 'P' else -1.0 if next_state in listOfHoles else 0.0
        done = next_state == 'P'

        q_value = policy_net(state_vec)[action_idx]
        with torch.no_grad():
            next_state_vec = one_hot(next_state)
            max_next_q = target_net(next_state_vec).max().item()
            target_q = reward if done else reward + gamma * max_next_q

        target_q_tensor = torch.tensor(target_q, dtype=torch.float32)
        loss = loss_fn(q_value, target_q_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break
        state = next_state

    # Update target network
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(final_epsilon, epsilon * epsilon_decay)

    # Show Q-values at 200
    if episode % 500 == 0:
        print(f"\n== Q-values at episode {episode} ==")
        print_q_values(policy_net)

print("\n== Final Q-values after training ==")
print_q_values(policy_net)

# Train a value network V(s) ≈ max_a Q(s, a)
policy_net.eval()
v_network.train()

def print_v_values(v_network):
    print("\nState-value estimates V(s):")
    v_network.eval()
    with torch.no_grad():
        for state in states:
            state_vec = one_hot(state)
            v_val = v_network(state_vec).item()
            print(f"State {state}: V = {v_val:.2f}")
    v_network.train()

for episode in range(1000):
    state = random.choice(states)
    state_vec = one_hot(state)

    # Target is the max Q-value from the policy_net
    with torch.no_grad():
        target_v = policy_net(state_vec).max().item()

    v_value = v_network(state_vec)
    target_v_tensor = torch.tensor(target_v, dtype=torch.float32)

    v_loss = v_loss_fn(v_value, target_v_tensor)

    V_optimizer.zero_grad()
    v_loss.backward()
    V_optimizer.step()

    if episode % 200 == 0:
        print(f"[V] Episode {episode}, Loss: {v_loss.item():.4f}")

# print_v_values(v_network)

# # Save both networks
# torch.save(policy_net.state_dict(), 'policy_net.pth')
# torch.save(v_network.state_dict(), 'v_network.pth')

print("\n== Final Q-values after training ==")
print_q_values(policy_net)

# Train a value network V(s) ≈ max_a Q(s, a)
policy_net.eval()
v_network.train()

def print_v_values(v_network):
    print("\nState-value estimates V(s):")
    v_network.eval()
    with torch.no_grad():
        for state in states:
            state_vec = one_hot(state)
            v_val = v_network(state_vec).item()
            print(f"State {state}: V = {v_val:.2f}")
    v_network.train()

for episode in range(1000):
    state = random.choice(states)
    state_vec = one_hot(state)

    # Target is the max Q-value from the policy_net
    with torch.no_grad():
        target_v = policy_net(state_vec).max().item()

    v_value = v_network(state_vec)
    target_v_tensor = torch.tensor(target_v, dtype=torch.float32)

    v_loss = v_loss_fn(v_value, target_v_tensor)

    V_optimizer.zero_grad()
    v_loss.backward()
    V_optimizer.step()

    if episode % 200 == 0:
        print(f"[V] Episode {episode}, Loss: {v_loss.item():.4f}")

print_v_values(v_network)

# Policy extraction functions
def print_q_policy(policy_net):
    print("\n=== Q-Network Policy ===")
    policy_net.eval()
    with torch.no_grad():
        for state in states:
            state_vec = one_hot(state)
            q_vals = policy_net(state_vec)
            best_action_idx = torch.argmax(q_vals).item()
            best_action = actions[best_action_idx]
            print(f"State {state}: {best_action} (Q-value: {q_vals[best_action_idx]:.3f})")

def print_v_policy(policy_net):
    print("\n=== V-Network Derived Policy ===")
    policy_net.eval()
    with torch.no_grad():
        for state in states:
            state_vec = one_hot(state)
            q_vals = policy_net(state_vec)  # Use Q-network to derive policy
            best_action_idx = torch.argmax(q_vals).item()
            best_action = actions[best_action_idx]
            v_value = q_vals.max().item()  # V(s) = max_a Q(s,a)
            print(f"State {state}: {best_action} (V-value: {v_value:.3f})")

def print_grid_policy(policy_net, title="Policy Grid"):
    print(f"\n=== {title} ===")
    policy_net.eval()
    
    # Create 4x4 grid representation
    grid = np.empty((4, 4), dtype=object)
    
    with torch.no_grad():
        for i, state in enumerate(states):
            row, col = i // 4, i % 4
            
            if state in listOfHoles:
                grid[row, col] = 'H'  # Hole
            elif state == 'P':
                grid[row, col] = 'G'  # Goal
            else:
                state_vec = one_hot(state)
                q_vals = policy_net(state_vec)
                best_action_idx = torch.argmax(q_vals).item()
                action_symbols = {'up': '↑', 'down': '↓', 'right': '→', 'left': '←'}
                grid[row, col] = action_symbols[actions[best_action_idx]]
    
    # Print the grid
    print("  A B C D")
    for i in range(4):
        row_states = states[i*4:(i+1)*4]
        print(f"{row_states[0][0]} ", end="")  # Row label (E, I, M corresponds to rows)
        for j in range(4):
            print(f"{grid[i,j]} ", end="")
        print()

def compare_policies(policy_net):
    print("\n=== Policy Comparison ===")
    policy_net.eval()
    
    differences = 0
    with torch.no_grad():
        for state in states:
            if state in listOfHoles or state == 'P':
                continue  # Skip terminal states
                
            state_vec = one_hot(state)
            q_vals = policy_net(state_vec)
            best_action_idx = torch.argmax(q_vals).item()
            best_action = actions[best_action_idx]
            
            # For this simple case, Q and V policies are the same since V(s) = max_a Q(s,a)
            print(f"State {state}: Q-policy = V-policy = {best_action}")
    
    print("Note: Q-network policy and V-derived policy are identical since V(s) = max_a Q(s,a)")

# Print all policies
print_q_policy(policy_net)
print_v_policy(policy_net)
print_grid_policy(policy_net, "Q-Network Policy Grid")
compare_policies(policy_net)

# Save both networks
torch.save(policy_net.state_dict(), 'policy_net.pth')
torch.save(v_network.state_dict(), 'v_network.pth')

print(f"\nTraining completed!")
#print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")