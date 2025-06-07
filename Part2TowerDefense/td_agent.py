import torch
import random
import numpy as np
from collections import deque
from model import TowerDefenseQNet, TowerDefenseTrainer
import constants as c
import os

MAX_MEMORY = 100_000  # Increased memory
BATCH_SIZE = 128      # Larger batch size for better learning
LR = 0.0003          # Better learning rate for tower defense
PRIORITIZED_REPLAY = True  # Use prioritized experience replay

class TowerDefenseAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80  # Start with high exploration
        self.epsilon_min = 5
        self.epsilon_decay = 0.997  # Slower decay for longer exploration
        self.gamma = 0.95  # Higher discount factor for longer-term thinking
        
        # Experience replay with priorities
        self.memory = deque(maxlen=MAX_MEMORY)
        self.priorities = deque(maxlen=MAX_MEMORY)  # For prioritized replay
        
        # Calculate action space
        self.action_size = c.ROWS * c.COLS + 20  # Space for turret upgrades
        self.state_size = 75
        
        print(f"Action space size: {self.action_size}")
        self.model = TowerDefenseQNet(self.state_size, 512, self.action_size)  # Larger network
        self.trainer = TowerDefenseTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Target network for stable learning
        self.target_model = TowerDefenseQNet(self.state_size, 512, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_frequency = 50  # More frequent updates
        self.update_counter = 0
        
        # Tracking metrics for introspection
        self.total_reward = 0
        self.reward_history = []  # Track rewards over episodes
        self.action_distribution = np.zeros(self.action_size)  # Track action frequencies

    def get_state(self, world, enemy_group, turret_group):
        """Get current state with increased debug info"""
        state = self.trainer.get_state(world, enemy_group, turret_group)
        
        # Debug info periodically
        if hasattr(self, 'n_games') and self.n_games % 10 == 0 and hasattr(self, 'debug_counter') and self.debug_counter % 50 == 0:
            print(f"State shape: {state.shape}, Range: [{state.min().item():.2f}, {state.max().item():.2f}]")
            
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        self.debug_counter += 1
            
        return state

    def remember(self, state, action, reward, next_state, done):
        # Convert to numpy arrays for consistent storage
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        # Calculate priority (absolute reward as simple priority)
        priority = abs(reward) + 0.01  # Small constant to ensure non-zero priority
        self.priorities.append(priority)
        
        # Track action distribution
        action_idx = np.argmax(action)
        self.action_distribution[action_idx] += 1
        
        # Log extreme rewards for debugging
        if abs(reward) > 20:
            print(f"High reward detected: {reward:.1f}")
                
    def train_long_memory(self):
        """Train on a batch of experiences from memory with prioritized replay"""
        if len(self.memory) < BATCH_SIZE:
            return None
        
        try:
            if PRIORITIZED_REPLAY and self.priorities:
                # Convert priorities to sampling probabilities
                probs = np.array(self.priorities) / sum(self.priorities)
                # Sample batch indices according to priorities
                indices = np.random.choice(len(self.memory), 
                                         size=min(BATCH_SIZE, len(self.memory)),
                                         replace=False, 
                                         p=probs)
                batch = [self.memory[idx] for idx in indices]
            else:
                # Standard uniform sampling
                batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
                
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            
            # Train the model
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
            
            # Update target network periodically
            self.update_counter += 1
            if self.update_counter % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Target network updated (step {self.update_counter})")
                
            return loss
        except Exception as e:
            print(f"Error in train_long_memory: {e}")
            return None
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single experience with error handling"""
        try:
            # Only train on significant experiences to reduce noise
            if abs(reward) < 0.01 and random.random() < 0.7:
                return 0.0  # Skip training on very small rewards most of the time
                
            # Track total reward for this episode
            self.total_reward += reward
                
            # Do the actual training
            loss = self.trainer.train_step(state, action, reward, next_state, done)
            
            # Log occasional stats during training
            if hasattr(self, 'debug_counter') and self.debug_counter % 500 == 0:
                action_idx = torch.argmax(action).item() if isinstance(action, torch.Tensor) else np.argmax(action)
                print(f"Training step - Reward: {reward:.2f}, Action: {action_idx}, Loss: {loss:.4f}")
                
            return loss
        except Exception as e:
            print(f"Error in train_short_memory: {e}")
            return 0.0
    def get_action(self, state, valid_positions):
        """Get action using improved epsilon-greedy policy with strategic exploration"""
        # Improved epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        final_move = np.zeros(self.action_size)
        
        try:
            # Strategic exploration with decreasing randomness
            if random.randint(0, 100) < self.epsilon:
                # Explore with different strategies based on game progression
                if self.n_games < 50:  # Early training: focus on turret placement
                    if valid_positions and random.random() < 0.9:
                        # Strong preference for valid turret placements
                        move = random.choice(valid_positions)
                    elif random.random() < 0.5:  # Some upgrade actions
                        move = random.randint(c.ROWS * c.COLS, min(c.ROWS * c.COLS + 10, self.action_size - 1))
                    else:
                        move = random.randint(0, c.ROWS * c.COLS - 1)
                else:  # Later training: more balanced approach
                    # Analyze current state to make better random decisions
                    if valid_positions and len(valid_positions) > 5 and random.random() < 0.7:
                        # Prefer placing turrets when many valid positions available
                        move = random.choice(valid_positions)
                    elif len(valid_positions) <= 5 and random.random() < 0.6:
                        # More upgrades when board is filling up
                        move = random.randint(c.ROWS * c.COLS, self.action_size - 1)
                    else:
                        # Either place turret or upgrade based on current situation
                        if valid_positions:
                            if random.random() < 0.7:
                                move = random.choice(valid_positions)
                            else:
                                move = random.randint(c.ROWS * c.COLS, self.action_size - 1)
                        else:
                            move = random.randint(c.ROWS * c.COLS, self.action_size - 1)
            else:
                # Exploitation: use model prediction with improved decision making
                self.model.eval()
                
                with torch.no_grad():
                    if isinstance(state, torch.Tensor):
                        state_tensor = state
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32)
                    
                    # Get Q-values from model
                    q_values = self.model(state_tensor)
                    
                    # Apply sophisticated masking based on game state
                    # 1. Mask invalid turret placements
                    if valid_positions:
                        # Create mask for valid positions using proper broadcasting
                        valid_mask = torch.full_like(q_values[:c.ROWS * c.COLS], float('-inf'))
                        for pos in valid_positions:
                            valid_mask[pos] = 0
                        q_values[:c.ROWS * c.COLS] += valid_mask
                    
                    # 2. Add strategic biases based on game progression
                    if hasattr(self, 'n_games') and self.n_games > 0:
                        # Calculate how many turrets we can place with current money
                        max_affordable_turrets = min(5, len(valid_positions))  # Cap at reasonable number
                        
                        # In early game, slightly prefer placing turrets strategically
                        if max_affordable_turrets > 2 and self.n_games < 100:
                            # Give a small boost to turret placement actions if we can afford multiple turrets
                            for pos in valid_positions:
                                q_values[pos] += 0.2
                    
                    # Choose the best action after all adjustments
                    move = torch.argmax(q_values).item()
            
            # Bounds checking to prevent index errors
            move = max(0, min(move, self.action_size - 1))
            final_move[move] = 1
            
            # Periodically log action distribution for analysis
            if hasattr(self, 'n_games') and self.n_games > 0 and self.n_games % 10 == 0:
                if hasattr(self, 'debug_action_counter'):
                    self.debug_action_counter += 1
                    if self.debug_action_counter % 100 == 0:
                        placement_actions = np.sum(self.action_distribution[:c.ROWS * c.COLS])
                        upgrade_actions = np.sum(self.action_distribution[c.ROWS * c.COLS:])
                        total_actions = np.sum(self.action_distribution)
                        if total_actions > 0:
                            print(f"Action distribution - Placements: {placement_actions/total_actions:.1%}, Upgrades: {upgrade_actions/total_actions:.1%}")
                else:
                    self.debug_action_counter = 0
            
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Default action with smarter fallback
            if valid_positions:
                move = random.choice(valid_positions)
                final_move[move] = 1
            elif len(self.turret_group) > 0:  # If we have turrets, try upgrading
                move = c.ROWS * c.COLS + random.randint(0, min(len(self.turret_group), 10) - 1)
                final_move[move] = 1
            else:
                final_move[0] = 1  # Just do something valid
            
        return final_move
        
    def save_model(self, file_name='td_model.pth'):
        """Save the trained model"""
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        
        # Save both main and target models
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'epsilon': self.epsilon,
            'n_games': self.n_games
        }, file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name='td_model.pth'):
        """Load a trained model"""
        model_folder_path = './models'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.n_games = checkpoint.get('n_games', 0)
            self.model.eval()
            print(f"Loaded model from {file_name}")
            print(f"Resumed at game {self.n_games} with epsilon {self.epsilon}")
        else:
            print(f"No model found at {file_name}")
            
    def set_evaluation_mode(self):
        """Set agent to evaluation mode (no exploration)"""
        self.epsilon = 0
        self.model.eval()

