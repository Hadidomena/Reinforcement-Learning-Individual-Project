import torch
import random
import numpy as np
from collections import deque
from model import TowerDefenseQNet, TowerDefenseTrainer
import constants as c
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class TowerDefenseAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Calculate action space
        self.action_size = c.ROWS * c.COLS + 10  # All tiles + up to 10 upgrade options
        self.state_size = 75  # base stats(5) + enemy positions(40) + turret info(30)
        
        print(f"Action space size: {self.action_size}")
        self.model = TowerDefenseQNet(self.state_size, 256, self.action_size)
        self.trainer = TowerDefenseTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, world, enemy_group, turret_group):
        return self.trainer.get_state(world, enemy_group, turret_group)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Always ensure sufficient batch size for batch normalization
        min_batch_size = 8  # Match with the trainer's min_batch_size
        
        if len(self.memory) > BATCH_SIZE:
            # Use the full batch size when we have enough memories
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        elif len(self.memory) > min_batch_size:
            # Use what we have if it's at least the minimum
            mini_sample = random.sample(self.memory, len(self.memory))
        else:
            # Not enough data yet
            print(f"Skipping long memory training - only {len(self.memory)} samples")
            return
            
        # Prepare the batch
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Explicitly set model to training mode before batch processing
        self.model.train()
        
        # Process the entire batch at once
        self.trainer.train_step(states, actions, rewards, next_states, dones)    
    def train_short_memory(self, state, action, reward, next_state, done):
        # Short memory training will accumulate samples in the trainer until
        # enough are gathered for a proper batch
        try:
            # Ensure model is in training mode for collecting experiences
            self.model.train()
            
            # For single experiences, we don't expect immediate training, just accumulation
            loss = self.trainer.train_step(state, action, reward, next_state, done)
            
            # Optional debugging
            if loss > 0:
                return loss  # Only returns non-zero when actual training occurred
        except Exception as e:
            print(f"Error in train_short_memory: {e}")
            
        return 0.0
        
    def get_action(self, state, valid_positions):
        # Enhanced exploration-exploitation balance
        # Start with more exploration and gradually reduce
        self.epsilon = max(0, 100 - self.n_games * 0.5)  # More gradual decline
        final_move = np.zeros(self.action_size)
        
        try:
            if random.randint(0, 200) < self.epsilon:
                # Exploration with strategic bias
                if valid_positions:
                    # Encourage placing turrets in the early game
                    early_game = self.n_games < 30
                    place_turret_prob = 0.8 if early_game else 0.6
                    
                    if random.random() < place_turret_prob:
                        # Place turret
                        move = random.choice(valid_positions)
                    else:
                        # Upgrade existing turret (if any)
                        max_idx = min(c.ROWS * c.COLS + min(10, len(valid_positions)), self.action_size - 1)
                        move = random.randint(c.ROWS * c.COLS, max_idx)
                else:
                    # No valid positions for new turrets, focus on upgrades
                    max_idx = min(c.ROWS * c.COLS + 10, self.action_size - 1)
                    move = random.randint(c.ROWS * c.COLS, max_idx)
            else:
                # Exploitation with occasional random action for continuous exploration
                if random.random() < 0.05:  # 5% chance of random action even in exploitation mode
                    if valid_positions and random.random() < 0.5:
                        move = random.choice(valid_positions)
                    else:
                        max_idx = min(c.ROWS * c.COLS + min(10, len(valid_positions)), self.action_size - 1)
                        move = random.randint(c.ROWS * c.COLS, max_idx)
                else:
                    # Normal exploitation
                    state0 = torch.tensor(state, dtype=torch.float)
                    prediction = self.model(state0)
                    move = torch.argmax(prediction).item()
            
            # Strict bounds checking
            if move >= self.action_size:
                print(f"Warning: Action {move} out of bounds, clamping to {self.action_size-1}")
                move = self.action_size - 1
            if move < 0:
                move = 0
                
            final_move[move] = 1
        except Exception as e:
            print(f"Error in get_action: {e}")
            # Default to do-nothing action
            final_move[-1] = 1
            
        return final_move
        
    def save_model(self, file_name='td_model.pth'):
        """Save the trained model"""
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name='td_model.pth'):
        """Load a trained model"""
        model_folder_path = './models'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.model.load_state_dict(torch.load(file_name))
            self.model.eval()
            print(f"Loaded model from {file_name}")