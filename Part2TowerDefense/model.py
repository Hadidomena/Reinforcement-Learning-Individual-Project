import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import constants as c
import numpy as np

class TowerDefenseQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Deeper network with batch normalization for better training
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.linear4 = nn.Linear(hidden_size // 2, output_size)
          # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Handle both single and batch inputs
        single_input = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            single_input = True
            
        # Use batch norm only during training with multiple samples
        # During inference or with a single sample, use eval mode
        if single_input:
            # Store original state
            bn1_training = self.bn1.training
            bn2_training = self.bn2.training
            bn3_training = self.bn3.training
            
            # Set batch norm layers to eval mode temporarily
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            
            # Forward pass without gradients for single inputs
            with torch.no_grad():
                x = F.relu(self.bn1(self.linear1(x)))
                x = F.relu(self.bn2(self.linear2(x)))
                x = F.relu(self.bn3(self.linear3(x)))
                x = self.linear4(x)
                
            # Restore original state after forward pass
            self.bn1.training = bn1_training
            self.bn2.training = bn2_training
            self.bn3.training = bn3_training
        else:
            # Normal training path with multiple samples
            x = F.relu(self.bn1(self.linear1(x)))
            x = self.dropout(x) if self.training else x
            x = F.relu(self.bn2(self.linear2(x)))
            x = self.dropout(x) if self.training else x
            x = F.relu(self.bn3(self.linear3(x)))
            x = self.linear4(x)
        
        # Remove batch dimension if input was a single sample
        if single_input:
            x = x.squeeze(0)
            
        return x

    def save(self, file_name='td_model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class TowerDefenseTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.lr = lr
        self.gamma = gamma  # Discount factor for future rewards
        self.model = model
        
        # Use Adam with weight decay for better regularization
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=self.lr,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler to reduce LR over time
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=100,  # Reduce LR every 100 training steps
            gamma=0.95      # Multiply LR by 0.95
        )
        
        # Huber loss (smooth L1) is more robust to outliers than MSE
        self.criterion = nn.SmoothL1Loss()
        
        # Add batch accumulation system
        self.accumulated_states = []
        self.accumulated_actions = []
        self.accumulated_rewards = []
        self.accumulated_next_states = []
        self.accumulated_dones = []
        self.min_batch_size = 8  # Minimum batch size for BatchNorm to work properly
        self.max_batch_size = 64  # Maximum batch size for memory efficiency

    def get_state(self, world, enemy_group, turret_group):
        """Convert game state to input tensor"""
        state = [
            world.money / 1000.0,  # Normalize money
            world.health / 100.0,  # Normalize health
            world.level / c.TOTAL_LEVELS,  # Use constant instead of world attribute
            len(enemy_group) / 20.0,  # Normalize enemy count
            len(turret_group) / 10.0,  # Normalize turret count
        ]

        # Add enemy positions
        enemy_positions = torch.zeros(20, 2)  # Max 20 enemies
        for i, enemy in enumerate(enemy_group):
            if i < 20:
                enemy_positions[i] = torch.tensor([
                    enemy.pos.x / c.SCREEN_WIDTH,
                    enemy.pos.y / c.SCREEN_HEIGHT
                ])
        state.extend(enemy_positions.flatten().tolist())

        # Add turret information
        turret_positions = torch.zeros(10, 3)  # Max 10 turrets
        for i, turret in enumerate(turret_group):
            if i < 10:
                turret_positions[i] = torch.tensor([
                    turret.tile_x / c.COLS,
                    turret.tile_y / c.ROWS,
                    turret.upgrade_level / c.TURRET_LEVELS                ])
        state.extend(turret_positions.flatten().tolist())

        return state
        
    def train_step(self, state, action, reward, next_state, done):
        try:
            # Convert to numpy first, with error handling
            try:
                state = np.array(state, dtype=np.float32)
                next_state = np.array(next_state, dtype=np.float32)
                action = np.array(action, dtype=np.int64)
                reward = np.array(reward, dtype=np.float32)
            except Exception as e:
                print(f"Error converting to numpy: {e}")
                return 0
            
            # Then convert to tensor with proper device management
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.tensor(state, dtype=torch.float, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            action = torch.tensor(action, dtype=torch.long, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)

            # Ensure we're always dealing with batched inputs
            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done, )
            
            # Check if we have a large enough batch already (from mini_sample in train_long_memory)
            batch_ready = state.shape[0] >= self.min_batch_size
            
            # If we received a single sample (from short memory) or small batch, accumulate it
            # But if we received a large batch (from long memory), process it directly
            if not batch_ready:
                # Accumulate samples
                for i in range(state.shape[0]):
                    self.accumulated_states.append(state[i].clone())
                    self.accumulated_actions.append(action[i].clone())
                    self.accumulated_rewards.append(reward[i].clone())
                    self.accumulated_next_states.append(next_state[i].clone())
                    self.accumulated_dones.append(done[i])
                
                # If we still don't have enough samples after accumulating, return early
                if len(self.accumulated_states) < self.min_batch_size:
                    return 0.0
                
                # If we've accumulated enough samples, create a batch
                batch_state = torch.stack(self.accumulated_states[:self.max_batch_size])
                batch_action = torch.stack(self.accumulated_actions[:self.max_batch_size])
                batch_reward = torch.stack(self.accumulated_rewards[:self.max_batch_size])
                batch_next_state = torch.stack(self.accumulated_next_states[:self.max_batch_size])
                batch_done = self.accumulated_dones[:self.max_batch_size]
                
                # Clear the used samples from accumulation buffer (keep any extras)
                self.accumulated_states = self.accumulated_states[self.max_batch_size:] 
                self.accumulated_actions = self.accumulated_actions[self.max_batch_size:]
                self.accumulated_rewards = self.accumulated_rewards[self.max_batch_size:]
                self.accumulated_next_states = self.accumulated_next_states[self.max_batch_size:]
                self.accumulated_dones = self.accumulated_dones[self.max_batch_size:]
                
                # Replace incoming single sample with accumulated batch
                state = batch_state
                action = batch_action
                reward = batch_reward
                next_state = batch_next_state
                done = batch_done
                print(f"Processing accumulated batch of size {state.shape[0]}")
            
            # Clear memory before forward passes
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Set model to training mode explicitly
            self.model.train()
            
            # Predict Q-values with current state
            pred = self.model(state)  # This should now always get a decently sized batch

            # Calculate targets more efficiently - avoid redundant forward passes
            with torch.no_grad():
                # Get next state predictions in one batch
                next_state_preds = self.model(next_state)
                
                target = pred.clone()
                for idx in range(len(done)):
                    Q_new = reward[idx]
                    if not done[idx]:
                        Q_new = reward[idx] + self.gamma * torch.max(next_state_preds[idx])
                    
                    # Handle various action tensor shapes
                    if len(action[idx].shape) > 0:
                        action_idx = torch.argmax(action[idx]).item()
                    else:
                        action_idx = action[idx].item()
                    
                    target[idx][action_idx] = Q_new
            
            # Compute loss and apply gradients with proper error handling
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Step the learning rate scheduler (only after a certain number of steps to avoid excessive decay)
            if hasattr(self, 'step_count'):
                self.step_count += 1
                if self.step_count % 100 == 0:  # Update LR every 100 steps
                    self.scheduler.step()
            else:
                self.step_count = 1
            
            # Return loss for logging/debugging
            return loss.item()
        except Exception as e:
            print(f"Error in train_step: {e}")
            return 0.0