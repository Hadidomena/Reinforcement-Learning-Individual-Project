import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class TowerDefenseQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Use LayerNorm instead of BatchNorm for single-sample compatibility
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # LayerNorm works with single samples
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.linear4 = nn.Linear(hidden_size // 2, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle both single and batch inputs consistently
        original_shape = x.shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Forward pass - same for training and inference
        x = F.relu(self.ln1(self.linear1(x)))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.ln2(self.linear2(x)))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.ln3(self.linear3(x)))
        x = self.linear4(x)
        
        # Restore original shape
        if len(original_shape) == 1:
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
        self.gamma = gamma
        self.model = model
        
        # Simpler optimizer setup
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=self.lr,
            weight_decay=1e-4
        )
        
        # Less aggressive learning rate scheduling
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=500,  # Less frequent updates
            gamma=0.9
        )
        
        # Use MSE loss for stability
        self.criterion = nn.MSELoss()
        
        # Remove complex batch accumulation - keep it simple
        self.step_count = 0

    def get_state(self, world, enemy_group, turret_group):
        """Convert game state to input tensor with better normalization"""
        try:
            state = [
                world.money / 1000.0,  # Normalize money
                world.health / 100.0,  # Normalize health
                min(1.0, world.level / 50.0),  # Normalize level
                len(enemy_group) / 20.0,  # Normalize enemy count
                len(turret_group) / 15.0,  # Normalize turret count
            ]

            # Add enemy positions (fixed size)
            enemy_positions = torch.zeros(20, 2)
            for i, enemy in enumerate(enemy_group):
                if i < 20:
                    enemy_positions[i] = torch.tensor([
                        enemy.pos.x / 800.0,  # Assuming screen width
                        enemy.pos.y / 600.0   # Assuming screen height
                    ])
            state.extend(enemy_positions.flatten().tolist())

            # Add turret information (fixed size)
            turret_positions = torch.zeros(10, 3)
            for i, turret in enumerate(turret_group):
                if i < 10:
                    turret_positions[i] = torch.tensor([
                        turret.tile_x / 20.0,  # Normalize by grid size
                        turret.tile_y / 15.0,  # Normalize by grid size
                        getattr(turret, 'upgrade_level', 1) / 4.0  # Normalize upgrade level
                    ])
            state.extend(turret_positions.flatten().tolist())

            return torch.tensor(state, dtype=torch.float32)
        except Exception as e:
            print(f"Error in get_state: {e}")
            # Return zero state as fallback
            return torch.zeros(75, dtype=torch.float32)
        
    def train_step(self, state, action, reward, next_state, done):
        """Simplified training step"""
        try:
            # Convert inputs to tensors
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.long)
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor(reward, dtype=torch.float32)
            
            # Ensure batch dimension
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                next_state = next_state.unsqueeze(0)
                action = action.unsqueeze(0)
                reward = reward.unsqueeze(0)
                done = [done] if not isinstance(done, list) else done
            
            # Set model to training mode
            self.model.train()
            
            # Forward pass
            current_q_values = self.model(state)
            
            # Calculate target Q-values
            with torch.no_grad():
                next_q_values = self.model(next_state)
                target_q_values = current_q_values.clone()
                
                for i in range(len(done)):
                    if isinstance(action[i], torch.Tensor) and len(action[i].shape) > 0:
                        action_idx = torch.argmax(action[i]).item()
                    else:
                        action_idx = action[i].item() if isinstance(action[i], torch.Tensor) else action[i]
                    
                    if done[i]:
                        target_q_values[i][action_idx] = reward[i]
                    else:
                        target_q_values[i][action_idx] = reward[i] + self.gamma * torch.max(next_q_values[i])
            
            # Compute loss and update
            loss = self.criterion(current_q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update learning rate occasionally
            self.step_count += 1
            if self.step_count % 500 == 0:
                self.scheduler.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            return 0.0