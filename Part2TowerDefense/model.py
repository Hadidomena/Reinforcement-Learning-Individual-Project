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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.ln1(self.linear1(x)))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.ln2(self.linear2(x)))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.ln3(self.linear3(x)))
        x = self.linear4(x)
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
        
        # Create target network for stable learning
        self.target_model = TowerDefenseQNet(model.linear1.in_features, 
                                           model.linear1.out_features, 
                                           model.linear4.out_features)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()
        
        # Improved optimizer with gradient clipping
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=self.lr,
            weight_decay=1e-5,  # Reduced weight decay
            eps=1e-8
        )
          # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.8,
            patience=500
        )
        
        # Use Huber loss for more stable training
        self.criterion = nn.SmoothL1Loss()  # Huber loss - more robust than MSE
        
        self.step_count = 0
        self.target_update_freq = 100  # Update target network every 100 steps

    def get_state(self, world, enemy_group, turret_group):
        """Enhanced state representation with more game dynamics information"""
        try:
            # Basic game state
            state = [
                world.money / 1000.0,  
                world.health / 100.0,  
                min(1.0, world.level / 50.0),  
                len(enemy_group) / 20.0,  
                len(turret_group) / 15.0,
                # Add more dynamic information
                world.selected_turret if hasattr(world, 'selected_turret') else 0,
                len([e for e in enemy_group if e.health > 0]) / 20.0,  # Healthy enemies
                sum([t.upgrade_level if hasattr(t, 'upgrade_level') else 1 for t in turret_group]) / max(1, len(turret_group)),  # Avg upgrade level
            ]

            # Enemy positions with velocity information
            enemy_positions = torch.zeros(20, 3)  # x, y, health_ratio
            for i, enemy in enumerate(enemy_group):
                if i < 20:
                    enemy_positions[i] = torch.tensor([
                        enemy.pos.x / 800.0,  
                        enemy.pos.y / 600.0,   
                        enemy.health / enemy.max_health if hasattr(enemy, 'max_health') else 1.0
                    ])
            state.extend(enemy_positions.flatten().tolist())

            # Enhanced turret information
            turret_positions = torch.zeros(10, 4)  # x, y, upgrade_level, range
            for i, turret in enumerate(turret_group):
                if i < 10:
                    turret_positions[i] = torch.tensor([
                        turret.tile_x / 20.0,  
                        turret.tile_y / 15.0,  
                        getattr(turret, 'upgrade_level', 1) / 4.0,
                        getattr(turret, 'range', 50) / 100.0  # Normalized range
                    ])
            state.extend(turret_positions.flatten().tolist())

            return torch.tensor(state, dtype=torch.float32)
        except Exception as e:
            print(f"Error in get_state: {e}")
            return torch.zeros(108, dtype=torch.float32) 
        
    def train_step(self, state, action, reward, next_state, done, weights=None):
        """Simplified training step with optional importance sampling weights"""
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
                next_q_values = self.target_model(next_state)
                target_q_values = current_q_values.clone()
                
                for i in range(len(done)):
                    if isinstance(action[i], torch.Tensor) and len(action[i].shape) > 0:
                        action_idx = torch.argmax(action[i]).item()
                    else:
                        action_idx = action[i].item() if isinstance(action[i], torch.Tensor) else action[i]
                    
                    # Ensure action_idx is a valid integer
                    if isinstance(action_idx, (np.ndarray, torch.Tensor)):
                        action_idx = action_idx.item()
                    action_idx = int(action_idx)
                    
                    # Ensure action_idx is within valid range
                    action_idx = max(0, min(action_idx, target_q_values.shape[1] - 1))
                    
                    if done[i]:
                        target_q_values[i][action_idx] = reward[i]
                    else:
                        target_q_values[i][action_idx] = reward[i] + self.gamma * torch.max(next_q_values[i])
            
            # Compute loss and update
            loss = self.criterion(current_q_values, target_q_values)
            
            # Apply importance sampling weights if provided
            if weights is not None:
                element_wise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
                loss = (element_wise_loss.mean(dim=1) * weights).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update target network
            if self.step_count % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Update learning rate occasionally
            self.step_count += 1
            if self.step_count % 500 == 0:
                self.scheduler.step(loss)
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            return 0.0