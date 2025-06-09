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
        
        # Create target network for stable learning
        self.target_model = TowerDefenseQNet(model.linear1.in_features, 
                                           model.linear1.out_features, 
                                           model.linear4.out_features)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()
        
        # Improved optimizer with adaptive learning rate
        self.optimizer = optim.AdamW(  # AdamW for better regularization
            model.parameters(), 
            lr=self.lr,
            weight_decay=1e-4,  # Stronger weight decay
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # More sophisticated learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=100,  # Restart every 100 steps
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6
        )
        
        # Use Huber loss for more stable training
        self.criterion = nn.SmoothL1Loss()  # Huber loss - more robust than MSE
        
        self.step_count = 0
        self.target_update_freq = 50  # More frequent target updates
        
        # Multi-step learning
        self.n_step = 3
        self.n_step_buffer = []
        
        # Double DQN
        self.double_dqn = True

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
            # Return enhanced zero state as fallback - Updated size to match new state representation
            return torch.zeros(108, dtype=torch.float32)  # 8 + 60 + 40 = 108

    def train_step(self, state, action, reward, next_state, done, weights=None):
        """Enhanced training step with Double DQN and multi-step learning"""
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
            
            # Calculate target Q-values using Double DQN
            with torch.no_grad():
                if hasattr(self, 'double_dqn') and self.double_dqn:
                    # Double DQN: use main network to select action, target network to evaluate
                    next_q_main = self.model(next_state)
                    next_actions = torch.argmax(next_q_main, dim=1)
                    next_q_target = self.target_model(next_state)
                    next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    # Standard DQN
                    next_q_values = torch.max(self.target_model(next_state), dim=1)[0]
                
                target_q_values = current_q_values.clone()
                
                for i in range(len(done)):
                    if isinstance(action[i], torch.Tensor) and len(action[i].shape) > 0:
                        action_idx = torch.argmax(action[i]).item()
                    else:
                        action_idx = action[i].item() if isinstance(action[i], torch.Tensor) else action[i]
                    
                    if done[i]:
                        target_q_values[i][action_idx] = reward[i]
                    else:
                        # Multi-step learning with higher gamma power
                        gamma_n = self.gamma ** getattr(self, 'n_step', 1)
                        target_q_values[i][action_idx] = reward[i] + gamma_n * next_q_values[i]
            
            # Compute loss with importance sampling weights
            if weights is not None:
                # Weighted loss for prioritized experience replay
                td_errors = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
                td_errors = td_errors.sum(dim=1)  # Sum over actions
                loss = (weights * td_errors).mean()
            else:
                loss = self.criterion(current_q_values, target_q_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping with adaptive norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            
            self.optimizer.step()
            
            # Step the scheduler
            if hasattr(self.scheduler, 'step') and hasattr(self, 'step_count'):
                if hasattr(self.scheduler, 'T_0'):  # CosineAnnealingWarmRestarts
                    self.scheduler.step()
                else:  # ReduceLROnPlateau
                    self.scheduler.step(loss)
            
            # Update target network
            if self.step_count % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            self.step_count += 1
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            return 0.0
    
    def adaptive_lr_step(self, loss):
        """Adaptive learning rate adjustment based on loss"""
        if hasattr(self, 'prev_loss'):
            if loss > self.prev_loss * 1.1:  # Loss increased significantly
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95
                    param_group['lr'] = max(param_group['lr'], 1e-6)
            elif loss < self.prev_loss * 0.9:  # Loss decreased significantly
                # Slightly increase learning rate                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 1.01
                    param_group['lr'] = min(param_group['lr'], 0.01)
        
        self.prev_loss = loss