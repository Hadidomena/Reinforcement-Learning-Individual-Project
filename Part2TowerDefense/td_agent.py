import torch
import random
import numpy as np
from collections import deque
from model import TowerDefenseQNet, TowerDefenseTrainer
from reward_shaping import AdvancedRewardShaper
import constants as c
import os
import math
import torch.nn.functional as F

# STABILITY FIX: Optimized hyperparameters for stable learning
MAX_MEMORY = 100_000   # INCREASED: Larger memory for better stability
BATCH_SIZE = 128       # INCREASED: Larger batches for more stable updates  
LR = 0.00005           # FURTHER REDUCED: Very low learning rate for fine-tuning
PRIORITIZED_REPLAY = True  # Keep using prioritized experience replay
NOISY_NETWORKS = False  # DISABLED: Reduce noise for more stable learning

class TowerDefenseAgent:    
    def __init__(self):
        self.n_games = 0
        # STABILITY FIX: Conservative exploration strategy  
        self.epsilon = 5   # REDUCED: Much lower starting epsilon
        self.epsilon_min = 0.1  # REDUCED: Very low minimum for exploitation
        self.epsilon_decay = 0.9998  # SLOWER: Very gradual decay
        self.epsilon_cycle_length = 200  # Longer cycles for stability
        self.base_epsilon = 0.5  # REDUCED: Lower base for resets
        self.gamma = 0.99  # Standard discount factor
        
        self.best_recent_score = 0
        self.performance_window = 100
        self.stable_performance_threshold = 0.85
        
        self.difficulty_level = 1
        self.games_per_difficulty = 50  # INCREASED: More games per difficulty
        self.performance_threshold = 10  # INCREASED: Higher threshold        # Experience replay with conservative priorities
        self.memory = deque(maxlen=MAX_MEMORY)
        self.priorities = deque(maxlen=MAX_MEMORY)
        self.elite_memory = deque(maxlen=20000)
        self.elite_threshold = 10
        
        self.alpha = 0.4
        self.beta = 0.3
        self.beta_increment = 0.000005
        
        self.action_size = c.ROWS * c.COLS + 20
        self.state_size = 108
        print(f"Action space size: {self.action_size}")
        print(f"State space size: {self.state_size}")
        
        self.model = TowerDefenseQNet(self.state_size, 512, self.action_size)
        self.trainer = TowerDefenseTrainer(self.model, lr=LR, gamma=self.gamma)
          # Multi-step learning and double DQN - more stable
        self.target_update_frequency = 100  # INCREASED: Less frequent updates for stability
        self.multi_step_n = 2  # REDUCED: Simpler learning
        self.double_dqn = True  # Use Double DQN for reduced overestimation
        self.steps_done = 0
          # Tracking metrics for introspection
        self.total_reward = 0
        self.reward_history = []
        self.action_distribution = np.zeros(self.action_size)
        self.running_loss = 0
        self.ema_loss = None  # Exponential moving average loss
        self.turret_group = []  # Will be updated during gameplay
          # Performance tracking for curriculum learning
        self.recent_scores = []
        self.plateau_counter = 0
        self.last_improvement = 0
        
        self.lr_decay_patience = 50
        self.lr_decay_factor = 0.7
        self.min_lr = 1e-6
          # Advanced reward shaping
        self.reward_shaper = AdvancedRewardShaper()
        self.action_history = deque(maxlen=100)
          # Anti-catastrophic forgetting mechanisms
        self.knowledge_consolidation = True
        self.stable_policy_buffer = deque(maxlen=1000)
        self.performance_regression_counter = 0
        self.last_high_score = 0
        self.consolidation_frequency = 20
        
        self.score_history = deque(maxlen=100)
        self.level_history = deque(maxlen=100)
        self.regression_threshold = 0.7
        
        self.performance_history = deque(maxlen=100)
        self.strategy_history = deque(maxlen=50)
        self.breakthrough_attempts = 0
        
        self.exploration_momentum = 0.0
        self.strategy_adaptation_rate = 0.1
        
        # Breakthrough mode tracking
        self.breakthrough_mode = False
        self.games_since_breakthrough = 0

    def get_state(self, world, enemy_group, turret_group):
        self.turret_group = turret_group
        state = self.trainer.get_state(world, enemy_group, turret_group)
          # Debug info periodically
        if hasattr(self, 'n_games') and self.n_games % 10 == 0 and hasattr(self, 'debug_counter') and self.debug_counter % 50 == 0:
            print(f"State shape: {state.shape}, Range: [{state.min().item():.2f}, {state.max().item():.2f}]")
            
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        self.debug_counter += 1
            
        return state

    def remember(self, state, action, reward, next_state, done, game_stats=None):
        try:
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()            
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu().numpy()
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
            if np.isnan(np.sum(state)) or np.isnan(np.sum(next_state)):
                print("Warning: NaN detected in state. Skipping this experience.")
                return
                
            if np.max(np.abs(state)) > 100 or np.max(np.abs(next_state)) > 100:
                print(f"Warning: Extreme values in state: {np.max(np.abs(state)):.1f}. Clipping.")
                state = np.clip(state, -100, 100)
                next_state = np.clip(next_state, -100, 100)
            
            # Enhanced reward shaping
            if game_stats:
                action_type = "upgrade" if np.argmax(action) >= c.ROWS * c.COLS else "place"
                shaped_reward = self.reward_shaper.calculate_reward(state, action_type, game_stats)
                curiosity_reward = self.reward_shaper.get_curiosity_reward(state, action, next_state)
                curriculum_bonus = self.reward_shaper.get_curriculum_bonus(
                    self.difficulty_level, game_stats.get('score', 0)
                )
                total_reward = reward + shaped_reward + curiosity_reward + curriculum_bonus
                
                # Log significant reward changes
                if abs(shaped_reward) > 5:
                    print(f"Shaped reward: {shaped_reward:.1f} (base: {reward:.1f}, curiosity: {curiosity_reward:.1f})")
            else:
                total_reward = reward
                
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
                current_q = self.model(s)
                
                # Get action index
                action_idx = np.argmax(action) if len(action.shape) > 0 else action
                
                # Calculate target Q value
                if done:
                    target_q = total_reward
                else:
                    next_q = self.trainer.target_model(ns)
                    target_q = total_reward + self.gamma * torch.max(next_q).item()
                
                # Calculate TD error (priority)
                if isinstance(action_idx, np.ndarray):
                    action_idx = action_idx.item()
                td_error = abs(target_q - current_q[0, action_idx].item())
            
            # Add small constant to prevent zero priority
            priority = (td_error + 0.1) ** self.alpha
              # Store experience with enhanced reward
            self.memory.append((state, action, total_reward, next_state, done))
            self.priorities.append(priority)
            
            if game_stats and game_stats.get('level', 0) >= self.elite_threshold:
                elite_priority = priority * 2.0  # Higher priority for elite experiences
                self.elite_memory.append((state, action, total_reward, next_state, done, elite_priority))
                if len(self.elite_memory) > 8000:  # Keep only recent elite experiences
                    self.elite_memory.popleft()
            
            # Track action distribution and history
            action_idx = np.argmax(action) if isinstance(action, np.ndarray) and action.size > 1 else action
            if isinstance(action_idx, np.ndarray):
                action_idx = action_idx.item()
            self.action_distribution[action_idx] += 1
            self.action_history.append(action_idx)
            
            # Log extreme rewards for debugging
            if abs(total_reward) > 20:
                print(f"High total reward detected: {total_reward:.1f}")
                    
        except Exception as e:
            print(f"Error in remember: {e}")
            import traceback
            traceback.print_exc()
    def train_long_memory(self):
        """Enhanced training with elite experience replay and anti-catastrophic forgetting"""
        if len(self.memory) < BATCH_SIZE:
            return None
        
        try:
            # Regular experience replay
            loss = self._train_batch(self.memory, self.priorities, "regular")
              # Elite experience replay to prevent catastrophic forgetting
            if len(self.elite_memory) >= BATCH_SIZE // 2:
                elite_experiences = list(self.elite_memory)
                elite_priorities = [exp[5] for exp in elite_experiences]  # Extract priorities
                elite_batch = [(exp[0], exp[1], exp[2], exp[3], exp[4]) for exp in elite_experiences]  # Remove priority
                
                elite_loss = self._train_batch(elite_batch, elite_priorities, "elite")
                if elite_loss is not None:
                    loss = (loss + elite_loss * 0.5) / 1.5 if loss is not None else elite_loss
                    
            return loss
        except Exception as e:
            print(f"Error in train_long_memory: {e}")
            return None
    
    def _train_batch(self, memory_buffer, priority_buffer, batch_type="regular"):
        """Train on a batch of experiences with prioritized sampling"""
        if len(memory_buffer) < BATCH_SIZE:
            return None
            
        try:
            # Increase beta over time to reduce sampling bias
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Sample based on batch type
            if batch_type == "elite":
                # For elite experiences, use uniform sampling to preserve all high-level strategies
                batch_size = min(BATCH_SIZE // 2, len(memory_buffer))
                indices = np.random.choice(len(memory_buffer), size=batch_size, replace=False)
                batch = [memory_buffer[idx] for idx in indices]
                weights = torch.ones(len(batch))  # Equal weights for elite experiences
            else:
                # Regular prioritized sampling
                if PRIORITIZED_REPLAY and priority_buffer:
                    probs = np.array(priority_buffer)
                    # Reduce temperature for more stability
                    temperature = max(0.8, 1.2 - (self.n_games / 300))
                    probs = np.power(probs, 1.0 / temperature)
                    probs = probs / np.sum(probs)
                    
                    indices = np.random.choice(
                        len(memory_buffer), 
                        size=min(BATCH_SIZE, len(memory_buffer)),
                        replace=False, 
                        p=probs
                    )
                    
                    # Calculate importance sampling weights
                    weights = (len(memory_buffer) * probs[indices]) ** (-self.beta)
                    weights = weights / np.max(weights)
                    weights = torch.tensor(weights, dtype=torch.float32)
                    
                    batch = [memory_buffer[idx] for idx in indices]
                else:
                    # Standard uniform sampling
                    indices = np.random.choice(
                        len(memory_buffer), 
                        size=min(BATCH_SIZE, len(memory_buffer)),
                        replace=False
                    )
                    batch = [memory_buffer[idx] for idx in indices]
                    weights = torch.ones(len(batch))
                    
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors with careful error handling
            try:
                states = torch.tensor(np.array(states), dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                actions = torch.tensor(np.array(actions), dtype=torch.float32) \
                    if isinstance(actions[0], np.ndarray) and actions[0].size > 1 \
                    else torch.tensor(np.array(actions), dtype=torch.long)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                
                # Check for NaN values
                if torch.isnan(states).any() or torch.isnan(next_states).any():
                    print("Warning: NaN values in batch. Cleaning up...")
                    states = torch.nan_to_num(states)
                    next_states = torch.nan_to_num(next_states)
            except Exception as e:
                print(f"Error converting batch to tensors: {e}")
                return None
            
            # Train the model with importance sampling weights
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
            
            # Update EMA loss and adaptive learning rate
            if loss is not None:
                if self.ema_loss is None:
                    self.ema_loss = loss
                else:
                    self.ema_loss = 0.98 * self.ema_loss + 0.02 * loss  # More stable EMA
                
                # Conservative adaptive learning rate
                if hasattr(self.trainer, 'adaptive_lr_step'):
                    self.trainer.adaptive_lr_step(loss)
                    
                # Log progress for elite batches
                if batch_type == "elite" and self.debug_counter % 100 == 0:
                    print(f"Elite batch training - Loss: {loss:.6f}, Batch size: {len(batch)}")
                    
            return loss
        except Exception as e:
            print(f"Error in _train_batch ({batch_type}): {e}")
            return None
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single experience with improved error handling"""
        try:
            # Only train on significant experiences to reduce noise
            if abs(reward) < 0.01 and random.random() < 0.9:
                return 0.0  # Skip training on very small rewards most of the time
                
            # Track total reward for this episode
            self.total_reward += reward
                
            # Clip extreme rewards
            if abs(reward) > 1000:
                print(f"Clipping extreme reward: {reward}")
                reward = np.sign(reward) * 1000
                
            # Handle large action arrays - convert to index if needed
            if isinstance(action, np.ndarray) and action.size > 1:
                if np.sum(action) == 1:  # If it's one-hot encoded
                    action = np.argmax(action)
                else:
                    # If it's not one-hot encoded but still large, use argmax
                    action = np.argmax(action)
                    
            # Do the actual training
            loss = self.trainer.train_step(state, action, reward, next_state, done)
            
            # Add to running loss for tracking
            if loss is not None:
                self.running_loss += loss
            
            # Log occasional stats during training
            if hasattr(self, 'debug_counter') and self.debug_counter % 500 == 0:
                action_idx = torch.argmax(action).item() if isinstance(action, torch.Tensor) and action.dim() > 0 else action
                if not isinstance(action_idx, (int, np.integer)):
                    action_idx = action_idx.item() if hasattr(action_idx, 'item') else 0
                print(f"Training step - Reward: {reward:.2f}, Action: {action_idx}, Loss: {loss:.6f}")
                
            return loss        
        except Exception as e:
            # Provide more detailed error information
            print(f"Error in train_short_memory: {e}")
            print(f"State type: {type(state)}, shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
            print(f"Action type: {type(action)}, shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
            print(f"Reward type: {type(reward)}, value: {reward}")
            print(f"Next state type: {type(next_state)}, shape: {next_state.shape if hasattr(next_state, 'shape') else 'N/A'}")
            print(f"Done type: {type(done)}, value: {done}")
            return 0.0
    def get_action(self, state, valid_positions):
        """Advanced epsilon-greedy policy with cyclic exploration and curriculum learning"""
        # Cyclic epsilon decay with plateau detection
        if self.n_games % self.epsilon_cycle_length == 0 and self.n_games > 0:
            # Check for plateau - if performance hasn't improved, reset exploration
            if len(self.recent_scores) >= 10:
                recent_avg = sum(self.recent_scores[-10:]) / 10
                if recent_avg <= sum(self.recent_scores[-20:-10]) / 10:  # No improvement
                    self.epsilon = min(60, self.epsilon * 1.5)  # Boost exploration
                    print(f"🔄 Plateau detected! Boosting exploration to {self.epsilon:.1f}")
        
        # Adaptive epsilon decay based on performance
        if hasattr(self, 'recent_performance') and len(self.recent_performance) > 10:
            avg_performance = sum(self.recent_performance[-10:]) / 10
            if avg_performance < 5:  # Poor performance - increase exploration
                self.epsilon = min(80, self.epsilon * 1.002)
            elif avg_performance > 15:  # Good performance - decrease exploration faster
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        self.steps_done += 1
        final_move = np.zeros(self.action_size)
        
        try:
            # Noisy networks approach - add noise to encourage exploration
            use_noise = self.n_games < 100 or random.random() < 0.1
              # Improved exploration strategy with temperature scaling
            if random.randint(0, 100) < self.epsilon:
                # Temperature-scaled exploration based on action diversity
                temperature = getattr(self, 'action_temperature', 1.0)
                
                # Adaptive exploration based on game phase, curriculum, and performance
                difficulty_factor = min(1.0, self.difficulty_level / 5.0)
                
                # Dynamic action type selection based on performance
                if self.n_games < 30:
                    # Early game: focus on placement with some randomness
                    action_type = "place" if random.random() < (0.95 - difficulty_factor * 0.1) else "upgrade"
                elif self.n_games < 100:
                    # Mid game: balanced approach with performance consideration
                    recent_performance = sum(self.recent_performance[-5:]) / min(len(self.recent_performance), 5) if hasattr(self, 'recent_performance') and self.recent_performance else 5
                    if recent_performance > 12:  # Good performance - more strategic
                        action_type = "place" if random.random() < (0.70 - difficulty_factor * 0.1) else "upgrade"
                    else:  # Poor performance - more placement focus
                        action_type = "place" if random.random() < (0.85 - difficulty_factor * 0.1) else "upgrade"
                else:
                    # Late game: performance-adaptive strategy
                    recent_performance = sum(self.recent_performance[-5:]) / min(len(self.recent_performance), 5) if hasattr(self, 'recent_performance') and self.recent_performance else 5
                    if recent_performance > 15:  # High performance - strategic upgrades
                        action_type = "place" if random.random() < (0.50 - difficulty_factor * 0.15) else "upgrade"
                    elif recent_performance > 10:  # Medium performance - balanced
                        action_type = "place" if random.random() < (0.70 - difficulty_factor * 0.1) else "upgrade"
                    else:  # Low performance - back to basics
                        action_type = "place" if random.random() < (0.80 - difficulty_factor * 0.05) else "upgrade"
                
                if action_type == "place" and valid_positions:
                    # Smart placement selection - prefer positions near existing turrets or key locations
                    if len(self.turret_group) > 0 and random.random() < 0.3:
                        # Sometimes place near existing turrets for synergy
                        move = random.choice(valid_positions)
                    else:
                        move = random.choice(valid_positions)
                else:
                    # Smart upgrade selection with action frequency balancing
                    if hasattr(self, 'turret_group') and len(self.turret_group) > 0:
                        upgrade_start = c.ROWS * c.COLS
                        upgrade_range = min(len(self.turret_group), 20)
                        
                        # UCB-like selection for upgrades to balance exploration
                        action_counts = self.action_distribution[upgrade_start:upgrade_start + upgrade_range]
                        if action_counts.sum() > 0:
                            # Select less-tried upgrades more often
                            weights = 1.0 / (action_counts + 1)
                            weights = weights / weights.sum()
                            move = upgrade_start + np.random.choice(upgrade_range, p=weights)
                        else:
                            move = random.randint(upgrade_start, upgrade_start + upgrade_range - 1)
                    else:
                        move = valid_positions[0] if valid_positions else 0
            else:
                # Exploitation with improved Q-value processing
                self.model.eval()
                with torch.no_grad():
                    if isinstance(state, torch.Tensor):
                        state_tensor = state
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32)
                    
                    if len(state_tensor.shape) == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                      # Enhanced Q-value processing with temperature scaling
                    q_values = self.model(state_tensor)[0]
                    
                    # Apply temperature scaling for more diverse action selection
                    temperature = getattr(self, 'action_temperature', 1.0)
                    if temperature != 1.0:
                        q_values = q_values / temperature
                    
                    if use_noise:
                        # Adaptive noise based on performance and exploration needs
                        noise_scale = 0.01 * (1.0 + getattr(self, 'plateau_episodes', 0) * 0.5)
                        noise = torch.randn_like(q_values) * noise_scale
                        q_values = q_values + noise
                    
                    # Enhanced action masking
                    mask = torch.full_like(q_values, float('-inf'))
                    
                    if valid_positions:
                        for pos in valid_positions:
                            if 0 <= pos < len(mask):
                                mask[pos] = 0
                    
                    if hasattr(self, 'turret_group') and len(self.turret_group) > 0:
                        upgrade_start = c.ROWS * c.COLS
                        for i in range(min(len(self.turret_group), 20)):
                            upgrade_idx = upgrade_start + i
                            if upgrade_idx < len(mask):
                                mask[upgrade_idx] = 0
                      # Select best action with optional softmax for diversity
                    masked_q_values = q_values + mask
                    
                    if torch.max(masked_q_values) == float('-inf'):
                        move = valid_positions[0] if valid_positions else 0
                    else:
                        if use_noise and random.random() < (0.15 + getattr(self, 'plateau_episodes', 0) * 0.05):
                            # Dynamic softmax selection probability based on plateau state
                            temperature = getattr(self, 'action_temperature', 1.0)
                            probs = F.softmax(masked_q_values / (0.1 * temperature), dim=0)
                            valid_probs = probs[masked_q_values != float('-inf')]
                            if len(valid_probs) > 0:
                                valid_indices = torch.where(masked_q_values != float('-inf'))[0]
                                selected_idx = torch.multinomial(valid_probs, 1)[0]
                                move = valid_indices[selected_idx].item()
                            else:
                                move = torch.argmax(masked_q_values).item()
                        else:
                            move = torch.argmax(masked_q_values).item()
            
            move = max(0, min(move, self.action_size - 1))
            final_move[move] = 1
            
            # Track action distribution for UCB-like selection
            self.action_distribution[move] += 1
            
        except Exception as e:
            print(f"Error in get_action: {e}")
            if valid_positions:
                move = valid_positions[0]
                final_move[move] = 1
            else:
                final_move[0] = 1
        
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
            self.trainer.target_model.load_state_dict(self.model.state_dict())  # Also update target network
            print(f"Model loaded from {file_name}")
    def log_episode_stats(self, game_reward, score):
        """Enhanced episode statistics logging with plateau detection"""
        self.n_games += 1
        self.reward_history.append(game_reward)
        self.recent_scores.append(score)
        
        # Keep only recent scores for plateau detection
        if len(self.recent_scores) > 50:
            self.recent_scores.pop(0)
        
        # Track recent performance for adaptive exploration
        if not hasattr(self, 'recent_performance'):
            self.recent_performance = []
        self.recent_performance.append(score)
        if len(self.recent_performance) > 20:
            self.recent_performance.pop(0)        # Plateau detection with improved sensitivity
        if len(self.recent_scores) >= 20:  # Faster detection with smaller window
            recent_avg = sum(self.recent_scores[-10:]) / 10  # Very recent window
            older_avg = sum(self.recent_scores[-20:-10]) / 10  # Older comparison window
            
            # Much more sensitive plateau detection for faster intervention
            improvement_threshold = 0.1 if recent_avg < 5 else 0.2 if recent_avg < 10 else 0.3
            
            if recent_avg <= older_avg + improvement_threshold:
                self.plateau_counter += 1
                if self.plateau_counter >= 5:  # Much faster response (was 8)
                    self.handle_plateau()
                    self.plateau_counter = 0
            else:
                self.plateau_counter = 0
                self.last_improvement = self.n_games
                # Reset plateau episodes when improvement is detected
                if hasattr(self, 'plateau_episodes'):
                    self.plateau_episodes = 0
        
        # Curriculum learning progression
        if self.n_games % self.games_per_difficulty == 0:
            recent_avg_score = sum(self.recent_scores[-self.games_per_difficulty:]) / self.games_per_difficulty
            if recent_avg_score >= self.performance_threshold:
                self.difficulty_level += 1
                self.performance_threshold += 2  # Increase threshold for next level
                print(f"🎯 Curriculum advanced to level {self.difficulty_level}! New threshold: {self.performance_threshold}")
        
        # Print enhanced stats every few games
        if self.n_games % 5 == 0:
            avg_reward = sum(self.reward_history[-20:]) / min(len(self.reward_history), 20)
            avg_loss = self.ema_loss if self.ema_loss is not None else 0
            avg_performance = sum(self.recent_performance[-10:]) / min(len(self.recent_performance), 10)
            
            print(f"Game {self.n_games}, Score: {score}, "
                  f"Reward: {game_reward:.1f} (Avg: {avg_reward:.1f}), "
                  f"Loss: {avg_loss:.6f}, Epsilon: {self.epsilon:.1f}, "
                  f"Avg Performance: {avg_performance:.1f}, "
                  f"Difficulty: {self.difficulty_level}, "
                  f"Plateau: {self.plateau_counter}/10")
            
        # Save model periodically
        if self.n_games % 20 == 0:
            self.save_model(f"td_model_checkpoint_{self.n_games}.pth")
              # Save best model based on score thresholds        if score > 15:
            self.save_model(f'td_model_best_{score}.pth')
            print(f"🏆 Best model saved with score: {score}")
    
    def handle_plateau(self):
        """PLATEAU FIX: Much gentler plateau handling to preserve learned patterns"""
        print(f"🔄 Plateau detected at game {self.n_games}! Applying gentle fixes...")
        
        # GENTLE plateau breaking - preserve learned knowledge
        if not hasattr(self, 'plateau_episodes'):
            self.plateau_episodes = 0
        self.plateau_episodes += 1
        
        # Much more conservative approach
        if self.plateau_episodes <= 3:
            # Gentle exploration boost
            self.epsilon = min(self.epsilon * 1.2, 8.0)  # REDUCED: Much gentler
            lr_multiplier = 1.1  # REDUCED: Minimal LR change
            print(f"   📈 Gentle Response: Slight exploration boost")
        elif self.plateau_episodes <= 6:
            # Moderate response
            self.epsilon = min(self.epsilon * 1.3, 12.0)  # REDUCED: Still conservative
            lr_multiplier = 1.15
            
            # Only clear small portion of memory
            if len(self.memory) > 30000:
                remove_count = len(self.memory) // 10  # REDUCED: Only 10% instead of 50%
                for _ in range(remove_count):
                    self.memory.popleft()
                    if self.priorities:
                        self.priorities.popleft()
                print(f"   🧹 Cleared {remove_count} old experiences (10%)")
            print(f"   📈 Moderate Response: Small memory refresh")
        else:
            # Final response - still conservative
            self.epsilon = min(self.epsilon * 1.5, 15.0)  # REDUCED: Max 15 instead of 99
            lr_multiplier = 1.2
            
            # Reset to slightly easier difficulty instead of level 1
            self.difficulty_level = max(1, self.difficulty_level - 1)  # REDUCED: -1 instead of -2
            self.performance_threshold = max(3, self.performance_threshold - 2)  # REDUCED
            
            # Minimal memory clearing
            if len(self.memory) > 25000:
                remove_count = len(self.memory) // 5  # REDUCED: Only 20% instead of 80%
                for _ in range(remove_count):
                    self.memory.popleft()
                    if self.priorities:
                        self.priorities.popleft()
                print(f"   🧹 CONSERVATIVE: Cleared {remove_count} experiences (20%)")
            
            print(f"   💫 Conservative Response: Minimal reset")

        # Gentle learning rate adjustment  
        for param_group in self.trainer.optimizer.param_groups:
            new_lr = min(0.0005, param_group['lr'] * lr_multiplier)  # REDUCED: Lower max LR
            param_group['lr'] = new_lr

        # Conservative parameter adjustment
        self.alpha = min(0.8, self.alpha * 1.1)  # REDUCED: Gentler changes
        
        # Gentle temperature scaling
        if not hasattr(self, 'action_temperature'):
            self.action_temperature = 1.0
        self.action_temperature = min(2.0, 1.0 + (0.2 * self.plateau_episodes))  # REDUCED

        print(f"   📈 Plateau episode #{self.plateau_episodes}")
        print(f"   🎯 New epsilon: {self.epsilon:.1f}")
        print(f"   📚 New LR: {self.trainer.optimizer.param_groups[0]['lr']:.6f}")
        print(f"   ⚖️  New alpha: {self.alpha:.3f}")
        print(f"   🌡️  Action temperature: {getattr(self, 'action_temperature', 1.0):.2f}")

