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

# ADVANCED STABILITY: Ultra-conservative parameters for long-term stability
MAX_MEMORY = 150_000   # INCREASED: Even larger memory for better sample diversity
BATCH_SIZE = 64        # REDUCED: Smaller batches for more stable gradient estimates  
LR = 0.000025          # ULTRA-LOW: Extremely low learning rate for fine control
PRIORITIZED_REPLAY = True  
NOISY_NETWORKS = False  # Keep disabled for stability

# ANTI-CONVERGENCE CONSTANTS
CONVERGENCE_WINDOW = 50      # Window to detect convergence
CONVERGENCE_THRESHOLD = 0.5  # Max std dev for convergence detection
BREAKTHROUGH_EPSILON = 15.0  # High exploration for breakthrough attempts
STABILITY_LOSS_THRESHOLD = 0.01  # Maximum acceptable loss for stability

class ConvergenceDetector:
    """Zaawansowany detektor konwergencji dla zapobiegania utkniecia w lokalnym minimum"""
    
    def __init__(self, window_size=50, convergence_threshold=0.5):
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.score_history = deque(maxlen=window_size)
        self.q_value_history = deque(maxlen=window_size)
        self.action_entropy_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        
    def update(self, score, avg_q_value, action_entropy, loss):
        """Aktualizuj historię metrykk"""
        self.score_history.append(score)
        self.q_value_history.append(avg_q_value)
        self.action_entropy_history.append(action_entropy)
        self.loss_history.append(loss)
        
    def detect_convergence(self):
        """Wykryj czy model zbiega do średniej wartości"""
        if len(self.score_history) < self.window_size:
            return False
            
        # Sprawdź wariancję w ostatnim oknie
        score_std = np.std(self.score_history)
        q_value_std = np.std(self.q_value_history) if self.q_value_history else float('inf')
        action_entropy_avg = np.mean(self.action_entropy_history) if self.action_entropy_history else 1.0
        
        # Kryteria konwergencji:
        # 1. Niska wariancja wyników
        # 2. Stabilne Q-wartości  
        # 3. Niska entropija akcji (brak eksploracji)
        convergence_detected = (
            score_std < self.convergence_threshold and
            q_value_std < 2.0 and
            action_entropy_avg < 0.5
        )
        
        return convergence_detected
        
    def get_stagnation_level(self):
        """Określ poziom stagnacji (0-1)"""
        if len(self.score_history) < 20:
            return 0.0
            
        recent_std = np.std(list(self.score_history)[-20:])
        older_std = np.std(list(self.score_history)[-40:-20]) if len(self.score_history) >= 40 else recent_std
        
        if older_std == 0:
            return 1.0 if recent_std < 0.1 else 0.0
            
        variance_reduction = 1.0 - (recent_std / (older_std + 1e-6))
        return max(0.0, min(1.0, variance_reduction))

class TowerDefenseAgent:    
    def __init__(self):
        self.n_games = 0        
        
        # ULTRA-CONSERVATIVE exploration parameters for stability
        self.epsilon = 2.0      # FURTHER REDUCED: Start even lower
        self.epsilon_min = 0.02 # ULTRA-LOW: Minimal exploration for exploitation
        self.epsilon_decay = 0.99995  # ULTRA-SLOW: Extremely gradual decay
        self.epsilon_cycle_length = 500  # MASSIVE: Very long cycles
        self.base_epsilon = 0.2  # ULTRA-LOW: Lower base for resets
        self.gamma = 0.995  # INCREASED: Higher discount for long-term planning
        
        # ANTI-CONVERGENCE SYSTEM
        self.convergence_detector = ConvergenceDetector()
        self.breakthrough_mode = False
        self.breakthrough_countdown = 0
        self.stability_violations = 0
        self.last_breakthrough_game = 0
        
        # PERFORMANCE STABILITY TRACKING
        self.performance_buffer = deque(maxlen=CONVERGENCE_WINDOW)
        self.loss_stability_buffer = deque(maxlen=30)
        self.q_value_tracking = deque(maxlen=100)
        
        # Performance-based epsilon management
        self.best_recent_score = 0
        self.performance_window = 100  # INCREASED: Longer window for stability
        self.stable_performance_threshold = 0.85  # INCREASED: Higher threshold
        
        # Curriculum learning parameters - more conservative
        self.difficulty_level = 1
        self.games_per_difficulty = 50  # INCREASED: More games per difficulty
        self.performance_threshold = 10  # INCREASED: Higher threshold
        
        # Experience replay with conservative priorities
        self.memory = deque(maxlen=MAX_MEMORY)
        self.priorities = deque(maxlen=MAX_MEMORY)
        # Elite experience buffer for high-performance episodes
        self.elite_memory = deque(maxlen=20000)  # INCREASED: More elite experiences
        self.elite_threshold = 10  # REDUCED: Lower threshold for more elite experiences
        
        self.alpha = 0.4    # REDUCED: Much lower for stability
        self.beta = 0.3     # REDUCED: Lower start
        self.beta_increment = 0.000005  # SLOWER: Much slower beta increase
        
        # Calculate action space
        self.action_size = c.ROWS * c.COLS + 20
        # Updated state size to match enhanced state representation:
        # 8 basic features + 20*3 enemy features + 10*4 turret features = 8 + 60 + 40 = 108
        self.state_size = 108        
        print(f"Action space size: {self.action_size}")
        print(f"State space size: {self.state_size}")
        
        # Improved network architecture with noisy layers
        self.model = TowerDefenseQNet(self.state_size, 512, self.action_size)  # Larger network
        self.trainer = TowerDefenseTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Multi-step learning and double DQN - IMPROVED stability
        self.target_update_frequency = 200  # INCREASED: Much less frequent updates for stability
        self.multi_step_n = 1  # REDUCED: Simpler single-step learning for stability
        self.double_dqn = True  # Use Double DQN for reduced overestimation
        self.steps_done = 0
        
        # ANTI-CATASTROPHIC FORGETTING: Additional stability measures
        self.performance_memory = deque(maxlen=200)  # Track longer performance history
        self.stable_score_threshold = 10  # Score threshold for "stable" performance
        self.regression_detection_window = 30  # Window for regression detection
        self.knowledge_preservation_rate = 0.95  # How much knowledge to preserve
        
        # Tracking metrics for introspection
        self.total_reward = 0
        self.reward_history = []  # Track rewards over episodes
        self.action_distribution = np.zeros(self.action_size)  # Track action frequencies
        self.running_loss = 0
        self.ema_loss = None  # Exponential moving average loss
        self.turret_group = []  # Will be updated during gameplay
        
        # Performance tracking for curriculum learning
        self.recent_scores = []
        self.plateau_counter = 0
        self.last_improvement = 0
        
        # Learning rate scheduling
        self.lr_decay_patience = 50
        self.lr_decay_factor = 0.7
        self.min_lr = 1e-6
        
        # Advanced reward shaping
        self.reward_shaper = AdvancedRewardShaper()
        
        # Action history for exploration bonus
        self.action_history = deque(maxlen=100)
        
        # Anti-catastrophic forgetting mechanisms
        self.knowledge_consolidation = True
        self.stable_policy_buffer = deque(maxlen=1000)  # Store stable high-level policies
        self.performance_regression_counter = 0
        self.last_high_score = 0
        self.consolidation_frequency = 20  # How often to consolidate knowledge
        
        # Performance tracking for regression detection
        self.score_history = deque(maxlen=100)
        self.level_history = deque(maxlen=100)
        self.regression_threshold = 0.7  # If performance drops below 70% of recent best
        
        # Meta-learning: Track performance patterns
        self.performance_history = deque(maxlen=100)
        self.strategy_history = deque(maxlen=50)
        self.breakthrough_attempts = 0
        
        # Advanced exploration parameters
        self.exploration_momentum = 0.0
        self.strategy_adaptation_rate = 0.1
        
        # Debug counter
        self.debug_counter = 0

    def get_state(self, world, enemy_group, turret_group):
        """Get current state with increased debug info"""
        # Save turret_group for use in fallback actions
        self.turret_group = turret_group
        
        # Use the trainer's get_state method for consistency
        state = self.trainer.get_state(world, enemy_group, turret_group)
        
        # Debug info periodically
        if hasattr(self, 'n_games') and self.n_games % 10 == 0 and hasattr(self, 'debug_counter') and self.debug_counter % 50 == 0:
            print(f"State shape: {state.shape}, Range: [{state.min().item():.2f}, {state.max().item():.2f}]")
            
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        self.debug_counter += 1
            
        return state

    def remember(self, state, action, reward, next_state, done, game_stats=None):
        """Store experience with enhanced reward shaping"""
        try:
            # Convert to numpy arrays for consistent storage
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu().numpy()
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
            # Handle NaN values
            if np.isnan(np.sum(state)) or np.isnan(np.sum(next_state)):
                print("Warning: NaN detected in state. Skipping this experience.")
                return
                
            # Handle extreme values
            if np.max(np.abs(state)) > 100 or np.max(np.abs(next_state)) > 100:
                print(f"Warning: Extreme values in state: {np.max(np.abs(state)):.1f}. Clipping.")
                state = np.clip(state, -100, 100)
                next_state = np.clip(next_state, -100, 100)
              # Convert action to index for consistent storage first
            if isinstance(action, np.ndarray) and action.size > 1:
                action_idx = np.argmax(action)
            elif isinstance(action, np.ndarray):
                action_idx = action.item()
            else:
                action_idx = action
            
            # Enhanced reward shaping
            if game_stats:
                action_type = "upgrade" if action_idx >= c.ROWS * c.COLS else "place"
                shaped_reward = self.reward_shaper.calculate_reward(state, action_type, game_stats)
                
                # Add curiosity reward for exploration
                curiosity_reward = self.reward_shaper.get_curiosity_reward(state, action, next_state)
                
                # Add curriculum bonus
                curriculum_bonus = self.reward_shaper.get_curriculum_bonus(
                    self.difficulty_level, game_stats.get('score', 0)
                )
                
                # Combine rewards
                total_reward = reward + shaped_reward + curiosity_reward + curriculum_bonus
            else:
                total_reward = reward
                
            # Calculate temporal difference (TD) error for better priority
            with torch.no_grad():
                # Convert to tensors for model inference
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
                # Get Q values
                current_q = self.model(s)
                
                # Calculate target Q value
                if done:
                    target_q = total_reward
                else:
                    # Use trainer's target model for more stable TD error calculation
                    next_q = self.trainer.target_model(ns)
                    target_q = total_reward + self.gamma * torch.max(next_q).item()
                
                # Calculate TD error (priority) using action index
                if isinstance(action_idx, np.ndarray):
                    action_idx = action_idx.item()  # Convert to scalar
                td_error = abs(target_q - current_q[0, action_idx].item())
              # Add small constant to prevent zero priority
            priority = (td_error + 0.1) ** self.alpha
                
            # Store experience with enhanced reward (store action as index, not array)
            self.memory.append((state, action_idx, total_reward, next_state, done))
            self.priorities.append(priority)
            
            # Store elite experiences for anti-catastrophic forgetting
            if game_stats and game_stats.get('level', 0) >= self.elite_threshold:
                elite_priority = priority * 2.0  # Higher priority for elite experiences
                self.elite_memory.append((state, action_idx, total_reward, next_state, done, elite_priority))
                if len(self.elite_memory) > 8000:  # Keep only recent elite experiences
                    self.elite_memory.popleft()
            
            # Track action distribution and history
            if isinstance(action_idx, np.ndarray):
                action_idx = action_idx.item()  # Convert to scalar
            self.action_distribution[action_idx] += 1
            self.action_history.append(action_idx)
                    
        except Exception as e:
            print(f"Error in remember: {e}")

    def get_action(self, state, valid_positions):
        """ENHANCED: Improved epsilon-greedy policy with anti-convergence mechanisms"""
        
        # Aktualizuj tryb przełamania jeśli aktywny
        if self.breakthrough_mode:
            self.update_breakthrough_mode()
        
        # Enhanced adaptive epsilon decay with anti-convergence protection
        if hasattr(self, 'recent_performance') and len(self.recent_performance) > 10:
            avg_performance = sum(self.recent_performance[-10:]) / 10
            performance_variance = np.var(self.recent_performance[-10:])
            
            # Wykryj potencjalną konwergencję na podstawie wariancji
            if performance_variance < 0.3 and not self.breakthrough_mode:
                print(f"⚠️  Low performance variance detected: {performance_variance:.3f}")
                self.epsilon = min(self.epsilon * 1.05, 8.0)  # Gentle exploration boost
            elif avg_performance > 8 and performance_variance < 4 and not self.breakthrough_mode:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.9995)  # Slower decay when stable
            elif avg_performance < 3:  # Poor performance - exploration increase
                self.epsilon = min(self.epsilon * 1.002, 6.0)  # Slightly more aggressive
            else:
                # Standard decay - but slower in breakthrough mode
                decay_rate = self.epsilon_decay * 1.1 if self.breakthrough_mode else self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
        else:
            # Standard decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        self.steps_done += 1
        final_move = np.zeros(self.action_size)
        
        try:
            # Enhanced exploration strategy
            if random.randint(0, 100) < self.epsilon:
                if valid_positions:
                    move = random.choice(valid_positions)
                else:
                    move = 0
            else:
                # Exploitation with Q-network
                self.model.eval()
                with torch.no_grad():
                    if isinstance(state, torch.Tensor):
                        state_tensor = state
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32)
                    
                    if len(state_tensor.shape) == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                    
                    # Get Q-values
                    q_values = self.model(state_tensor)[0]
                    
                    # NOWY: Śledź stabilność Q-wartości
                    self.track_q_value_stability(q_values)
                    
                    # Enhanced noise injection based on breakthrough mode
                    if self.breakthrough_mode and random.random() < 0.1:
                        noise_scale = 0.02
                        noise = torch.randn_like(q_values) * noise_scale
                        q_values = q_values + noise
                    
                    # Action masking
                    mask = torch.full_like(q_values, float('-inf'))
                    
                    # Enable valid placements
                    if valid_positions:
                        for pos in valid_positions:
                            if 0 <= pos < len(mask):
                                mask[pos] = 0
                    
                    # Enable upgrades for existing turrets
                    if hasattr(self, 'turret_group') and len(self.turret_group) > 0:
                        upgrade_start = c.ROWS * c.COLS
                        for i in range(min(len(self.turret_group), 20)):
                            upgrade_idx = upgrade_start + i
                            if upgrade_idx < len(mask):
                                mask[upgrade_idx] = 0
                    
                    # Select best action
                    masked_q_values = q_values + mask
                    
                    if torch.max(masked_q_values) == float('-inf'):
                        move = valid_positions[0] if valid_positions else 0
                    else:
                        move = torch.argmax(masked_q_values).item()
            
            # Ensure valid action
            move = max(0, min(move, self.action_size - 1))
            final_move[move] = 1
            
            # Track action distribution for diversity analysis
            self.action_distribution[move] += 1
            
        except Exception as e:
            print(f"Error in get_action: {e}")
            if valid_positions:
                move = valid_positions[0]
                final_move[move] = 1
            else:
                final_move[0] = 1
        
        return final_move

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single experience with improved error handling"""
        try:
            # Only train on significant experiences to reduce noise
            if abs(reward) < 0.01 and random.random() < 0.9:
                return 0.0
                
            # Track total reward for this episode
            self.total_reward += reward
                
            # Clip extreme rewards
            if abs(reward) > 1000:
                reward = np.sign(reward) * 1000
                
            # Handle large action arrays - convert to index if needed
            if isinstance(action, np.ndarray) and action.size > 1:
                action = np.argmax(action)
                    
            # Do the actual training
            loss = self.trainer.train_step(state, action, reward, next_state, done)
              # Add to running loss for tracking
            if loss is not None:
                self.running_loss += loss
                
            return loss        
        except Exception as e:
            print(f"Error in train_short_memory: {e}")
            return 0.0

    def train_long_memory(self):
        """Enhanced training with elite experience replay"""
        if len(self.memory) < BATCH_SIZE:
            return None
        
        try:
            # Sample batch from memory
            batch_size = min(BATCH_SIZE, len(self.memory))
            indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
            batch = [self.memory[idx] for idx in indices]
            
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors (actions are already indices now)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            
            # Train the model
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
            
            return loss
        except Exception as e:
            print(f"Error in train_long_memory: {e}")
            return None

    def log_episode_stats(self, game_reward, score):
        """ENHANCED: Episode statistics logging with convergence detection"""
        self.n_games += 1
        
        # Basic stats tracking
        self.recent_scores.append(score)
        self.total_reward += game_reward
        
        if hasattr(self, 'running_loss') and self.running_loss > 0:
            avg_loss = self.running_loss / max(1, self.n_games - 1)
        else:
            avg_loss = 0.0
            
        # Calculate performance metrics
        if len(self.recent_scores) > 10:
            avg_performance = sum(self.recent_scores[-10:]) / 10
            if not hasattr(self, 'recent_performance'):
                self.recent_performance = deque(maxlen=50)
            self.recent_performance.append(avg_performance)
        else:
            avg_performance = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
            
        # NOWY: Wykryj konwergencję i uruchom przeciwdziałania
        convergence_detected = self.detect_and_handle_convergence(score, avg_loss)
        
        # Add to performance memory for regression detection
        self.performance_memory.append(score)
        
        # Keep only recent scores for plateau detection
        if len(self.recent_scores) > 50:
            self.recent_scores.pop(0)

    def detect_and_handle_convergence(self, score, loss):
        """NOWY: Wykryj i przeciwdziałaj konwergencji do średnich wartości"""
        
        # Oblicz entropię akcji dla trackingu
        if hasattr(self, 'action_distribution'):
            total_actions = np.sum(self.action_distribution) 
            if total_actions > 0:
                probs = self.action_distribution / total_actions
                probs = probs[probs > 0]
                action_entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
            else:
                action_entropy = 0.0
        else:
            action_entropy = 1.0
        
        # Oblicz średnią Q-wartość
        if hasattr(self, 'q_value_tracking') and self.q_value_tracking:
            avg_q_value = np.mean(self.q_value_tracking)
        else:
            avg_q_value = 0.0
            
        # Aktualizuj detektor konwergencji
        self.convergence_detector.update(score, avg_q_value, action_entropy, loss or 0.0)
        self.performance_buffer.append(score)
        
        # Wykryj konwergencję
        if self.convergence_detector.detect_convergence():
            stagnation_level = self.convergence_detector.get_stagnation_level()
            
            if stagnation_level > 0.8 and not self.breakthrough_mode:
                print(f"🚨 CONVERGENCE DETECTED! Stagnation level: {stagnation_level:.2f}")
                print(f"   Score std: {np.std(self.performance_buffer):.2f}")
                print(f"   Action entropy: {action_entropy:.3f}")
                print(f"   Initiating breakthrough protocol...")
                
                self.initiate_breakthrough_mode()
                return True
                
        return False

    def initiate_breakthrough_mode(self):
        """NOWY: Uruchom tryb przełamania konwergencji"""
        print("🚀 BREAKTHROUGH MODE ACTIVATED")
        
        self.breakthrough_mode = True
        self.breakthrough_countdown = 30  # 30 gier w trybie przełamania
        self.last_breakthrough_game = self.n_games
        
        # DRASTYCZNE zwiększenie eksploracji
        self.epsilon = min(BREAKTHROUGH_EPSILON, self.epsilon * 3.0)
        print(f"   🎯 Exploration boosted to {self.epsilon:.1f}")
        
        # Zwiększ learning rate tymczasowo
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = min(0.001, param_group['lr'] * 2.0)
        print(f"   📈 Learning rate boosted to {self.trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Zresetuj część pamięci (zachowaj elite experiences)
        if len(self.memory) > 20000:
            remove_count = len(self.memory) // 3
            for _ in range(remove_count):
                if self.memory:
                    self.memory.popleft()
                if self.priorities:
                    self.priorities.popleft()
            print(f"   🧹 Cleared {remove_count} regular experiences")
            
        # Zresetuj reward shaping
        if hasattr(self.reward_shaper, 'reset_adaptive_parameters'):
            self.reward_shaper.reset_adaptive_parameters()
            
        # Zresetuj tracking action distribution
        self.action_distribution = np.zeros(self.action_size)

    def update_breakthrough_mode(self):
        """NOWY: Aktualizuj stan trybu przełamania"""
        if self.breakthrough_mode:
            self.breakthrough_countdown -= 1
            
            if self.breakthrough_countdown <= 0:
                print("🎯 BREAKTHROUGH MODE COMPLETED")
                print("   Returning to stable learning parameters...")
                
                self.breakthrough_mode = False
                
                # Przywróć konserwatywne parametry
                self.epsilon = max(1.0, self.epsilon * 0.3)
                print(f"   🎯 Exploration normalized to {self.epsilon:.1f}")
                
                # Przywróć normalny learning rate
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = LR
                print(f"   📉 Learning rate normalized to {self.trainer.optimizer.param_groups[0]['lr']:.6f}")
                
                # Zresetuj detektor konwergencji
                self.convergence_detector = ConvergenceDetector()

    def track_q_value_stability(self, q_values):
        """NOWY: Śledź stabilność Q-wartości"""
        if torch.is_tensor(q_values):
            avg_q = torch.mean(q_values).item()
            max_q = torch.max(q_values).item()
            min_q = torch.min(q_values).item()
            
            self.q_value_tracking.append(avg_q)
            
            # Wykryj niestabilne Q-wartości
            if len(self.q_value_tracking) >= 50:
                q_std = np.std(list(self.q_value_tracking)[-50:])
                q_range = max_q - min_q
                
                # Jeśli Q-wartości są zbyt stabilne, może to oznaczać konwergencję
                if q_std < 0.5 and q_range < 2.0:
                    self.stability_violations += 1
                    if self.stability_violations > 10:
                        print(f"⚠️  Q-value stability violation detected (std: {q_std:.3f}, range: {q_range:.3f})")
                        return True
                else:
                    self.stability_violations = max(0, self.stability_violations - 1)
                    
        return False

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
            self.trainer.target_model.load_state_dict(self.model.state_dict())
            print(f"Model loaded from {file_name}")