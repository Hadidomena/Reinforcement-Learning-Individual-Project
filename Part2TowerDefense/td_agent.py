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

MAX_MEMORY = 150_000
BATCH_SIZE = 64
LR = 0.000025
PRIORITIZED_REPLAY = True
NOISY_NETWORKS = False

CONVERGENCE_WINDOW = 50
CONVERGENCE_THRESHOLD = 0.5
BREAKTHROUGH_EPSILON = 15.0
STABILITY_LOSS_THRESHOLD = 0.01

class ConvergenceDetector:
    def __init__(self, window_size=50, convergence_threshold=0.5):
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.score_history = deque(maxlen=window_size)
        self.q_value_history = deque(maxlen=window_size)
        self.action_entropy_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        
    def update(self, score, avg_q_value, action_entropy, loss):
        self.score_history.append(score)
        self.q_value_history.append(avg_q_value)
        self.action_entropy_history.append(action_entropy)
        self.loss_history.append(loss)
        
    def detect_convergence(self):
        if len(self.score_history) < self.window_size:
            return False
            
        score_std = np.std(self.score_history)
        q_value_std = np.std(self.q_value_history) if self.q_value_history else float('inf')
        action_entropy_avg = np.mean(self.action_entropy_history) if self.action_entropy_history else 1.0
        
        convergence_detected = (
            score_std < self.convergence_threshold and
            q_value_std < 2.0 and
            action_entropy_avg < 0.5        )
        
        return convergence_detected
        
    def get_stagnation_level(self):
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
        
        self.epsilon = 2.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99995
        self.epsilon_cycle_length = 500
        self.base_epsilon = 0.2
        self.gamma = 0.995
        
        self.convergence_detector = ConvergenceDetector()
        self.breakthrough_mode = False
        self.breakthrough_countdown = 0
        self.stability_violations = 0
        self.last_breakthrough_game = 0
        
        self.performance_buffer = deque(maxlen=CONVERGENCE_WINDOW)
        self.loss_stability_buffer = deque(maxlen=30)
        self.q_value_tracking = deque(maxlen=100)
        
        self.best_recent_score = 0
        self.performance_window = 100
        self.stable_performance_threshold = 0.85
        
        self.difficulty_level = 1
        self.games_per_difficulty = 50
        self.performance_threshold = 10
        
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
        
        self.target_update_frequency = 200
        self.multi_step_n = 1
        self.double_dqn = True
        self.steps_done = 0
        
        self.performance_memory = deque(maxlen=200)
        self.stable_score_threshold = 10
        self.regression_detection_window = 30
        self.knowledge_preservation_rate = 0.95
        
        self.total_reward = 0
        self.reward_history = []
        self.action_distribution = np.zeros(self.action_size)
        self.running_loss = 0
        self.ema_loss = None
        self.turret_group = []
        
        self.recent_scores = []
        self.plateau_counter = 0
        self.last_improvement = 0
        
        self.lr_decay_patience = 50
        self.lr_decay_factor = 0.7
        self.min_lr = 1e-6
        
        self.reward_shaper = AdvancedRewardShaper()
        self.action_history = deque(maxlen=100)
        
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
        self.debug_counter = 0

    def get_state(self, world, enemy_group, turret_group):
        self.turret_group = turret_group
        state = self.trainer.get_state(world, enemy_group, turret_group)
        
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

            if isinstance(action, np.ndarray) and action.size > 1:
                action_idx = np.argmax(action)
            elif isinstance(action, np.ndarray):
                action_idx = action.item()
            else:
                action_idx = action
                
            # Ensure action_idx is a proper integer
            if isinstance(action_idx, (np.ndarray, torch.Tensor)):
                action_idx = action_idx.item()
            action_idx = int(action_idx)
            
            if game_stats:
                action_type = "upgrade" if action_idx >= c.ROWS * c.COLS else "place"
                shaped_reward = self.reward_shaper.calculate_reward(state, action_type, game_stats)
                curiosity_reward = self.reward_shaper.get_curiosity_reward(state, action, next_state)
                curriculum_bonus = self.reward_shaper.get_curriculum_bonus(
                    self.difficulty_level, game_stats.get('score', 0)
                )
                total_reward = reward + shaped_reward + curiosity_reward + curriculum_bonus
            else:
                total_reward = reward
                
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ns = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
                current_q = self.model(s)
                
                if done:
                    target_q = total_reward
                else:
                    next_q = self.trainer.target_model(ns)
                    target_q = total_reward + self.gamma * torch.max(next_q).item()
                
                if isinstance(action_idx, np.ndarray):
                    action_idx = action_idx.item()
                td_error = abs(target_q - current_q[0, action_idx].item())

            priority = (td_error + 0.1) ** self.alpha
                
            self.memory.append((state, action_idx, total_reward, next_state, done))
            self.priorities.append(priority)
            
            if game_stats and game_stats.get('level', 0) >= self.elite_threshold:
                elite_priority = priority * 2.0
                self.elite_memory.append((state, action_idx, total_reward, next_state, done, elite_priority))
                if len(self.elite_memory) > 8000:
                    self.elite_memory.popleft()
            
            if isinstance(action_idx, np.ndarray):
                action_idx = action_idx.item()
            self.action_distribution[action_idx] += 1
            self.action_history.append(action_idx)
                    
        except Exception as e:
            print(f"Error in remember: {e}")

    def get_action(self, state, valid_positions):
        if self.breakthrough_mode:
            self.update_breakthrough_mode()
        
        if hasattr(self, 'recent_performance') and len(self.recent_performance) > 10:
            avg_performance = sum(self.recent_performance[-10:]) / 10
            performance_variance = np.var(self.recent_performance[-10:])
            
            if performance_variance < 0.3 and not self.breakthrough_mode:
                print(f"⚠️  Low performance variance detected: {performance_variance:.3f}")
                self.epsilon = min(self.epsilon * 1.05, 8.0)
            elif avg_performance > 8 and performance_variance < 4 and not self.breakthrough_mode:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.9995)
            elif avg_performance < 3:
                self.epsilon = min(self.epsilon * 1.002, 6.0)
            else:
                decay_rate = self.epsilon_decay * 1.1 if self.breakthrough_mode else self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        self.steps_done += 1
        final_move = np.zeros(self.action_size)
        
        try:
            if random.randint(0, 100) < self.epsilon:
                if valid_positions:
                    move = random.choice(valid_positions)
                else:
                    move = 0
            else:
                self.model.eval()
                with torch.no_grad():
                    if isinstance(state, torch.Tensor):
                        state_tensor = state
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32)
                    
                    if len(state_tensor.shape) == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                    
                    q_values = self.model(state_tensor)[0]
                    self.track_q_value_stability(q_values)
                    
                    if self.breakthrough_mode and random.random() < 0.1:
                        noise_scale = 0.02
                        noise = torch.randn_like(q_values) * noise_scale
                        q_values = q_values + noise
                    
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
                    
                    masked_q_values = q_values + mask
                    
                    if torch.max(masked_q_values) == float('-inf'):
                        move = valid_positions[0] if valid_positions else 0
                    else:
                        move = torch.argmax(masked_q_values).item()
            
            move = max(0, min(move, self.action_size - 1))
            final_move[move] = 1
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
        try:
            if abs(reward) < 0.01 and random.random() < 0.9:
                return 0.0
                
            self.total_reward += reward
                
            if abs(reward) > 1000:
                reward = np.sign(reward) * 1000
                
            if isinstance(action, np.ndarray) and action.size > 1:
                action = np.argmax(action)
            
            # Ensure action is a proper integer
            if isinstance(action, (np.ndarray, torch.Tensor)):
                action = action.item()
            action = int(action)
                    
            loss = self.trainer.train_step(state, action, reward, next_state, done)

            if loss is not None:
                self.running_loss += loss
                
            return loss        
        except Exception as e:
            print(f"Error in train_short_memory: {e}")
            return 0.0

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        
        try:
            batch_size = min(BATCH_SIZE, len(self.memory))
            indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
            batch = [self.memory[idx] for idx in indices]
            
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            
            loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
            
            return loss
        except Exception as e:
            print(f"Error in train_long_memory: {e}")
            return None

    def log_episode_stats(self, game_reward, score):
        self.n_games += 1
        self.recent_scores.append(score)
        self.total_reward += game_reward
        
        if hasattr(self, 'running_loss') and self.running_loss > 0:
            avg_loss = self.running_loss / max(1, self.n_games - 1)
        else:
            avg_loss = 0.0
            
        if len(self.recent_scores) > 10:
            avg_performance = sum(self.recent_scores[-10:]) / 10
            if not hasattr(self, 'recent_performance'):
                self.recent_performance = deque(maxlen=50)
            self.recent_performance.append(avg_performance)
        else:
            avg_performance = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
            
        convergence_detected = self.detect_and_handle_convergence(score, avg_loss)
        self.performance_memory.append(score)
        
        if len(self.recent_scores) > 50:
            self.recent_scores.pop(0)

    def detect_and_handle_convergence(self, score, loss):
        if hasattr(self, 'action_distribution'):
            total_actions = np.sum(self.action_distribution) 
            if total_actions > 0:
                action_probs = self.action_distribution / total_actions
                action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            else:
                action_entropy = 1.0
        else:
            action_entropy = 1.0
        
        if hasattr(self, 'q_value_tracking') and self.q_value_tracking:
            avg_q_value = np.mean(self.q_value_tracking)
        else:
            avg_q_value = 0.0
            
        self.convergence_detector.update(score, avg_q_value, action_entropy, loss or 0.0)
        self.performance_buffer.append(score)
        
        if self.convergence_detector.detect_convergence():
            self.initiate_breakthrough_mode()
            return True
                
        return False

    def initiate_breakthrough_mode(self):
        if not self.breakthrough_mode:
            self.breakthrough_mode = True
            self.breakthrough_countdown = 200
            self.last_breakthrough_game = self.n_games
            self.breakthrough_attempts += 1
            
            self.epsilon = min(BREAKTHROUGH_EPSILON, self.epsilon * 3.0)
            print(f"🚀 BREAKTHROUGH MODE activated! Game {self.n_games}")
            print(f"   🎯 Exploration boosted to {self.epsilon:.1f}")

    def update_breakthrough_mode(self):
        if self.breakthrough_mode:
            self.breakthrough_countdown -= 1
            if self.breakthrough_countdown <= 0:
                self.breakthrough_mode = False
                self.epsilon = max(self.base_epsilon, self.epsilon * 0.3)
                print(f"🏁 Breakthrough mode ended. Epsilon: {self.epsilon:.2f}")

    def track_q_value_stability(self, q_values):
        if len(q_values.shape) > 0:
            avg_q = torch.mean(q_values).item()
            self.q_value_tracking.append(avg_q)
            
            if len(self.q_value_tracking) >= 50:
                recent_q_std = np.std(list(self.q_value_tracking)[-50:])
                if recent_q_std < 0.1:
                    print(f"⚠️  Q-value stability detected: std={recent_q_std:.3f}")

    def save_model(self, filename='td_model.pth'):
        """Save the current model to a file"""
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Create full file path
            filepath = os.path.join(models_dir, filename)
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'n_games': self.n_games,
                'epsilon': self.epsilon,
                'total_reward': self.total_reward,
                'action_size': self.action_size,
                'state_size': self.state_size
            }, filepath)
            
            print(f"Model saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename='td_model.pth'):
        """Load a previously saved model"""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            filepath = os.path.join(models_dir, filename)
            
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.n_games = checkpoint.get('n_games', 0)
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.total_reward = checkpoint.get('total_reward', 0)
                
                # Update target network
                self.trainer.target_model.load_state_dict(self.model.state_dict())
                
                print(f"Model loaded from {filepath}")
                print(f"Resuming from game {self.n_games} with epsilon {self.epsilon:.4f}")
                return True
            else:
                print(f"Model file {filepath} not found")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False