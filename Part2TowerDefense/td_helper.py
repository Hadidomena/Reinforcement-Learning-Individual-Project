import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import time
import os
import torch
import constants as c

plt.ion()

# Store figure and axes globally
fig = None
ax1 = None
ax2 = None

def plot(scores, mean_scores, rewards):
    global fig, ax1, ax2
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        ax1.clear()
        ax2.clear()
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        ax1.clear()
        ax2.clear()

        scores = list(scores) if scores else []
        mean_scores = list(mean_scores) if mean_scores else []
        rewards = list(rewards) if rewards else []

        ax1.set_title('Training Scores')
        ax1.set_xlabel('Number of Games')
        ax1.set_ylabel('Score')
        
        if scores:
            ax1.plot(scores, 'b-', label='Score')
        if mean_scores:
            ax1.plot(mean_scores, 'r-', label='Mean Score')
        
        if len(scores) >= 5:
            window_size = min(5, len(scores))
            try:
                moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                start_idx = window_size - 1
                end_idx = len(scores)
                indices = list(range(start_idx, end_idx))
                if len(moving_avg) == len(indices):
                    ax1.plot(indices, moving_avg, 'g--', label=f'{window_size}-game Moving Avg')
            except Exception as e:
                print(f"Warning: Could not plot moving average for scores: {e}")
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax2.set_title('Rewards per Game')
        ax2.set_xlabel('Number of Games')
        ax2.set_ylabel('Reward')
        
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            if max_reward - min_reward < 1:
                max_reward = min_reward + 1 if min_reward < 0 else 1
            
            y_range = max_reward - min_reward
            buffer = max(y_range * 0.1, 1)
            
            ax2.set_ylim([min_reward - buffer, max_reward + buffer])
            ax2.plot(rewards, 'g-', label='Reward')
            
            if len(rewards) >= 10:
                window_size = min(10, len(rewards))
                try:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    start_idx = window_size - 1
                    end_idx = len(rewards)
                    indices = list(range(start_idx, end_idx))
                    if len(moving_avg) == len(indices):
                        ax2.plot(indices, moving_avg, 'r--', label=f'{window_size}-game Moving Avg')
                except Exception as e:
                    print(f"Warning: Could not plot moving average for rewards: {e}")

    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    display.clear_output(wait=True)
    display.display(plt.gcf())

class TrainingAnalyzer:
    def __init__(self, agent, log_dir='training_logs'):
        self.agent = agent
        self.log_dir = log_dir
        self.start_time = time.time()
        self.game_times = []
        self.rewards_history = []
        self.level_history = []
        self.action_frequencies = []
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_game_stats(self, game_num, score, level, reward, epsilon):
        game_time = time.time() - self.start_time
        self.game_times.append(game_time)
        self.rewards_history.append(reward)
        self.level_history.append(level)
        
        if hasattr(self.agent, 'action_distribution'):
            action_dist = self.agent.action_distribution.copy()
            total_actions = action_dist.sum() if action_dist.sum() > 0 else 1
            normalized_dist = action_dist / total_actions
            self.action_frequencies.append(normalized_dist)
        
        log_file = os.path.join(self.log_dir, f'game_stats.csv')
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, 'a') as f:
            if not file_exists:
                f.write('game,time,score,level,reward,epsilon\n')
            f.write(f'{game_num},{game_time:.2f},{score},{level},{reward:.2f},{epsilon:.2f}\n')
        
        if game_num % 10 == 0:
            self.analyze_training_trends()
    
    def analyze_training_trends(self):
        """Analyze trends in training performance"""
        if len(self.rewards_history) < 2:
            return
            
        # Calculate reward improvement trend
        recent_rewards = self.rewards_history[-10:] if len(self.rewards_history) >= 10 else self.rewards_history
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Calculate level progression rate
        recent_levels = self.level_history[-10:] if len(self.level_history) >= 10 else self.level_history
        avg_recent_level = sum(recent_levels) / len(recent_levels)
        
        # Calculate training speed
        games_per_minute = len(self.game_times) / (self.game_times[-1] / 60) if self.game_times else 0
        
        # Print analysis
        print("\n----- TRAINING ANALYSIS -----")
        print(f"Average Recent Reward: {avg_recent_reward:.2f}")
        print(f"Average Recent Level: {avg_recent_level:.2f}")
        print(f"Training Speed: {games_per_minute:.2f} games/minute")
        
        # Analyze action distribution if available
        if self.action_frequencies and hasattr(self.agent, 'action_distribution'):
            # Calculate recent action distribution
            recent_dist = self.action_frequencies[-1]
            placement_ratio = recent_dist[:c.ROWS * c.COLS].sum()
            upgrade_ratio = recent_dist[c.ROWS * c.COLS:].sum()
            
            print(f"Action Ratio - Placements: {placement_ratio:.1%}, Upgrades: {upgrade_ratio:.1%}")
            
            # Check for potential issues
            if placement_ratio < 0.2:
                print("Warning: Agent is not placing enough turrets. Consider adjusting rewards.")
            if upgrade_ratio < 0.1:
                print("Warning: Agent is rarely upgrading turrets. Consider adjusting upgrade rewards.")
        
        print("--------------------------\n")
    
    def save_model_analysis(self, model, filename='model_analysis.txt'):
        """Analyze and save insights about the Q-network model"""
        if not isinstance(model, torch.nn.Module):
            print("Error: Not a PyTorch model")
            return
            
        try:
            # Get model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Analyze parameter distributions
            param_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_stats[name] = {
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'min': param.data.min().item(),
                        'max': param.data.max().item(),
                    }
            
            # Save analysis to file
            with open(os.path.join(self.log_dir, filename), 'w') as f:
                f.write(f"Model Analysis\n")
                f.write(f"=============\n\n")
                f.write(f"Total parameters: {total_params}\n")
                f.write(f"Trainable parameters: {trainable_params}\n\n")
                
                f.write("Parameter Statistics:\n")
                for name, stats in param_stats.items():
                    f.write(f"\n{name}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"  {stat_name}: {value:.6f}\n")
            
            print(f"Model analysis saved to {os.path.join(self.log_dir, filename)}")
            
        except Exception as e:
            print(f"Error analyzing model: {e}")