import matplotlib.pyplot as plt
from IPython import display
import numpy as np

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

    # Plot scores
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')
    ax1.plot(scores, 'b-', label='Score')
    ax1.plot(mean_scores, 'r-', label='Mean Score')
    
    # Add moving average for scores
    if len(scores) >= 5:
        window_size = min(5, len(scores))
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(scores)), moving_avg, 'g--', label=f'{window_size}-game Moving Avg')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Plot rewards - ensure non-zero Y-axis range
    ax2.set_title('Rewards per Game')
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Reward')
    
    # Calculate a reasonable y-axis range for rewards
    min_reward = min(rewards) if rewards else 0
    max_reward = max(rewards) if rewards else 1
    if max_reward - min_reward < 1:  # Avoid flat line by setting a minimum range
        max_reward = min_reward + 1 if min_reward < 0 else 1
    
    # Add a small buffer to the range
    y_range = max_reward - min_reward
    buffer = max(y_range * 0.1, 1)  # 10% buffer or minimum 1
    
    # Apply adjusted range with minimum span
    y_min = min_reward - buffer
    y_max = max_reward + buffer
    ax2.set_ylim(y_min, y_max)
    
    # Plot the reward data with a more visible line
    ax2.plot(rewards, 'b-', linewidth=1.5, label='Reward per Game')
    
    # Add moving average for rewards
    if len(rewards) >= 5:
        window_size = min(5, len(rewards))
        reward_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(rewards)), reward_avg, 'r--', 
                linewidth=1.5, label=f'{window_size}-game Moving Avg')
    
    # Add horizontal line at zero
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Show mean reward as text
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        ax2.text(0.02, 0.95, f'Mean Reward: {mean_reward:.2f}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5))
    
    ax2.legend(loc='upper left')

    plt.tight_layout()
    display.clear_output(wait=True)
    display.display(fig)
    plt.pause(.1)