import pygame as pg
import json
from world import World
from enemy import Enemy
from turret import Turret, TurretSlow, TurretPowerful
import constants as c
from td_agent import TowerDefenseAgent
from td_helper import plot
import os
import numpy as np
import random
import torch
from perks import initialize_perks, get_random_perks

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize pygame first
pg.init()

# Set display mode BEFORE loading images with convert_alpha
screen = pg.display.set_mode((c.SCREEN_WIDTH + c.SIDE_PANEL, c.SCREEN_HEIGHT))
pg.display.set_caption("Tower Defence AI Training")

# Now load assets
map_image = pg.image.load(os.path.join(SCRIPT_DIR, 'levels/level.png')).convert_alpha()

# Dynamically load turret types from turrets directory
TURRETS_DIR = os.path.join(SCRIPT_DIR, 'assets/images/turrets')
turret_type_dirs = [d for d in os.listdir(TURRETS_DIR) if os.path.isdir(os.path.join(TURRETS_DIR, d))]
turret_types = []
turret_spritesheets_dict = {}
cursor_turrets = {}
for turret_dir in turret_type_dirs:
    turret_type = turret_dir.replace('turret', '').lower() if turret_dir.lower().startswith('turret') else turret_dir.lower()
    if not turret_type:
        turret_type = 'basic'
    turret_types.append(turret_type)
    # Load spritesheets
    spritesheets = []
    for x in range(1, c.TURRET_LEVELS + 1):
        sheet_path = os.path.join(TURRETS_DIR, turret_dir, f'turret_{x}.png')
        spritesheets.append(pg.image.load(sheet_path).convert_alpha())
    turret_spritesheets_dict[turret_type] = spritesheets
    # Load cursor image
    cursor_path = os.path.join(TURRETS_DIR, turret_dir, 'cursor_turret.png')
    cursor_turrets[turret_type] = pg.image.load(cursor_path).convert_alpha()

enemy_images = {
    "weak": pg.image.load(os.path.join(SCRIPT_DIR, 'assets/images/enemies/enemy_1.png')).convert_alpha(),
    "medium": pg.image.load(os.path.join(SCRIPT_DIR, 'assets/images/enemies/enemy_2.png')).convert_alpha(),
    "strong": pg.image.load(os.path.join(SCRIPT_DIR, 'assets/images/enemies/enemy_3.png')).convert_alpha(),
    "elite": pg.image.load(os.path.join(SCRIPT_DIR, 'assets/images/enemies/enemy_4.png')).convert_alpha()
}

shot_fx = pg.mixer.Sound(os.path.join(SCRIPT_DIR, 'assets/audio/shot.wav'))
shot_fx.set_volume(0.1)  # Lower volume for training

with open(os.path.join(SCRIPT_DIR, 'levels/level.tmj')) as file:
    world_data = json.load(file)

# Load perk images
perks_enabled_at_level = 3  # Start offering perks at level 3
perk_frequency = 3  # Offer perks every 3 levels

class TowerDefenseAI:
    def __init__(self):
        # Use the already initialized display
        self.screen = screen
        pg.display.set_caption("Tower Defence AI Training")
        self.clock = pg.time.Clock()
        self.game_speed = 10
        self.level_started = True  # Force start immediately
        self.last_enemy_spawn = pg.time.get_ticks()        # Assign assets as instance variables
        self.turret_spritesheets_dict = turret_spritesheets_dict
        self.shot_fx = shot_fx
        self.enemy_images = enemy_images
        
        # Initialize perk system
        self.perks_dict = initialize_perks(SCRIPT_DIR)
        self.perk_selection_active = False
        self.perk_options = []
        self.chosen_perks = []

        self.reset()
        print("AI mode initialized")
        
    def reset(self):
        # Force AI mode
        self.world = World(world_data, map_image)
        self.world.process_data()
        self.world.process_enemies()
        
        self.enemy_group = pg.sprite.Group()
        self.turret_group = pg.sprite.Group()
        self.frame_iteration = 0
        
        # Store the previous game's reward for plotting before resetting
        self.game_reward = 0  # This will track rewards just for this game/episode
        
        # Critical: Force AI mode
        self.level_started = True
        self.world.level_started = True  # If this exists in World class
        self.world.game_speed = self.game_speed
        self.last_enemy_spawn = pg.time.get_ticks()
        self.world.level = 1
        
        # Initialize tracking variables
        self.enemies_spawned = 0
        self.enemies_killed = 0
        self.initial_turret_count = 0
        self.perks_dict = initialize_perks(SCRIPT_DIR)
        self.perk_selection_active = False
        self.perk_options = []
        
        # Start spawning immediately by setting time values
        pg.time.delay(10)  # Small delay to ensure time differences
        
        valid_pos = self.get_valid_positions()
        print(f"Reset complete. Valid positions: {len(valid_pos)}")
        return valid_pos
    
    def choose_random_perk(self):
        """AI automatically chooses a random perk"""
        if not self.perk_options:
            return
            
        # Select a random perk
        perk = random.choice(self.perk_options)
        
        # Apply the perk effect
        perk.apply_effect(self.world, self.turret_group)
        self.chosen_perks.append(perk.name)
        
        # End perk selection
        self.perk_selection_active = False
        
        return perk.name
        
    def get_valid_positions(self):
        valid_positions = []
        for y in range(c.ROWS):
            for x in range(c.COLS):
                tile_num = (y * c.COLS) + x
                if self.world.tile_map[tile_num] == 7:
                    space_is_free = True
                    for turret in self.turret_group:
                        if (x, y) == (turret.tile_x, turret.tile_y):
                            space_is_free = False
                            break
                    if space_is_free:
                        valid_positions.append(tile_num)
        return valid_positions
        
    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False
        enemy_initial_count = len(self.enemy_group)
        initial_health = self.world.health
        initial_money = self.world.money
        
        # Track enemies that reached the end this step
        if not hasattr(self.world, 'enemies_reached_end'):
            self.world.enemies_reached_end = 0

        # Handle only quit event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
        
        # Handle perk selection if active
        if self.perk_selection_active:
            chosen_perk = self.choose_random_perk()
            print(f"AI selected perk: {chosen_perk}")
            # More moderate reward for getting a perk
            reward += 2.5
                
        # Process action with improved rewards
        action_idx = action.argmax()
        action_successful = False
        
        if action_idx < c.COLS * c.ROWS:  # Place turret
            x = action_idx % c.COLS
            y = action_idx // c.COLS
            valid_positions = self.get_valid_positions()
            
            if self.world.money >= c.BUY_COST and (y * c.COLS + x) in valid_positions:
                # Strategic positioning reward - check distance to waypoints
                position_quality = self._evaluate_turret_position(x, y)
                
                # Randomly choose between turret types
                rand_val = random.random()
                if rand_val < 0.33:  # ~33% chance for each type
                    turret_type = 'basic'
                    new_turret = Turret(self.turret_spritesheets_dict[turret_type], x, y, self.shot_fx)
                elif rand_val < 0.67:
                    turret_type = 'slow'
                    new_turret = TurretSlow(self.turret_spritesheets_dict[turret_type], x, y, self.shot_fx)
                else:
                    turret_type = 'powerful'
                    new_turret = TurretPowerful(self.turret_spritesheets_dict[turret_type], x, y, self.shot_fx)
                
                self.turret_group.add(new_turret)
                self.world.money -= c.BUY_COST
                
                # Improved reward calculation with more moderate values
                base_reward = 2.0  # Reduced from 5.0
                position_bonus = position_quality * 1.5  # Reduced from 3.0
                timing_bonus = self._calculate_timing_bonus() * 0.5  # Scaled down
                placement_reward = base_reward + position_bonus + timing_bonus
                reward += placement_reward
                action_successful = True
                
            else:
                # Smaller penalty for invalid actions
                reward -= 0.5  # Reduced from 1.0

        elif action_idx < c.COLS * c.ROWS + len(self.turret_group):  # Upgrade turret
            turret_idx = action_idx - (c.COLS * c.ROWS)
            turrets = list(self.turret_group)
            if turret_idx < len(turrets):
                turret = turrets[turret_idx]
                if (self.world.money >= c.UPGRADE_COST and 
                    hasattr(turret, "upgrade_level") and 
                    turret.upgrade_level < c.TURRET_LEVELS):
                    turret.upgrade()
                    self.world.money -= c.UPGRADE_COST
                    # More moderate progressive reward for upgrades
                    upgrade_reward = 1.5 + (turret.upgrade_level * 1.0)  # Reduced from 3.0 + level*2.0
                    reward += upgrade_reward
                    action_successful = True
                else:
                    # Small penalty for invalid upgrade
                    reward -= 0.25  # Reduced from 0.5

        # Reset enemies reached end counter for this step
        self.world.enemies_reached_end = 0

        # Update game state multiple times for increased speed
        for _ in range(self.game_speed):
            # Store health before update to track lost health
            pre_update_health = self.world.health
            
            self.enemy_group.update(self.world)
            self.turret_group.update(self.enemy_group, self.world)
            
            # Count enemies that reached the end this update
            if pre_update_health > self.world.health:
                self.world.enemies_reached_end += pre_update_health - self.world.health

            # Spawn enemies
            current_time = pg.time.get_ticks()
            if current_time - self.last_enemy_spawn > c.SPAWN_COOLDOWN / self.game_speed:
                if self.world.spawned_enemies < len(self.world.enemy_list):
                    enemy_type = self.world.enemy_list[self.world.spawned_enemies]
                    enemy = Enemy(enemy_type, self.world.waypoints, self.enemy_images)
                    self.enemy_group.add(enemy)
                    self.world.spawned_enemies += 1
                    self.enemies_spawned += 1
                    self.last_enemy_spawn = current_time

        # Calculate enemies killed (main objective)
        enemies_killed_this_step = enemy_initial_count + self.world.enemies_reached_end - len(self.enemy_group)
        if enemies_killed_this_step > 0:
            # More moderate reward for killing enemies
            kill_reward = enemies_killed_this_step * 4.0  # Reduced from 8.0
            reward += kill_reward
            self.enemies_killed += enemies_killed_this_step

        # Penalty for letting enemies through with slightly lower penalties
        health_lost = initial_health - self.world.health
        if health_lost > 0:
            # Less severe penalty based on remaining health
            health_penalty = health_lost * (1.0 + (100 - self.world.health) * 0.03)  # Reduced from 2.0 + 0.05
            reward -= health_penalty        # Survival bonus - bardziej stała nagroda za przetrwanie
        if self.world.health > 0:
            # Stała podstawa + mniejszy wpływ zdrowia dla większej stabilności
            base_survival = 0.1  # Stała nagroda za samo przetrwanie 
            health_bonus = 0.05 * (self.world.health / 100.0)  # Mniejszy wpływ aktualnego zdrowia
            survival_bonus = base_survival + health_bonus
            reward += survival_bonus

        # Efficiency bonus for maintaining good kill ratio
        if self.enemies_spawned > 0:
            kill_ratio = self.enemies_killed / self.enemies_spawned
            if kill_ratio > 0.8:  # High efficiency
                reward += 1.0 * kill_ratio  # Reduced from 2.0        # Check level completion
        if self.world.check_level_complete():
            self.world.money += c.LEVEL_COMPLETE_REWARD
            # Liniowy system nagród - stała nagroda plus niewielki mnożnik za poziom
            base_reward = 30.0  # Stała podstawowa nagroda
            level_multiplier = 2.0  # Niewielki mnożnik za poziom
            reward += base_reward + min(15, self.world.level) * level_multiplier  # Ograniczenie wpływu poziomów
            self.world.level += 1
            
            # Check if we should offer perks at this level
            if self.world.level >= perks_enabled_at_level and self.world.level % perk_frequency == 0:
                self.perk_options = get_random_perks(self.perks_dict, 3)
                if self.perk_options:  # Make sure we have perks to show
                    self.perk_selection_active = True
                    print(f"Perk selection active - level {self.world.level}")
            
            self.world.reset_level()
            self.world.process_enemies()
            self.last_enemy_spawn = pg.time.get_ticks()
            
        # Check game over conditions
        if self.world.health <= 0:
            game_over = True
            # More moderate penalty based on performance
            death_penalty = -25.0  # Reduced from -50.0
            # Less penalty if made some progress
            progress_bonus = min(10.0, self.world.level * 1.5 + self.enemies_killed * 0.25)  # Reduced from 20.0, 2.0, 0.5
            reward = death_penalty + progress_bonus
            
        elif self.frame_iteration > 2000:  # Increased timeout
            game_over = True
            # More moderate reward based on survival and progress
            reward += self.world.level * 3.0 + self.enemies_killed * 0.1  # Reduced from 5.0, 0.2
              # Remove aggressive reward scaling and clipping to preserve learning signal
        # reward = reward * 0.5  # REMOVED - was limiting learning
            
        # Track rewards for the current game only
        self.game_reward += reward

        # Update display less frequently for speed
        if self.frame_iteration % 5 == 0:  # Update every 5 frames
            self.update_ui()
            self.clock.tick(c.FPS * self.game_speed)

        # More reasonable reward clipping to prevent extreme values only
        reward = max(-200, min(200, reward))  # Increased range from [-50, 50] to [-200, 200]
        
        return reward, game_over, self.world.level

    def update_ui(self):
        self.world.draw(self.screen)
        self.enemy_group.draw(self.screen)
        for turret in self.turret_group:
            turret.draw(self.screen)
        pg.display.flip()

    def _evaluate_turret_position(self, x, y):
        """Calculate strategic value of a turret position based on proximity to path"""
        if not hasattr(self.world, 'waypoints') or not self.world.waypoints:
            return 0.5  # Default medium value if no waypoints
            
        # Calculate minimum distance to any waypoint
        min_distance = float('inf')
        for waypoint in self.world.waypoints:
            # Convert tile coordinates to pixel coordinates for distance calculation
            turret_x = x * c.TILE_SIZE + c.TILE_SIZE // 2
            turret_y = y * c.TILE_SIZE + c.TILE_SIZE // 2
            dist = ((turret_x - waypoint[0]) ** 2 + (turret_y - waypoint[1]) ** 2) ** 0.5
            min_distance = min(min_distance, dist)
        
        # Normalize distance: closer is better, but not too close
        # Ideal distance is around 1-2 tiles away from path
        ideal_dist = c.TILE_SIZE * 1.5
        position_score = 0
        
        if min_distance < c.TILE_SIZE * 0.5:  # Too close to path
            position_score = 0.5
        elif min_distance < c.TILE_SIZE * 3:  # Good range
            position_score = 2.0 - abs(min_distance - ideal_dist) / (c.TILE_SIZE * 2)
        else:  # Too far from path
            position_score = max(0.1, 1.0 - (min_distance - c.TILE_SIZE * 3) / (c.TILE_SIZE * 5))
            
        return position_score

    def _calculate_timing_bonus(self):
        """Calculate bonus based on when turret is placed"""
        enemy_count = len(self.enemy_group)
        if enemy_count > 5:  # Many enemies approaching
            return 2.0
        elif enemy_count > 2:  # Some enemies
            return 1.0
        elif self.world.spawned_enemies < len(self.world.enemy_list) // 2:  # Early game
            return 1.5
        else:
            return 0.5

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_rewards = []
    total_score = 0
    total_rewards = 0
    record = 0
    agent = TowerDefenseAgent()
    game = TowerDefenseAI()
    
    # Initialize with more exploration focus to discover better policies
    agent.epsilon = 95  # Higher initial exploration rate
    
    # Add performance tracking with longer window for better stability
    recent_scores = []
    recent_rewards = []
    recent_levels = []
    performance_window = 20  # Track last 20 games (increased from 10)
    
    # Initialize training analyzer
    try:
        from td_helper import TrainingAnalyzer
        analyzer = TrainingAnalyzer(agent)
        has_analyzer = True
        print("Training analyzer initialized for enhanced monitoring")
    except Exception as e:
        print(f"Warning: Training analyzer not available: {e}")
        has_analyzer = False
        
    # Load best model if available for continued training
    best_models = [f for f in os.listdir('./models') if f.startswith('td_model_best_')]
    if best_models:
        # Sort by score (extract number from filename)
        best_model = sorted(best_models, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)[0]
        try:
            agent.load_model(best_model)
            print(f"Loaded best model: {best_model} for continued training")
        except Exception as e:
            print(f"Error loading best model: {e}")
    
    try:  # Add overall try/except to handle training interruptions gracefully
        while True:
            # Get old state
            state_old = agent.get_state(game.world, game.enemy_group, game.turret_group)
            valid_positions = game.get_valid_positions()

            # Get move
            final_move = agent.get_action(state_old, valid_positions)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game.world, game.enemy_group, game.turret_group)

            # Train short memory with enhanced error checking
            try:
                # Check for extreme states before training
                if torch.is_tensor(state_old) and torch.max(torch.abs(state_old)) > 100:
                    print(f"Warning: Extreme values in state_old: {torch.max(torch.abs(state_old)):.1f}. Clipping.")
                    state_old = torch.clamp(state_old, -100, 100)
                
                # Train with error handling    
                loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
                if loss is not None and loss > 5:  # Lower threshold for warning (was 10)
                    print(f"Warning: High loss detected: {loss:.4f}. This may indicate instability.")
            except Exception as e:
                print(f"Error during short memory training: {e}")
                loss = None            # Remember experience for batch learning with enhanced reward shaping
            try:
                # Prepare game stats for reward shaping
                game_stats = {
                    'score': score,
                    'level': game.world.level,
                    'money': game.world.money,
                    'health': game.world.health,
                    'enemies_killed': game.enemies_killed,
                    'game_over': done
                }
                agent.remember(state_old, final_move, reward, state_new, done, game_stats)
            except Exception as e:
                print(f"Error storing experience: {e}")

            if done:
                # Train long memory (batch learning) with error handling
                try:
                    long_memory_loss = agent.train_long_memory()
                except Exception as e:
                    print(f"Error during long memory training: {e}")
                    long_memory_loss = None
                
                # Track performance metrics
                current_reward = game.game_reward
                current_level = game.world.level
                current_enemies_killed = game.enemies_killed
                
                # Reset game for next episode
                valid_positions = game.reset()
                
                # Use the agent's custom logging function instead of directly incrementing n_games
                agent.log_episode_stats(current_reward, score)
                
                # Update tracking lists
                recent_scores.append(score)
                recent_rewards.append(current_reward)
                recent_levels.append(current_level)
                
                # Keep only recent performance for better trend analysis
                if len(recent_scores) > performance_window:
                    recent_scores.pop(0)
                    recent_rewards.pop(0)
                    recent_levels.pop(0)

                # Calculate statistics for logging and visualization
                total_rewards += current_reward
                mean_reward = total_rewards / agent.n_games
                recent_avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                recent_avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                recent_avg_level = sum(recent_levels) / len(recent_levels) if recent_levels else 0
                
                # Adaptive epsilon adjustment based on performance trends
                if agent.n_games % 5 == 0 and agent.n_games > 20:  # Wait for enough data
                    # Dynamic adjustment based on recent performance relative to best
                    if record > 0:
                        performance_ratio = recent_avg_score / record
                        
                        if performance_ratio > 0.9:  # Performing very well
                            # Reduce exploration more quickly
                            agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.98)
                        elif performance_ratio < 0.5 and agent.n_games > 30:
                            # Increase exploration if doing poorly after initial learning
                            agent.epsilon = min(60, agent.epsilon * 1.02)
                            print(f"⚠️ Increasing exploration to epsilon={agent.epsilon:.1f} due to poor performance")
                
                # Save best model
                if score > record:
                    record = score
                    agent.save_model(f'td_model_best_{score}.pth')
                    print(f"🏆 New record model saved! Score: {score}")
                    
                # Save model checkpoints periodically
                if agent.n_games % 20 == 0:
                    agent.save_model(f'td_model_checkpoint_{agent.n_games}.pth')
                    print(f"💾 Checkpoint saved at game {agent.n_games}")
                    
                    # Analyze model every 20 games if analyzer available
                    if has_analyzer:
                        try:
                            analyzer.save_model_analysis(agent.model)
                        except Exception as e:
                            print(f"Error analyzing model: {e}")
                
                # Enhanced logging with more metrics for better debugging
                print(f'Game {agent.n_games:4d} | '
                      f'Score: {score:3d} | '
                      f'Level: {current_level:2d} | '
                      f'Enemies killed: {current_enemies_killed:3d} | '
                      f'Reward: {current_reward:7.1f} | '
                      f'Recent Avg Score: {recent_avg_score:5.1f} | '
                      f'Recent Avg Level: {recent_avg_level:4.1f} | '
                      f'Epsilon: {agent.epsilon:4.1f} | '
                      f'Loss: {long_memory_loss or 0:.4f}')
                
                # Show active perks if any
                if hasattr(game, 'chosen_perks') and game.chosen_perks:
                    print(f'  Active perks: {", ".join(game.chosen_perks)}')
                    
                # Log stats with the analyzer
                if has_analyzer:
                    try:
                        analyzer.log_game_stats(agent.n_games, score, current_level, current_reward, agent.epsilon)
                    except Exception as e:
                        print(f"Error logging stats: {e}")

                # Update plots for visualization
                plot_scores.append(score)
                plot_rewards.append(current_reward)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                try:
                    plot(plot_scores, plot_mean_scores, plot_rewards)
                except Exception as e:
                    print(f"Error updating plot: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save final model on interrupt
        agent.save_model('td_model_interrupted.pth')
        print("Final model saved as td_model_interrupted.pth")
    except Exception as e:
        print(f"Training stopped due to error: {e}")
        # Save emergency backup on error
        try:
            agent.save_model('td_model_emergency_backup.pth')
            print("Emergency backup saved")
        except:
            print("Could not save emergency backup")
    
    print("Training complete!")

if __name__ == "__main__":
    train()