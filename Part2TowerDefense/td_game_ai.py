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
        
        # Start spawning immediately by setting time values
        pg.time.delay(10)  # Small delay to ensure time differences
        
        valid_pos = self.get_valid_positions()
        print(f"Reset complete. Valid positions: {len(valid_pos)}")
        return valid_pos
        
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
        
        # Track enemies that reached the end this step
        if not hasattr(self.world, 'enemies_reached_end'):
            self.world.enemies_reached_end = 0

        # Handle only quit event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
                
        # Process action
        action_idx = action.argmax()
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
                  # Base reward for placing a turret + strategic position bonus (increased scale)
                placement_reward = 10 + (position_quality * 5)
                reward += placement_reward

        elif action_idx < c.COLS * c.ROWS + len(self.turret_group):  # Upgrade turret
            turret_idx = action_idx - (c.COLS * c.ROWS)
            for i, turret in enumerate(self.turret_group):
                # Only upgrade if not at max level
                if (
                    i == turret_idx
                    and self.world.money >= c.UPGRADE_COST
                    and hasattr(turret, "upgrade_level")
                    and turret.upgrade_level < c.TURRET_LEVELS
                ):
                    turret.upgrade()
                    self.world.money -= c.UPGRADE_COST
                    reward += 15 + (turret.upgrade_level * 5)  # Progressive reward for higher upgrades (increased scale)

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
                    self.last_enemy_spawn = current_time

        # Calculate enemies killed (main objective)
        enemies_killed_this_step = enemy_initial_count + self.world.enemies_reached_end - len(self.enemy_group)
        if enemies_killed_this_step > 0:            # Reward for killing enemies (increased scale)
            reward += enemies_killed_this_step * 20
          # Penalty for letting enemies through (increased scale)
        health_lost = initial_health - self.world.health
        if health_lost > 0:
            reward -= health_lost * 5

        # Check level completion
        if self.world.check_level_complete():
            self.world.money += c.LEVEL_COMPLETE_REWARD            # Significant reward for completing a level (increased scale)
            level_reward = 50 + (self.world.level * 20)  # Progressive reward for later levels
            reward += level_reward
            self.world.level += 1
            self.world.reset_level()
            self.world.process_enemies()
            self.last_enemy_spawn = pg.time.get_ticks()        # Check game over conditions
        if self.world.health <= 0:
            game_over = True
            reward = -100  # Base negative reward for losing
            # Add multiplier based on how far the agent got
            reward -= 20 * max(0, 30 - self.world.level)  # More negative if lost early
        elif self.frame_iteration > 1500:
            game_over = True
            reward = self.world.level * 20  # Reward based on progress
            
        # Update rewards - track both per-game rewards and cumulative rewards
        if reward != 0:  # Debug print to verify rewards are being generated
            print(f"Reward in this step: {reward}")
        
        # Track rewards for the current game only
        self.game_reward += reward
        
        # Also track total rewards across all games (optional)
        if not hasattr(self, 'total_reward'):
            self.total_reward = 0
        self.total_reward += reward

        # Update display
        self.update_ui()
        self.clock.tick(c.FPS * self.game_speed)

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

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_rewards = []
    total_score = 0
    total_rewards = 0  # Track cumulative rewards for mean calculation
    record = 0
    agent = TowerDefenseAgent()
    game = TowerDefenseAI()
    
    while True:
        # Get old state
        state_old = agent.get_state(game.world, game.enemy_group, game.turret_group)
        valid_positions = game.get_valid_positions()

        # Get move
        final_move = agent.get_action(state_old, valid_positions)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game.world, game.enemy_group, game.turret_group)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save_model()            # Scale up rewards for better visualization (multiplier of 10)
            scaled_reward = game.game_reward * 10
              # Track mean rewards too
            total_rewards += scaled_reward
            mean_reward = total_rewards / agent.n_games
            
            print(f'Game {agent.n_games}, Score {score}, Reward {scaled_reward:.1f}, Mean Reward {mean_reward:.1f}, Record: {record}')

            plot_scores.append(score)
            plot_rewards.append(scaled_reward)  # Use scaled per-game rewards for plotting
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, plot_rewards)

if __name__ == '__main__':
    train()
