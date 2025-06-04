import pygame as pg
from pygame.math import Vector2
import math
import constants as c
from enemy_data import ENEMY_DATA

class Enemy(pg.sprite.Sprite):
  def __init__(self, enemy_type, waypoints, images):
    pg.sprite.Sprite.__init__(self)
    self.waypoints = waypoints
    self.pos = Vector2(self.waypoints[0])
    self.target_waypoint = 1
    self.health = ENEMY_DATA.get(enemy_type)["health"]
    self.speed = ENEMY_DATA.get(enemy_type)["speed"]
    self.base_speed = self.speed
    self.slow_timer = 0
    
    # Initialize the image properties
    self.enemy_type = enemy_type
    self.original_image = images[enemy_type]
    self.image = self.original_image.copy()
    self.rect = self.image.get_rect()
    self.rect.center = self.pos

  def update(self, world):
    self.move(world)
    self.rotate()
    self.check_alive(world)
    # Handle slow effect
    if self.slow_timer > 0:
      self.slow_timer -= 1
      if self.slow_timer == 0:
        self.speed = self.base_speed

  def move(self, world):
    #define a target waypoint
    if self.target_waypoint < len(self.waypoints):
      self.target = Vector2(self.waypoints[self.target_waypoint])
      self.movement = self.target - self.pos
    else:
      #enemy has reached the end of the path
      self.kill()
      world.health -= 1
      world.missed_enemies += 1

    #calculate distance to target
    dist = self.movement.length()
    #check if remaining distance is greater than the enemy speed
    if dist >= (self.speed * world.game_speed):
      self.pos += self.movement.normalize() * (self.speed * world.game_speed)
    else:
      if dist != 0:
        self.pos += self.movement.normalize() * dist
      self.target_waypoint += 1

  def rotate(self):
    #calculate distance to next waypoint
    dist = self.target - self.pos
    #use distance to calculate angle
    self.angle = math.degrees(math.atan2(-dist[1], dist[0]))
    #rotate image and update rectangle
    self.image = pg.transform.rotate(self.original_image, self.angle)
    self.rect = self.image.get_rect()
    self.rect.center = self.pos

  def check_alive(self, world):
    if self.health <= 0:
      world.killed_enemies += 1
      # Reward based on enemy type
      reward_values = {
          "weak": 1,   # Weakest enemy gives 1 coin
          "medium": 2, # Medium enemy gives 2 coins
          "strong": 3, # Strong enemy gives 3 coins
          "elite": 4   # Elite enemy gives 4 coins
      }
      reward = reward_values.get(self.enemy_type, 1)  # Default to 1 if type not found
      world.money += reward
      self.kill()