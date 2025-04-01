import pygame
import math

class Tower:
    """Base Tower class with common properties and methods"""
    def __init__(self, position, range, damage, attack_speed, cost):
        self.position = position
        self.range = range
        self.damage = damage
        self.attack_speed = attack_speed  # attacks per second
        self.cost = cost
        self.level = 1
        self.cooldown = 0
        self.target = None
        
    def upgrade(self):
        """Upgrade tower stats"""
        self.level += 1
        self.damage += self.damage * 0.2  # 20% damage increase per level
        self.range += 10  # Range increase
        self.attack_speed *= 1.1  # 10% attack speed increase
        return self.get_upgrade_cost()
        
    def get_upgrade_cost(self):
        """Calculate cost for next upgrade"""
        return int(self.cost * 0.7 * self.level)
        
    def get_position(self):
        return self.position
        
    def find_target(self, enemies):
        """Find a valid target within range"""
        closest_enemy = None
        closest_distance = float('inf')
        
        for enemy in enemies:
            if not enemy.is_alive:
                continue
                
            distance = math.sqrt((enemy.position[0] - self.position[0])**2 + 
                                 (enemy.position[1] - self.position[1])**2)
            if distance <= self.range and distance < closest_distance:
                closest_enemy = enemy
                closest_distance = distance
                
        return closest_enemy
        
    def attack(self, enemies, delta_time):
        """Attack enemies within range"""
        # Update cooldown
        self.cooldown -= delta_time
        
        # Can't attack if on cooldown
        if self.cooldown > 0:
            return 0
        
        # Find target
        self.target = self.find_target(enemies)
        
        # Attack target if found
        if self.target:
            reward = self.target.take_damage(self.damage)
            self.cooldown = 1.0 / self.attack_speed  # Reset cooldown
            return reward
            
        return 0
        
    def draw(self, surface):
        """Draw the tower on the surface"""
        # Base tower
        pygame.draw.rect(surface, (100, 100, 100), 
                        (self.position[0] - 15, self.position[1] - 15, 30, 30))
        
        # Range indicator (only when selected)
        if hasattr(self, 'selected') and self.selected:
            pygame.draw.circle(surface, (100, 100, 200, 100), 
                              (int(self.position[0]), int(self.position[1])), 
                              self.range, 1)
        
        # Attack line if targeting something
        if self.target and self.target.is_alive and self.cooldown > 0.8 / self.attack_speed:
            pygame.draw.line(surface, (255, 255, 0), 
                            self.position, self.target.position, 2)

class BasicTower(Tower):
    """Basic tower implementation for testing"""
    def __init__(self, position):
        super().__init__(position, range=150, damage=20, attack_speed=1.0, cost=100)
        self.tower_type = "Basic"
        self.color = (0, 100, 200)
        
    def draw(self, surface):
        # Draw base tower
        super().draw(surface)
        
        # Draw unique features for this tower
        pygame.draw.circle(surface, self.color, 
                          (int(self.position[0]), int(self.position[1])), 10)
        
        # Draw level indicator
        font = pygame.font.SysFont("Arial", 12)
        level_text = font.render(str(self.level), True, (255, 255, 255))
        surface.blit(level_text, (self.position[0] - 4, self.position[1] - 6))
