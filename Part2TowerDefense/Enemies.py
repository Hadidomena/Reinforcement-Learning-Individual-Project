import pygame
import math

class Enemy:
    """Base Enemy class with common properties and methods"""
    def __init__(self, position, health, speed, damage, reward):
        self.position = position
        self.max_health = health
        self.health = health
        self.speed = speed
        self.damage = damage
        self.reward = reward
        self.is_alive = True
        self.path = []
        self.path_index = 0
        
    def take_damage(self, amount):
        """Take damage and return reward if killed"""
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
            return self.reward
        return 0
    
    def move(self, delta_time):
        """Move enemy along its path"""
        if not self.is_alive or self.path_index >= len(self.path):
            return False
            
        target = self.path[self.path_index]
        direction_x = target[0] - self.position[0]
        direction_y = target[1] - self.position[1]
        distance = math.sqrt(direction_x**2 + direction_y**2)
        
        if distance < self.speed * delta_time:
            self.position = target
            self.path_index += 1
            return True
        else:
            move_x = direction_x / distance * self.speed * delta_time
            move_y = direction_y / distance * self.speed * delta_time
            self.position = (self.position[0] + move_x, self.position[1] + move_y)
            return True
            
    def set_path(self, path):
        self.path = path
        self.path_index = 0

    def draw(self, surface):
        """Draw the enemy on the surface"""
        pygame.draw.circle(surface, (255, 0, 0), (int(self.position[0]), int(self.position[1])), 10)
        
        # Health bar
        health_ratio = self.health / self.max_health
        bar_width = 30
        bar_height = 5
        pygame.draw.rect(surface, (255, 0, 0), (
            int(self.position[0] - bar_width/2),
            int(self.position[1] - 20),
            bar_width,
            bar_height
        ))
        pygame.draw.rect(surface, (0, 255, 0), (
            int(self.position[0] - bar_width/2),
            int(self.position[1] - 20),
            int(bar_width * health_ratio),
            bar_height
        ))

class MediumEnemy(Enemy):
    """Medium health/speed enemy that follows BFS path"""
    def __init__(self, position, entrance):
        super().__init__(position, health=100, speed=50, damage=10, reward=20)
        self.entrance = entrance
        
    def set_path_from_maze(self, maze):
        """Set path using BFS algorithm"""
        self.path = self.bfs_path(maze, self.entrance)
    
    def bfs_path(self, maze, start):
        """BFS pathfinding algorithm"""
        queue = [start]
        visited = {start: None}
        
        while queue:
            current = queue.pop(0)
            if current == maze.exit:
                break
                
            for neighbor in maze.get_neighbors(current):
                if neighbor not in visited and not maze.is_blocked(neighbor):
                    queue.append(neighbor)
                    visited[neighbor] = current
        
        # Reconstruct path
        path = []
        current = maze.exit
        while current and current != start:
            path.append(current)
            current = visited.get(current)
            
        path.reverse()
        return path

class TankEnemy(Enemy):
    """High health, slow enemy that follows DFS path"""
    def __init__(self, position, entrance):
        super().__init__(position, health=200, speed=30, damage=15, reward=35)
        self.entrance = entrance
        
    def set_path_from_maze(self, maze):
        """Set path using DFS algorithm"""
        self.path = self.dfs_path(maze, self.entrance)
    
    def dfs_path(self, maze, start):
        """DFS pathfinding algorithm"""
        stack = [start]
        visited = {start: None}
        
        while stack:
            current = stack.pop()
            if current == maze.exit:
                break
                
            neighbors = maze.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and not maze.is_blocked(neighbor):
                    stack.append(neighbor)
                    visited[neighbor] = current
        
        # Reconstruct path
        path = []
        current = maze.exit
        while current and current != start:
            path.append(current)
            current = visited.get(current)
            
        path.reverse()
        return path

class FastEnemy(Enemy):
    """Fast but fragile enemy following right-hand path"""
    def __init__(self, position, entrance):
        super().__init__(position, health=50, speed=80, damage=5, reward=25)
        self.entrance = entrance
        
    def set_path_from_maze(self, maze):
        """Set path using right-hand rule"""
        self.path = self.right_hand_path(maze, self.entrance)
    
    def right_hand_path(self, maze, start):
        """Right-hand rule pathfinding algorithm"""
        path = [start]
        current = start
        direction = (0, 1)  # Initial direction (down)
        
        while current != maze.exit and len(path) < 1000:  # Prevent infinite loops
            # Try turning right
            right_dir = self._turn_right(direction)
            right_pos = (current[0] + right_dir[0], current[1] + right_dir[1])
            
            if not maze.is_blocked(right_pos):
                direction = right_dir
                current = right_pos
            # Try going straight
            elif not maze.is_blocked((current[0] + direction[0], current[1] + direction[1])):
                current = (current[0] + direction[0], current[1] + direction[1])
            # Try turning left
            else:
                direction = self._turn_left(direction)
                
            path.append(current)
                
        return path
    
    def _turn_right(self, direction):
        """Turn right from current direction"""
        if direction == (0, 1):    # Down
            return (1, 0)          # Right
        elif direction == (1, 0):  # Right
            return (0, -1)         # Up
        elif direction == (0, -1): # Up
            return (-1, 0)         # Left
        else:                      # Left
            return (0, 1)          # Down
    
    def _turn_left(self, direction):
        """Turn left from current direction"""
        if direction == (0, 1):    # Down
            return (-1, 0)         # Left
        elif direction == (-1, 0): # Left
            return (0, -1)         # Up
        elif direction == (0, -1): # Up
            return (1, 0)          # Right
        else:                      # Right
            return (0, 1)          # Down

class SuperTankEnemy(Enemy):
    """Slow and tanky enemy following left-hand path"""
    def __init__(self, position, entrance):
        super().__init__(position, health=300, speed=20, damage=20, reward=50)
        self.entrance = entrance
        
    def set_path_from_maze(self, maze):
        """Set path using left-hand rule"""
        self.path = self.left_hand_path(maze, self.entrance)
    
    def left_hand_path(self, maze, start):
        """Left-hand rule pathfinding algorithm"""
        path = [start]
        current = start
        direction = (0, 1)  # Initial direction (down)
        
        while current != maze.exit and len(path) < 1000:  # Prevent infinite loops
            # Try turning left
            left_dir = self._turn_left(direction)
            left_pos = (current[0] + left_dir[0], current[1] + left_dir[1])
            
            if not maze.is_blocked(left_pos):
                direction = left_dir
                current = left_pos
            # Try going straight
            elif not maze.is_blocked((current[0] + direction[0], current[1] + direction[1])):
                current = (current[0] + direction[0], current[1] + direction[1])
            # Try turning right
            else:
                direction = self._turn_right(direction)
                
            path.append(current)
                
        return path
    
    def _turn_right(self, direction):
        """Turn right from current direction"""
        if direction == (0, 1):    # Down
            return (1, 0)          # Right
        elif direction == (1, 0):  # Right
            return (0, -1)         # Up
        elif direction == (0, -1): # Up
            return (-1, 0)         # Left
        else:                      # Left
            return (0, 1)          # Down
    
    def _turn_left(self, direction):
        """Turn left from current direction"""
        if direction == (0, 1):    # Down
            return (-1, 0)         # Left
        elif direction == (-1, 0): # Left
            return (0, -1)         # Up
        elif direction == (0, -1): # Up
            return (1, 0)          # Right
        else:                      # Right
            return (0, 1)          # Down
