import pygame
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog

# Constants
BlockSize = 32
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
GRAY = (200, 200, 200)

class LevelEditor:
    def __init__(self, w=1280, h=960):
        # Initialize pygame
        pygame.init()
        self.w = w
        self.h = h
        
        # Initialize level as empty grid - FIX: Use width for x dimension
        self.level = np.zeros((self.h // BlockSize, self.w // BlockSize), dtype=int)
        
        # Set up display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tower Defense - Level Editor')
        self.clock = pygame.time.Clock()
        
        # Tile type definitions
        self.EMPTY = 0
        self.PATH = 1
        self.ENTRANCE = 2
        self.EXIT = 3
        
        # Map tile values to colors
        self.colors = {
            self.EMPTY: BROWN,     # Brown for empty tiles
            self.PATH: BLACK,      # Black for paths
            self.ENTRANCE: RED,    # Red for entrances
            self.EXIT: GREEN       # Green for exit
        }
        
        self.tile_names = {
            self.EMPTY: "Empty",
            self.PATH: "Path",
            self.ENTRANCE: "Entrance",
            self.EXIT: "Exit"
        }
        
        # Font for UI text
        self.font = pygame.font.SysFont('Arial', 20)
        
        # Ensure levels directory exists
        self.levels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")
        self.custom_levels_dir = os.path.join(self.levels_dir, "selfmade")
        
        # Create directories if they don't exist
        os.makedirs(self.custom_levels_dir, exist_ok=True)
        
        # Initialize Tkinter for dialogs
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
    
    def display_level(self):
        """Display the current level grid"""
        # Fill the screen with background
        self.display.fill(WHITE)
        
        # Draw each tile
        for y in range(self.level.shape[0]):
            for x in range(self.level.shape[1]):
                tile_type = self.level[y, x]
                pygame.draw.rect(self.display, self.colors[tile_type], 
                                (x * BlockSize, y * BlockSize, BlockSize, BlockSize))
                # Draw grid lines
                pygame.draw.rect(self.display, GRAY, 
                                (x * BlockSize, y * BlockSize, BlockSize, BlockSize), 1)
        
        # Show instructions
        instructions = [
            "Click to cycle through tile types:",
            "0 - Empty (Brown)",
            "1 - Path (Black)",
            "2 - Entrance (Red)",
            "3 - Exit (Green)",
            "S - Save level",
            "ESC - Exit editor"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = self.font.render(text, True, BLACK)
            self.display.blit(text_surface, (10, 10 + i * 25))
        
        # Update the display
        pygame.display.flip()
    
    def handle_click(self, pos):
        """Handle mouse click to change tile type"""
        x, y = pos
        grid_x = x // BlockSize
        grid_y = y // BlockSize
        
        # Check if within grid bounds
        if 0 <= grid_x < self.level.shape[1] and 0 <= grid_y < self.level.shape[0]:
            # Cycle through tile types (0->1->2->3->0)
            self.level[grid_y, grid_x] = (self.level[grid_y, grid_x] + 1) % 4
            print(f"Changed tile ({grid_x}, {grid_y}) to {self.tile_names[self.level[grid_y, grid_x]]}")
    
    def save_level(self):
        """Save the current level to a file with custom name"""
        try:
            # Ask for level name using a dialog
            level_name = simpledialog.askstring("Save Level", "Enter level name:", parent=self.root)
            
            # If user canceled or entered empty name
            if not level_name:
                print("Save canceled or no name provided")
                return
                
            # Clean the filename (remove invalid characters)
            level_name = ''.join(c for c in level_name if c.isalnum() or c in [' ', '_'])
            
            # Create the full path
            file_path = os.path.join(self.custom_levels_dir, f"{level_name}.npy")
            
            # Save the level
            np.save(file_path, self.level)
            print(f"Level saved to {file_path}")
            
            # Show confirmation
            pygame.display.set_caption(f'Tower Defense - Level Editor - Saved: {level_name}')
        except Exception as e:
            print(f"Error saving level: {e}")
    
    def run(self):
        """Main editor loop"""
        running = True
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        self.save_level()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle tile changes on mouse click
                    self.handle_click(pygame.mouse.get_pos())
            
            # Display the level
            self.display_level()
            
            # Cap the frame rate
            self.clock.tick(30)
        
        # Remove pygame.quit() - Main menu is managing the pygame instance
        # This fixes the crash when returning to main menu


if __name__ == "__main__":
    editor = LevelEditor()
    editor.run()
