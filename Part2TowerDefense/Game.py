from collections import namedtuple
from enum import Enum
import pygame
import numpy as np
import os
from LevelEditor import LevelEditor

Point = namedtuple('Point', 'x, y')
BlockSize = 32
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
Level = None

class Button:
    """A simple button class for the menu system"""
    def __init__(self, text, x, y, width, height, color, hover_color, text_color):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.font = pygame.font.SysFont('Arial', 32)
        
    def draw(self, display):
        """Draw the button"""
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(display, color, (self.x, self.y, self.width, self.height), 0, 10)
        pygame.draw.rect(display, BLACK, (self.x, self.y, self.width, self.height), 2, 10)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        # Center text in button
        text_x = self.x + (self.width - text_surface.get_width()) // 2
        text_y = self.y + (self.height - text_surface.get_height()) // 2
        display.blit(text_surface, (text_x, text_y))
        
    def is_clicked(self, pos):
        """Check if the button is clicked"""
        return (self.x <= pos[0] <= self.x + self.width and 
                self.y <= pos[1] <= self.y + self.height)
                
    def check_hover(self, pos):
        """Check if mouse is hovering over button"""
        self.is_hovered = (self.x <= pos[0] <= self.x + self.width and 
                           self.y <= pos[1] <= self.y + self.height)


class MainMenu:
    """Main menu for the game"""
    def __init__(self, w=1280, h=960):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Tower Defense - Main Menu')
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Define colors
        self.light_blue = (100, 149, 237)
        self.dark_blue = (70, 130, 180)
        self.orange = (255, 165, 0)
        self.light_orange = (255, 195, 77)
        self.red = (220, 20, 60)
        self.light_red = (255, 99, 71)
        
        # Create buttons
        button_width = 300
        button_height = 80
        center_x = w // 2 - button_width // 2
        
        self.play_button = Button("Play Game", center_x, h // 2 - 100, 
                                  button_width, button_height, 
                                  self.light_blue, self.dark_blue, WHITE)
        
        self.editor_button = Button("Level Editor", center_x, h // 2, 
                                   button_width, button_height,
                                   self.orange, self.light_orange, BLACK)
        
        self.quit_button = Button("Quit", center_x, h // 2 + 100,
                                 button_width, button_height,
                                 self.red, self.light_red, WHITE)
        
        # Level selection menu
        self.level_buttons = []
        self.back_button = Button("Back", 20, 20, 150, 60, 
                                 self.red, self.light_red, WHITE)
        
        # Setup levels directory paths
        self.levels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")
        self.custom_levels_dir = os.path.join(self.levels_dir, "selfmade")
        
        # Create directories if they don't exist
        os.makedirs(self.levels_dir, exist_ok=True)
        os.makedirs(self.custom_levels_dir, exist_ok=True)
        
        # Font for UI text
        self.title_font = pygame.font.SysFont('Arial', 64)
        self.font = pygame.font.SysFont('Arial', 32)
        
        # Current screen state
        self.current_screen = "main"  # "main", "level_select", "game"
        
    def get_available_levels(self):
        """Get a list of all available levels"""
        levels = []
        
        # Check built-in levels
        if os.path.exists(self.levels_dir):
            for file in os.listdir(self.levels_dir):
                if file.endswith('.npy') and os.path.isfile(os.path.join(self.levels_dir, file)):
                    name = os.path.splitext(file)[0]
                    levels.append({
                        'name': name,
                        'path': os.path.join(self.levels_dir, file),
                        'custom': False
                    })
        
        # Check custom levels
        if os.path.exists(self.custom_levels_dir):
            for file in os.listdir(self.custom_levels_dir):
                if file.endswith('.npy') and os.path.isfile(os.path.join(self.custom_levels_dir, file)):
                    name = os.path.splitext(file)[0]
                    levels.append({
                        'name': name,
                        'path': os.path.join(self.custom_levels_dir, file),
                        'custom': True
                    })
        
        return levels
    
    def create_level_buttons(self):
        """Create buttons for available levels"""
        self.level_buttons = []
        levels = self.get_available_levels()
        
        button_width = 350
        button_height = 60
        center_x = self.w // 2 - button_width // 2
        start_y = 150
        spacing = 70
        
        for i, level in enumerate(levels):
            y_pos = start_y + i * spacing
            label = f"{level['name']} ({'Custom' if level['custom'] else 'Default'})"
            button = Button(label, center_x, y_pos, button_width, button_height,
                           self.light_blue, self.dark_blue, WHITE)
            self.level_buttons.append({'button': button, 'level': level})
    
    def draw_main_menu(self):
        """Draw the main menu screen"""
        self.display.fill(WHITE)
        
        # Draw title
        title_surface = self.title_font.render("Tower Defense", True, BLACK)
        title_x = self.w // 2 - title_surface.get_width() // 2
        self.display.blit(title_surface, (title_x, 80))
        
        # Draw buttons
        self.play_button.draw(self.display)
        self.editor_button.draw(self.display)
        self.quit_button.draw(self.display)
        
        pygame.display.flip()
        
    def draw_level_select(self):
        """Draw the level selection screen"""
        self.display.fill(WHITE)
        
        # Draw title
        title_surface = self.title_font.render("Select Level", True, BLACK)
        title_x = self.w // 2 - title_surface.get_width() // 2
        self.display.blit(title_surface, (title_x, 40))
        
        # Draw back button
        self.back_button.draw(self.display)
        
        # Draw level buttons
        for item in self.level_buttons:
            item['button'].draw(self.display)
            
        # If no levels available
        if not self.level_buttons:
            text = self.font.render("No levels found. Create one in the Level Editor!", True, BLACK)
            text_x = self.w // 2 - text.get_width() // 2
            self.display.blit(text, (text_x, self.h // 2))
        
        pygame.display.flip()
    
    def handle_main_menu_events(self):
        """Handle events for the main menu"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEMOTION:
                # Update button hover states
                mouse_pos = pygame.mouse.get_pos()
                self.play_button.check_hover(mouse_pos)
                self.editor_button.check_hover(mouse_pos)
                self.quit_button.check_hover(mouse_pos)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if self.play_button.is_clicked(mouse_pos):
                    # Create level buttons before showing level select
                    self.create_level_buttons()
                    self.current_screen = "level_select"
                    
                elif self.editor_button.is_clicked(mouse_pos):
                    # Launch level editor
                    self.launch_level_editor()
                    
                elif self.quit_button.is_clicked(mouse_pos):
                    self.running = False
    
    def handle_level_select_events(self):
        """Handle events for the level selection screen"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEMOTION:
                # Update button hover states
                mouse_pos = pygame.mouse.get_pos()
                self.back_button.check_hover(mouse_pos)
                
                for item in self.level_buttons:
                    item['button'].check_hover(mouse_pos)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if self.back_button.is_clicked(mouse_pos):
                    self.current_screen = "main"
                    
                else:
                    # Check if a level button was clicked
                    for item in self.level_buttons:
                        if item['button'].is_clicked(mouse_pos):
                            self.launch_game(item['level'])
                            break
    
    def launch_level_editor(self):
        """Launch the level editor"""
        editor = LevelEditor()
        editor.run()
        # After editor closes, return to main menu
        self.current_screen = "main"
        
    def launch_game(self, selected_level):
        """Launch the game with selected level"""
        pygame.display.set_caption('Tower Defense')
        
        try:
            level_data = np.load(selected_level['path'])
            print(f"Loading level: {selected_level['name']}")
            game = GameLoop(self.w, self.h, level_data)
            game.run()
        except Exception as e:
            print(f"Error loading level: {e}")
            # Don't start the game if level loading fails
        
        # After game closes, return to main menu
        self.current_screen = "main"
    
    def run(self):
        """Main menu loop"""
        while self.running:
            if self.current_screen == "main":
                self.draw_main_menu()
                self.handle_main_menu_events()
            elif self.current_screen == "level_select":
                self.draw_level_select()
                self.handle_level_select_events()
                
            self.clock.tick(30)
            
        pygame.quit()

class GameLoop:
    def __init__(self, w=1280, h=960, loaded_level=None):
        pygame.init()
        global Level
        self.w = w
        self.h = h
        
        if loaded_level is None:
            raise ValueError("No level provided")
        self.levels_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")
        self.custom_levels_dir = os.path.join(self.levels_dir, "selfmade")
        os.makedirs(self.levels_dir, exist_ok=True)
        os.makedirs(self.custom_levels_dir, exist_ok=True)
        loaded_height, loaded_width = loaded_level.shape
        target_height, target_width = self.h // BlockSize, self.w // BlockSize
        
        if loaded_height != target_height or loaded_width != target_width:
            print(f"Resizing level from {loaded_width}x{loaded_height} to {target_width}x{target_height}")
            resized_level = np.zeros((target_height, target_width), dtype=int)
            copy_height = min(loaded_height, target_height)
            copy_width = min(loaded_width, target_width)
            resized_level[:copy_height, :copy_width] = loaded_level[:copy_height, :copy_width]
            Level = resized_level
        else:
            Level = loaded_level
            
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Tower Defense')
        self.clock = pygame.time.Clock()
        self.score = 0
        self.displayLevel()
    
    def displayLevel(self):
        # Define color mapping
        EMPTY = 0
        PATH = 1
        ENTRANCE = 2
        EXIT = 3
        
        # Define colors for each tile type
        BROWN = (139, 69, 19)  # Color for empty tiles
        
        # Map tile values to colors
        colors = {
            EMPTY: BROWN,      # Brown for empty tiles
            PATH: BLACK,       # Black for paths
            ENTRANCE: RED,     # Red for entrances
            EXIT: GREEN        # Green for exit
        }
        
        # Get level dimensions
        level_height, level_width = Level.shape
        
        # Calculate block size to fit level on screen
        block_size_h = self.h // level_height
        block_size_w = self.w // level_width
        
        # Use the smaller of the two to maintain square tiles
        actual_block_size = min(block_size_h, block_size_w)
        
        # Calculate centering offsets
        offset_x = (self.w - level_width * actual_block_size) // 2
        offset_y = (self.h - level_height * actual_block_size) // 2
        
        # Fill the screen with background
        self.display.fill(WHITE)
        
        # Draw each tile
        for y in range(level_height):
            for x in range(level_width):
                tile_type = Level[y, x]
                pygame.draw.rect(self.display, colors[tile_type], 
                                (offset_x + x * actual_block_size, 
                                 offset_y + y * actual_block_size, 
                                 actual_block_size, actual_block_size))
        
        # Update the display
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        running = True
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle other events (keyboard, mouse) here
            
            # Update game state
            
            # Redraw the screen
            self.displayLevel()
            
            # Cap the frame rate
            self.clock.tick(30)
        
        pygame.quit()