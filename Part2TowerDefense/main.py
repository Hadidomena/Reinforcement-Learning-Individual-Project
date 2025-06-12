import pygame as pg
import json
from enemy import Enemy 
from world import World
from turret import Turret
from button import Button
import constants as c
import os
from perks import initialize_perks, get_random_perks
from asset_loader import load_turret_assets, load_enemy_images, load_ui_images

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((c.SCREEN_WIDTH + c.SIDE_PANEL, c.SCREEN_HEIGHT))
pg.display.set_caption("Tower Defence")

game_over = False
game_outcome = 0
level_started = False
last_enemy_spawn = pg.time.get_ticks()
placing_turrets = False
selected_turret = None
selected_turret_type = 0

perks_dict = initialize_perks(SCRIPT_DIR)
perk_selection_active = False
perk_options = []
perks_enabled_at_level = 3
perk_frequency = 3

map_image = pg.image.load(os.path.join(SCRIPT_DIR, 'levels/level.png')).convert_alpha()
turret_types, turret_spritesheets_dict, cursor_turrets = load_turret_assets(SCRIPT_DIR)
enemy_images = load_enemy_images(SCRIPT_DIR)
ui_images = load_ui_images(SCRIPT_DIR)

#load sounds
shot_fx = pg.mixer.Sound(os.path.join(SCRIPT_DIR, 'assets/audio/shot.wav'))
shot_fx.set_volume(0.5)

#load json data for level
with open(os.path.join(SCRIPT_DIR, 'levels/level.tmj')) as file:
    world_data = json.load(file)

#load fonts for displaying text on the screen
text_font = pg.font.SysFont("Consolas", 24, bold = True)
large_font = pg.font.SysFont("Consolas", 36)

#function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
  img = font.render(text, True, text_col)
  screen.blit(img, (x, y))

def display_data():
  #draw panel
  pg.draw.rect(screen, "maroon", (c.SCREEN_WIDTH, 0, c.SIDE_PANEL, c.SCREEN_HEIGHT))
  pg.draw.rect(screen, "grey0", (c.SCREEN_WIDTH, 0, c.SIDE_PANEL, c.SCREEN_HEIGHT), 2)
  #display data
  draw_text("LEVEL: " + str(world.level), text_font, "grey100", c.SCREEN_WIDTH + 10, 10)
  # Show number of enemies in this level
  draw_text("Enemies: " + str(len(world.enemy_list)), text_font, "grey100", c.SCREEN_WIDTH + 150, 10)
  screen.blit(ui_images['heart'], (c.SCREEN_WIDTH + 10, 35))
  draw_text(str(world.health), text_font, "grey100", c.SCREEN_WIDTH + 50, 40)
  screen.blit(ui_images['coin'], (c.SCREEN_WIDTH + 10, 65))
  draw_text(str(world.money), text_font, "grey100", c.SCREEN_WIDTH + 50, 70)
  
  # Show perk information
  if world.level >= perks_enabled_at_level:
      next_perk_level = ((world.level // perk_frequency) + 1) * perk_frequency
      levels_until_perk = next_perk_level - world.level
      if levels_until_perk == 0:
          perk_text = "Perk available this round!"
          text_color = "yellow"
      else:
          perk_text = f"Next perk in {levels_until_perk} rounds"
          text_color = "grey100"
      draw_text(perk_text, text_font, text_color, c.SCREEN_WIDTH + 10, 95)
  
def render_perk_selection():
    # Create a semi-transparent overlay
    overlay = pg.Surface((c.SCREEN_WIDTH, c.SCREEN_HEIGHT))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(150)
    screen.blit(overlay, (0, 0))
    
    # Draw perk selection panel
    panel_width = 600
    panel_height = 300
    panel_x = (c.SCREEN_WIDTH - panel_width) // 2
    panel_y = (c.SCREEN_HEIGHT - panel_height) // 2
    
    # Draw panel background
    pg.draw.rect(screen, "dodgerblue", 
                (panel_x, panel_y, panel_width, panel_height), 
                border_radius=15)
    pg.draw.rect(screen, "navy", 
                (panel_x, panel_y, panel_width, panel_height), 
                width=4, border_radius=15)
    
    # Draw heading
    draw_text("Choose a Perk", large_font, "white", panel_x + 200, panel_y + 20)
    
    # Calculate positions for three perk options
    option_width = 150
    option_spacing = 40
    starting_x = panel_x + (panel_width - (3 * option_width + 2 * option_spacing)) // 2
    
    # Draw each perk option
    selected_perk = None
    for i, perk in enumerate(perk_options):
        # Calculate position
        x = starting_x + i * (option_width + option_spacing)
        y = panel_y + 80
        
        # Draw option background (brighter if mouse is over it)
        perk_rect = pg.Rect(x, y, option_width, 180)
        mouse_pos = pg.mouse.get_pos()
        
        if perk_rect.collidepoint(mouse_pos):
            pg.draw.rect(screen, "skyblue", perk_rect, border_radius=10)
            if pg.mouse.get_pressed()[0]:
                selected_perk = perk
        else:
            pg.draw.rect(screen, "lightblue", perk_rect, border_radius=10)
        
        pg.draw.rect(screen, "white", perk_rect, width=2, border_radius=10)
        
        # Draw perk icon
        icon_rect = perk.image.get_rect()
        icon_rect.center = (x + option_width // 2, y + 80)
        screen.blit(perk.image, icon_rect)
        
        # Draw perk name
        draw_text(perk.name, text_font, "navy", x + 10, y + 100)
        
        # Create wrapped description text
        description = perk.description
        if perk.max_count < float('inf'):
            description += f" ({perk.count + 1}/{perk.max_count})"
        
        # Split description into words
        words = description.split()
        lines = []
        current_line = ""
        
        # Word wrap to fit within the perk button width
        max_chars_per_line = 11  # Adjust based on font size and perk width
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw description with line breaks
        for line_idx, line in enumerate(lines):
            # Limit to 3 lines maximum to avoid overflow
            if line_idx < 3:
                draw_text(line, text_font, "black", x + 10, y + 130 + line_idx * 20)
    
    return selected_perk
def create_turret(mouse_pos):
    mouse_tile_x = mouse_pos[0] // c.TILE_SIZE
    mouse_tile_y = mouse_pos[1] // c.TILE_SIZE
    #calculate the sequential number of the tile
    mouse_tile_num = (mouse_tile_y * c.COLS) + mouse_tile_x
    #check if that tile is grass
    if world.tile_map[mouse_tile_num] == 7:
        #check that there isn't already a turret there
        space_is_free = True
        for turret in turret_group:
            if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
                space_is_free = False
        #if it is a free space then create turret
        if space_is_free:
            t_type = turret_types[selected_turret_type]
            if t_type == 'basic':
                new_turret = Turret(turret_spritesheets_dict[t_type], mouse_tile_x, mouse_tile_y, shot_fx)
            elif t_type == 'slow':
                from turret import TurretSlow
                new_turret = TurretSlow(turret_spritesheets_dict[t_type], mouse_tile_x, mouse_tile_y, shot_fx)
            elif t_type == 'powerful':
                from turret import TurretPowerful
                new_turret = TurretPowerful(turret_spritesheets_dict[t_type], mouse_tile_x, mouse_tile_y, shot_fx)
            turret_group.add(new_turret)
            #deduct cost of turret
            world.money -= c.BUY_COST

def select_turret(mouse_pos):
  mouse_tile_x = mouse_pos[0] // c.TILE_SIZE
  mouse_tile_y = mouse_pos[1] // c.TILE_SIZE
  for turret in turret_group:
    if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
      return turret

def clear_selection():
  for turret in turret_group:
    turret.selected = False

#create world
world = World(world_data, map_image)
world.process_data()
world.process_enemies()

#create groups
enemy_group = pg.sprite.Group()
turret_group = pg.sprite.Group()

upgrade_button = Button(c.SCREEN_WIDTH + 30, 145, ui_images['upgrade_turret'], True)
cancel_button = Button(c.SCREEN_WIDTH, c.SCREEN_HEIGHT - 60, ui_images['cancel'], True)
begin_button = Button(c.SCREEN_WIDTH + 110, c.SCREEN_HEIGHT - 60, ui_images['begin'], True)
restart_button = Button(310, 300, ui_images['restart'], True)
fast_forward_button = Button(c.SCREEN_WIDTH + 110, c.SCREEN_HEIGHT - 60, ui_images['fast_forward'], False)

# Create turret type buttons - positioned better with dynamic spacing
turret_type_buttons = []
# Calculate spacing based on number of turret types to avoid overlap
spacing_between_buttons = 80
for i, turret_type in enumerate(turret_types):
    # Space turret buttons vertically in the side panel
    button_x = c.SCREEN_WIDTH + 70  # Center the button in the panel
    button_y = 180 + (i * spacing_between_buttons)  # Vertical spacing between turret buttons
    turret_type_buttons.append((
        Button(button_x, button_y, cursor_turrets[turret_type], True),
        turret_type
    ))

#game loop
run = True
while run:

  clock.tick(c.FPS)

  #########################
  # UPDATING SECTION
  #########################

  if game_over == False:
    #check if player has lost
    if world.health <= 0:
      game_over = True
      game_outcome = -1 #loss
    # No win condition based on level count - infinite rounds

    #update groups
    enemy_group.update(world)
    turret_group.update(enemy_group, world)

    #highlight selected turret
    if selected_turret:
      selected_turret.selected = True

  #########################
  # DRAWING SECTION
  #########################

  #draw level
  world.draw(screen)

  #draw groups
  enemy_group.draw(screen)
  for turret in turret_group:
    turret.draw(screen)

  display_data()

  if game_over == False:
    # If perk selection is active, show it and handle selection
    if perk_selection_active:
      # Just render the perk selection UI, actual selection is handled in event handling
      render_perk_selection()
      # Don't allow other game actions while perk selection is active
    # First draw the cancel button if in placing turrets mode
    elif placing_turrets == True:
        if cancel_button.draw(screen):
            placing_turrets = False
            
    #check if the level has been started or not
    if level_started == False:
      if begin_button.draw(screen):
        level_started = True
    else:
      #fast forward option
      world.game_speed = 1
      if fast_forward_button.draw(screen):
        world.game_speed = 2
      #spawn enemies
      if pg.time.get_ticks() - last_enemy_spawn > c.SPAWN_COOLDOWN:
        if world.spawned_enemies < len(world.enemy_list):
          enemy_type = world.enemy_list[world.spawned_enemies]
          enemy = Enemy(enemy_type, world.waypoints, enemy_images)
          enemy_group.add(enemy)
          world.spawned_enemies += 1
          last_enemy_spawn = pg.time.get_ticks()

    #check if the wave is finished
    if world.check_level_complete() == True:
      world.money += c.LEVEL_COMPLETE_REWARD
      world.level += 1
      level_started = False
      last_enemy_spawn = pg.time.get_ticks()
      world.reset_level()
      world.process_enemies()
      
      # Check if this level should trigger perk selection
      if world.level >= perks_enabled_at_level and world.level % perk_frequency == 0:
        perk_options = get_random_perks(perks_dict, 3)
        if perk_options:  # Make sure we have perks to show
          perk_selection_active = True
    #draw buttons
    
    # Draw turret type buttons with labels and costs
    for i, (btn, t_type) in enumerate(turret_type_buttons):
        screen.blit(ui_images['coin'], (c.SCREEN_WIDTH + 10, 185 + (i * 80)))
        draw_text(str(c.BUY_COST), text_font, "grey100", c.SCREEN_WIDTH + 45, 185 + (i * 80))
        if btn.draw(screen):
            selected_turret_type = turret_types.index(t_type)
            placing_turrets = True
        draw_text(t_type.capitalize(), text_font, "grey100", c.SCREEN_WIDTH + 160, 185 + (i * 80))
    
    # Show the currently selected type below the perk information
    draw_text(f"Selected: {turret_types[selected_turret_type].capitalize()}", 
              text_font, "grey100", c.SCREEN_WIDTH + 10, 120)
    
    #if placing turrets then show the cursor
    if placing_turrets == True:
        #show cursor turret based on selected type
        cursor_rect = cursor_turrets[turret_types[selected_turret_type]].get_rect()
        cursor_pos = pg.mouse.get_pos()
        cursor_rect.center = cursor_pos
        if cursor_pos[0] <= c.SCREEN_WIDTH:
            screen.blit(cursor_turrets[turret_types[selected_turret_type]], cursor_rect)
    
    # Show upgrade button if a turret is selected        
    if selected_turret:
      if selected_turret.upgrade_level < c.TURRET_LEVELS:
        screen.blit(ui_images['coin'], (c.SCREEN_WIDTH + 95, 145))
        if upgrade_button.draw(screen):
          if world.money >= c.UPGRADE_COST:
            selected_turret.upgrade()
            world.money -= c.UPGRADE_COST
  else:
    pg.draw.rect(screen, "dodgerblue", (200, 200, 400, 200), border_radius = 30)
    draw_text("GAME OVER", large_font, "grey0", 310, 230)
    draw_text(f"You survived {world.level} rounds!", text_font, "grey0", 310, 270)
    #restart level
    if restart_button.draw(screen):
      game_over = False
      level_started = False
      placing_turrets = False
      selected_turret = None
      selected_turret_type = 0  # Reset to first turret type
      perk_selection_active = False
      last_enemy_spawn = pg.time.get_ticks()
      
      # Reset perks
      perks_dict = initialize_perks(SCRIPT_DIR)
      
      # Reset damage to default
      c.DAMAGE = 5  # Reset to default damage value
      
      # Reset world
      world = World(world_data, map_image)
      world.process_data()
      world.process_enemies()
      
      #empty groups
      enemy_group.empty()
      turret_group.empty()

  #update display
  pg.display.flip()
  
  #event handler
  for event in pg.event.get():
    #quit program
    if event.type == pg.QUIT:
      run = False
    #mouse click
    if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
      mouse_pos = pg.mouse.get_pos()
      
      # Handle perk selection separately
      if perk_selection_active:
        # Check if a perk was clicked on
        panel_width = 600
        panel_height = 300
        panel_x = (c.SCREEN_WIDTH - panel_width) // 2
        panel_y = (c.SCREEN_HEIGHT - panel_height) // 2
        
        option_width = 150
        option_spacing = 40
        starting_x = panel_x + (panel_width - (3 * option_width + 2 * option_spacing)) // 2
        
        # Check each perk option
        for i, perk in enumerate(perk_options):
          x = starting_x + i * (option_width + option_spacing)
          y = panel_y + 80
          perk_rect = pg.Rect(x, y, option_width, 180)
          
          if perk_rect.collidepoint(mouse_pos):
            # Apply the selected perk effect
            perk.apply_effect(world, turret_group)
            # Close perk selection
            perk_selection_active = False
            break
      #check if mouse is on the game area (only if not in perk selection)
      elif mouse_pos[0] < c.SCREEN_WIDTH and mouse_pos[1] < c.SCREEN_HEIGHT:
        #clear selected turrets
        selected_turret = None
        clear_selection()
        if placing_turrets == True:
          #check if there is enough money for a turret
          if world.money >= c.BUY_COST:
            create_turret(mouse_pos)
            # Exit placing turrets mode after placement
            placing_turrets = False
        else:
          selected_turret = select_turret(mouse_pos)

#quit pygame
pg.quit()