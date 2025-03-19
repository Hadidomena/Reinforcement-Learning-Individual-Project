import pygame
from snake import SnakeGame

def main():
    pygame.init()
    game = SnakeGame()
    clock = pygame.time.Clock()
    running = True
    
    # game loop
    while running:

        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
        clock.tick(15)
        
    print('Final Score', score)
        
        
    pygame.quit()

if __name__ == "__main__":
    main()