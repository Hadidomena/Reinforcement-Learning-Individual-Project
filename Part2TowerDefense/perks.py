import pygame as pg
import os
import random
import constants as c

class Perk:
    def __init__(self, image, name, description, effect_function, max_count=float('inf')):
        self.image = image
        self.name = name
        self.description = description
        self.effect_function = effect_function
        self.max_count = max_count
        self.count = 0

    def apply_effect(self, world, turret_group):
        if self.count < self.max_count:
            self.effect_function(world, turret_group)
            self.count += 1
            return True
        return False

    def is_available(self):
        return self.count < self.max_count

def initialize_perks(script_dir):
    sword_img = pg.image.load(os.path.join(script_dir, 'assets/images/perks/sword.png')).convert_alpha()
    heart_img = pg.image.load(os.path.join(script_dir, 'assets/images/perks/heart.png')).convert_alpha()
    speed_img = pg.image.load(os.path.join(script_dir, 'assets/images/perks/speed.png')).convert_alpha()
    coin_img = pg.image.load(os.path.join(script_dir, 'assets/images/perks/coin.png')).convert_alpha()

    def increase_damage(world, turret_group):
        c.DAMAGE += 2
    
    def heal_player(world, turret_group):
        world.health = min(c.HEALTH, world.health + 25)
    
    def increase_fire_rate(world, turret_group):
        for turret in turret_group:
            turret.cooldown = max(100, int(turret.cooldown * 0.8))
    
    def give_money(world, turret_group):
        world.money += 150
    
    perks = {
        "damage": Perk(sword_img, "Damage Up", "Increase damage", increase_damage, 2),
        "heal": Perk(heart_img, "Healing", "Restore health", heal_player),
        "speed": Perk(speed_img, "Fast Fire", "Increase firing speed", increase_fire_rate, 2),
        "money": Perk(coin_img, "Fortune", "Gain 150 gold", give_money)
    }
    
    return perks

def get_random_perks(perks, count=3):
    available_perks = [perk for perk in perks.values() if perk.is_available()]
    
    if len(available_perks) <= count:
        return available_perks
    
    return random.sample(available_perks, count)