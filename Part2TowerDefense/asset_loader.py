import pygame as pg
import os
import constants as c

def load_turret_assets(script_dir):
    """Load turret spritesheets and cursor images"""
    turrets_dir = os.path.join(script_dir, 'assets/images/turrets')
    turret_type_dirs = [d for d in os.listdir(turrets_dir) if os.path.isdir(os.path.join(turrets_dir, d))]
    turret_types = []
    turret_spritesheets_dict = {}
    cursor_turrets = {}
    
    for turret_dir in turret_type_dirs:
        turret_type = turret_dir.replace('turret', '').lower() if turret_dir.lower().startswith('turret') else turret_dir.lower()
        if not turret_type:
            turret_type = 'basic'
        turret_types.append(turret_type)
        
        spritesheets = []
        for x in range(1, c.TURRET_LEVELS + 1):
            sheet_path = os.path.join(turrets_dir, turret_dir, f'turret_{x}.png')
            spritesheets.append(pg.image.load(sheet_path).convert_alpha())
        turret_spritesheets_dict[turret_type] = spritesheets
        
        cursor_path = os.path.join(turrets_dir, turret_dir, 'cursor_turret.png')
        cursor_turrets[turret_type] = pg.image.load(cursor_path).convert_alpha()
    
    return turret_types, turret_spritesheets_dict, cursor_turrets

def load_enemy_images(script_dir):
    """Load enemy images"""
    return {
        "weak": pg.image.load(os.path.join(script_dir, 'assets/images/enemies/enemy_1.png')).convert_alpha(),
        "medium": pg.image.load(os.path.join(script_dir, 'assets/images/enemies/enemy_2.png')).convert_alpha(),
        "strong": pg.image.load(os.path.join(script_dir, 'assets/images/enemies/enemy_3.png')).convert_alpha(),
        "elite": pg.image.load(os.path.join(script_dir, 'assets/images/enemies/enemy_4.png')).convert_alpha()
    }

def load_ui_images(script_dir):
    """Load UI images"""
    return {
        'buy_turret': pg.image.load(os.path.join(script_dir, 'assets/images/buttons/buy_turret.png')).convert_alpha(),
        'cancel': pg.image.load(os.path.join(script_dir, 'assets/images/buttons/cancel.png')).convert_alpha(),
        'upgrade_turret': pg.image.load(os.path.join(script_dir, 'assets/images/buttons/upgrade_turret.png')).convert_alpha(),
        'begin': pg.image.load(os.path.join(script_dir, 'assets/images/buttons/begin.png')).convert_alpha(),
        'restart': pg.image.load(os.path.join(script_dir, 'assets/images/buttons/restart.png')).convert_alpha(),
        'fast_forward': pg.image.load(os.path.join(script_dir, 'assets/images/buttons/fast_forward.png')).convert_alpha(),
        'heart': pg.image.load(os.path.join(script_dir, "assets/images/gui/heart.png")).convert_alpha(),
        'coin': pg.image.load(os.path.join(script_dir, "assets/images/gui/coin.png")).convert_alpha(),
        'logo': pg.image.load(os.path.join(script_dir, "assets/images/gui/logo.png")).convert_alpha()
    }
