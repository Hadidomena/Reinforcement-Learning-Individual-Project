ENEMY_SPAWN_DATA = [
  {
    #1
    "weak": 15,
    "medium": 0,
    "strong": 0,
    "elite": 0
  },
  {
    #2
    "weak": 30,
    "medium": 0,
    "strong": 0,
    "elite": 0
  },
  {
    #3
    "weak": 20,
    "medium": 5,
    "strong": 0,
    "elite": 0
  },
  {
    #4
    "weak": 30,
    "medium": 15,
    "strong": 0,
    "elite": 0
  },
  {
    #5
    "weak": 5,
    "medium": 20,
    "strong": 0,
    "elite": 0
  },
  {
    #6
    "weak": 15,
    "medium": 15,
    "strong": 4,
    "elite": 0
  },
  {
    #7
    "weak": 20,
    "medium": 25,
    "strong": 5,
    "elite": 0
  },
  {
    #8
    "weak": 10,
    "medium": 20,
    "strong": 15,
    "elite": 0
  },
  {
    #9
    "weak": 15,
    "medium": 10,
    "strong": 5,
    "elite": 0
  },
  {
    #10
    "weak": 0,
    "medium": 100,
    "strong": 0,
    "elite": 0
  },
  {
    #11
    "weak": 5,
    "medium": 10,
    "strong": 12,
    "elite": 2
  },
  {
    #12
    "weak": 0,
    "medium": 15,
    "strong": 10,
    "elite": 5
  },
  {
    #13
    "weak": 20,
    "medium": 0,
    "strong": 25,
    "elite": 10
  },
  {
    #14
    "weak": 15,
    "medium": 15,
    "strong": 15,
    "elite": 15
  },
  {
    #15
    "weak": 25,
    "medium": 25,
    "strong": 25,
    "elite": 25
  }
]

ENEMY_DATA = {
    "weak": {
    "health": 10,
    "speed": 2
  },
    "medium": {
    "health": 15,
    "speed": 3
  },
    "strong": {
    "health": 20,
    "speed": 4
  },
    "elite": {
    "health": 30,
    "speed": 6
  }
}

def get_dynamic_enemy_count(level):
    """
    Dynamically generate enemy counts based on level number.
    For levels beyond the predefined ones, this will create progressively harder waves.
    """
    # Use predefined data for first 15 levels
    if level <= 15:
        return ENEMY_SPAWN_DATA[level - 1]
    
    # For levels beyond 15, generate increasing counts
    base_count = level - 15  # Levels beyond 15 get progressively harder
    
    # Calculate enemy count by type with a formula that scales with level
    weak_count = max(0, 20 + (base_count * 2) - (base_count // 3 * 10))  # Weak enemies increase then taper off
    medium_count = max(0, 15 + (base_count * 3))  # Medium enemies increase steadily
    strong_count = max(0, 10 + (base_count * 4))  # Strong enemies increase faster
    elite_count = max(0, 5 + (base_count * 5))  # Elite enemies increase fastest
    
    # Ensure there's a minimum number of enemies in total
    total = weak_count + medium_count + strong_count + elite_count
    if total < level * 5:  # Minimum of 5 enemies per level number
        # Add the difference to the strongest enemy type available
        if level > 40:  # Very high levels focus on elites
            elite_count += (level * 5) - total
        elif level > 30:  # High levels focus on strong enemies
            strong_count += (level * 5) - total
        elif level > 20:  # Medium levels focus on medium enemies
            medium_count += (level * 5) - total
        else:  # Lower levels add more weak enemies
            weak_count += (level * 5) - total
    
    return {
        "weak": weak_count,
        "medium": medium_count,
        "strong": strong_count,
        "elite": elite_count
    }