"""
Enhanced reward shaping system for better learning progression
"""
import math
import numpy as np

class AdvancedRewardShaper:
    def __init__(self):
        self.prev_state = None
        self.prev_score = 0
        self.prev_level = 1
        self.prev_money = 650
        self.prev_health = 100
        self.prev_enemies_killed = 0
        
        # Dynamic reward scaling factors that adapt to performance
        self.score_weight = 10.0
        self.level_weight = 50.0  # Higher reward for reaching new levels
        self.efficiency_weight = 5.0
        self.survival_weight = 2.0
        self.strategic_weight = 3.0
        
        # Penalty factors
        self.money_waste_penalty = -0.5
        self.health_loss_penalty = -2.0
        self.inefficient_placement_penalty = -1.0
        
        # Performance tracking
        self.total_turrets_placed = 0
        self.total_upgrades = 0
        self.wave_clear_times = []
        
        # Advanced plateau handling
        self.plateau_boost = 1.0  # Multiplier for rewards during plateau
        self.high_level_threshold = 15  # When to apply advanced rewards
        self.streak_bonus = 0  # Bonus for consecutive good performance
        
    def calculate_reward(self, current_state, action_type, game_stats):
        total_reward = 0.0
        
        current_score = game_stats.get('score', 0)
        current_level = game_stats.get('level', 1)
        current_money = game_stats.get('money', 650)
        current_health = game_stats.get('health', 100)
        current_enemies_killed = game_stats.get('enemies_killed', 0)
        is_game_over = game_stats.get('game_over', False)
        score_gain = current_score - self.prev_score
        if score_gain > 0:
            score_reward = self.score_weight * (1 + math.log(1 + score_gain))
            total_reward += score_reward
            
        level_gain = current_level - self.prev_level
        if level_gain > 0:
            if current_level >= self.high_level_threshold:
                base_level_reward = self.level_weight * level_gain
                exponential_bonus = (current_level - self.high_level_threshold + 1) ** 1.5 * 20
                level_reward = base_level_reward + exponential_bonus
                
                self.streak_bonus += 10
                level_reward += self.streak_bonus
                print(f"🚀 HIGH LEVEL BONUS: {level_reward:.1f} (base: {base_level_reward:.1f}, exp: {exponential_bonus:.1f}, streak: {self.streak_bonus})")
            else:
                level_reward = self.level_weight * level_gain * (1 + current_level * 0.1)
                
            level_reward *= self.plateau_boost
            total_reward += level_reward
            print(f"🎉 Level up reward: {level_reward:.1f}")
        else:
            self.streak_bonus = max(0, self.streak_bonus * 0.95)
        
        if action_type == "place" and current_money < self.prev_money:
            # Reward efficient turret placement based on subsequent enemy kills
            enemies_killed_gain = current_enemies_killed - self.prev_enemies_killed
            if enemies_killed_gain > 0:
                efficiency_reward = self.efficiency_weight * enemies_killed_gain * 0.5
                total_reward += efficiency_reward
            self.total_turrets_placed += 1
            
        elif action_type == "upgrade" and current_money < self.prev_money:
            enemies_killed_gain = current_enemies_killed - self.prev_enemies_killed
            if enemies_killed_gain > 0:
                upgrade_reward = self.efficiency_weight * enemies_killed_gain * 0.9
                total_reward += upgrade_reward
            self.total_upgrades += 1
        
        health_change = current_health - self.prev_health
        if health_change < 0:
            health_penalty = self.health_loss_penalty * abs(health_change)
            if score_gain > 0:
                health_penalty *= 0.5
            total_reward += health_penalty
        elif current_health == 100 and current_level > self.prev_level:
            total_reward += self.survival_weight * 5
        
        if current_money > 100 and score_gain > 0:
            strategic_reward = self.strategic_weight * 0.5
            total_reward += strategic_reward
        
        money_spent = self.prev_money - current_money
        if money_spent > 0 and score_gain == 0 and current_enemies_killed == self.prev_enemies_killed:
            waste_penalty = self.money_waste_penalty * money_spent * 0.1
            total_reward += waste_penalty
        
        if is_game_over:
            if current_score >= 10:
                survival_bonus = math.sqrt(current_score) * 10
                total_reward += survival_bonus
            else:
                early_game_penalty = -20
                total_reward += early_game_penalty

        if current_level >= self.high_level_threshold:
            difficulty_scale = 1.0 + (current_level - self.high_level_threshold) * 0.01
            difficulty_scale *= self.plateau_boost
        else:
            difficulty_scale = 1.0 + (current_level - 1) * 0.02
        total_reward *= difficulty_scale
        
        if hasattr(self, 'action_history'):
            if len(self.action_history) > 10:
                recent_actions = self.action_history[-10:]
                action_diversity = len(set(recent_actions)) / len(recent_actions)
                if action_diversity > 0.6:  # Good variety in actions
                    exploration_bonus = 2.0
                    total_reward += exploration_bonus

        self.prev_score = current_score
        self.prev_level = current_level
        self.prev_money = current_money
        self.prev_health = current_health
        self.prev_enemies_killed = current_enemies_killed
        
        max_reward = 300 if current_level >= self.high_level_threshold else 200
        total_reward = np.clip(total_reward, -100, max_reward)
        
        return total_reward
    
    def get_curiosity_reward(self, state_old, action, state_new):
        if not hasattr(self, 'state_action_counts'):
            self.state_action_counts = {}
        
        state_hash = hash(tuple(state_old.flatten()[:30]))
        action_idx = np.argmax(action) if isinstance(action, np.ndarray) else action
        
        state_action_pair = (state_hash, action_idx)
        
        if state_action_pair not in self.state_action_counts:
            self.state_action_counts[state_action_pair] = 0
        
        self.state_action_counts[state_action_pair] += 1
        
        base_curiosity = 1.0 / math.sqrt(self.state_action_counts[state_action_pair])
        curiosity_reward = base_curiosity * 0.7 * self.plateau_boost
        
        return curiosity_reward
    
    def reset(self):
        self.prev_score = 0
        self.prev_level = 1
        self.prev_money = 650
        self.prev_health = 100
        self.prev_enemies_killed = 0
    
    def get_curriculum_bonus(self, agent_level, game_performance):
        base_bonus = 0
        
        if game_performance > agent_level * 3:
            base_bonus = min(15, game_performance - agent_level * 3)

        if game_performance >= self.high_level_threshold:
            breakthrough_bonus = (game_performance - self.high_level_threshold + 1) * 2
            base_bonus += breakthrough_bonus
            
        # Apply plateau boost
        return base_bonus * self.plateau_boost
    
    def reset_adaptive_parameters(self):
        """Reset adaptive parameters for breakthrough learning"""
        # Reset streak bonus and plateau boost
        self.streak_bonus = 0
        self.plateau_boost = 1.0
        
        # Reset state-action counts for fresh curiosity
        if hasattr(self, 'state_action_counts'):
            self.state_action_counts.clear()
        
        # Reset performance tracking
        self.total_turrets_placed = 0
        self.total_upgrades = 0
        self.wave_clear_times.clear()
        
        print("🔄 Reset reward shaping adaptive parameters for breakthrough")
    
    def set_breakthrough_mode(self, enabled=True):
        if enabled:
            self.plateau_boost = 2.0  # Double rewards during breakthrough
            self.high_level_threshold = max(8, self.high_level_threshold - 3)  # Lower threshold
            print(f"🚀 Breakthrough mode enabled! Boost: {self.plateau_boost}x, Threshold: {self.high_level_threshold}")
        else:
            self.plateau_boost = 1.0
            self.high_level_threshold = 15  # Reset to normal
            print("📈 Breakthrough mode disabled")
