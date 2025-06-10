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
        """
        Calculate sophisticated reward based on multiple factors
        """
        total_reward = 0.0
        
        # Extract current game state
        current_score = game_stats.get('score', 0)
        current_level = game_stats.get('level', 1)
        current_money = game_stats.get('money', 650)
        current_health = game_stats.get('health', 100)
        current_enemies_killed = game_stats.get('enemies_killed', 0)
        is_game_over = game_stats.get('game_over', False)
          # 1. Score progression reward (most important)
        score_gain = current_score - self.prev_score
        if score_gain > 0:
            # Exponential reward for higher scores to encourage consistent improvement
            score_reward = self.score_weight * (1 + math.log(1 + score_gain))
            total_reward += score_reward
            
        # 2. Level progression reward with exponential scaling for high levels
        level_gain = current_level - self.prev_level
        if level_gain > 0:
            # Exponential reward scaling for higher levels to encourage breakthrough
            if current_level >= self.high_level_threshold:
                # Much higher rewards for levels 15+
                base_level_reward = self.level_weight * level_gain
                exponential_bonus = (current_level - self.high_level_threshold + 1) ** 1.5 * 20
                level_reward = base_level_reward + exponential_bonus
                
                # Streak bonus for consecutive level completion
                self.streak_bonus += 10
                level_reward += self.streak_bonus
                print(f"🚀 HIGH LEVEL BONUS: {level_reward:.1f} (base: {base_level_reward:.1f}, exp: {exponential_bonus:.1f}, streak: {self.streak_bonus})")
            else:
                level_reward = self.level_weight * level_gain * (1 + current_level * 0.1)
                
            # Apply plateau boost if active
            level_reward *= self.plateau_boost
            total_reward += level_reward
            print(f"🎉 Level up reward: {level_reward:.1f}")
        else:
            # Decay streak bonus if no level progress
            self.streak_bonus = max(0, self.streak_bonus * 0.95)
        
        # 3. Efficiency rewards
        if action_type == "place" and current_money < self.prev_money:
            # Reward efficient turret placement based on subsequent enemy kills
            enemies_killed_gain = current_enemies_killed - self.prev_enemies_killed
            if enemies_killed_gain > 0:
                efficiency_reward = self.efficiency_weight * enemies_killed_gain * 0.5
                total_reward += efficiency_reward
            self.total_turrets_placed += 1
            
        elif action_type == "upgrade" and current_money < self.prev_money:
            # Reward upgrades that lead to better performance
            enemies_killed_gain = current_enemies_killed - self.prev_enemies_killed
            if enemies_killed_gain > 0:
                upgrade_reward = self.efficiency_weight * enemies_killed_gain * 0.7
                total_reward += upgrade_reward
            self.total_upgrades += 1
        
        # 4. Survival and health management
        health_change = current_health - self.prev_health
        if health_change < 0:
            # Penalty for losing health, but reduced if still progressing
            health_penalty = self.health_loss_penalty * abs(health_change)
            if score_gain > 0:  # Mitigate penalty if still scoring
                health_penalty *= 0.5
            total_reward += health_penalty
        elif current_health == 100 and current_level > self.prev_level:
            # Bonus for completing levels without taking damage
            total_reward += self.survival_weight * 5
        
        # 5. Strategic planning rewards
        # Reward maintaining good money reserves while still being aggressive
        if current_money > 100 and score_gain > 0:
            strategic_reward = self.strategic_weight * 0.5
            total_reward += strategic_reward
        
        # 6. Penalties for poor decisions
        # Money waste penalty (spending without progress)
        money_spent = self.prev_money - current_money
        if money_spent > 0 and score_gain == 0 and current_enemies_killed == self.prev_enemies_killed:
            waste_penalty = self.money_waste_penalty * money_spent * 0.1
            total_reward += waste_penalty
        
        # 7. Game over handling
        if is_game_over:
            if current_score >= 10:
                # Bonus for decent performance
                survival_bonus = math.sqrt(current_score) * 10
                total_reward += survival_bonus
            else:
                # Penalty for early game over
                early_game_penalty = -20
                total_reward += early_game_penalty
          # 8. Progressive difficulty scaling with plateau compensation
        if current_level >= self.high_level_threshold:
            # Less aggressive scaling for high levels + plateau boost
            difficulty_scale = 1.0 + (current_level - self.high_level_threshold) * 0.01
            difficulty_scale *= self.plateau_boost
        else:
            # Standard scaling for lower levels
            difficulty_scale = 1.0 + (current_level - 1) * 0.02
        total_reward *= difficulty_scale
        
        # 9. Exploration bonus for trying new strategies
        if hasattr(self, 'action_history'):
            if len(self.action_history) > 10:
                recent_actions = self.action_history[-10:]
                action_diversity = len(set(recent_actions)) / len(recent_actions)
                if action_diversity > 0.6:  # Good variety in actions
                    exploration_bonus = 2.0
                    total_reward += exploration_bonus
          # Update previous state
        self.prev_score = current_score
        self.prev_level = current_level
        self.prev_money = current_money
        self.prev_health = current_health
        self.prev_enemies_killed = current_enemies_killed
        
        # Clip reward with higher range for advanced levels
        max_reward = 300 if current_level >= self.high_level_threshold else 200
        total_reward = np.clip(total_reward, -100, max_reward)
        
        return total_reward
    
    def get_curiosity_reward(self, state_old, action, state_new):
        """
        Enhanced intrinsic motivation reward with adaptive scaling
        """
        if not hasattr(self, 'state_action_counts'):
            self.state_action_counts = {}
        
        # More sophisticated state hash for better exploration
        state_hash = hash(tuple(state_old.flatten()[:30]))  # Increased from 20 to 30 features
        action_idx = np.argmax(action) if isinstance(action, np.ndarray) else action
        
        state_action_pair = (state_hash, action_idx)
        
        if state_action_pair not in self.state_action_counts:
            self.state_action_counts[state_action_pair] = 0
        
        self.state_action_counts[state_action_pair] += 1
        
        # Enhanced curiosity reward with plateau boost
        base_curiosity = 1.0 / math.sqrt(self.state_action_counts[state_action_pair])
        curiosity_reward = base_curiosity * 0.7 * self.plateau_boost  # Increased from 0.5
        
        return curiosity_reward
    
    def reset(self):
        """Reset for new episode"""
        self.prev_score = 0
        self.prev_level = 1
        self.prev_money = 650
        self.prev_health = 100
        self.prev_enemies_killed = 0
    
    def get_curriculum_bonus(self, agent_level, game_performance):
        """
        Enhanced curriculum learning progression with breakthrough bonuses
        """
        base_bonus = 0
        
        # Standard curriculum bonus
        if game_performance > agent_level * 3:
            base_bonus = min(15, game_performance - agent_level * 3)  # Increased from 10
        
        # Breakthrough bonus for reaching new performance tiers
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
        """Enable/disable breakthrough mode with enhanced rewards"""
        if enabled:
            self.plateau_boost = 2.0  # Double rewards during breakthrough
            self.high_level_threshold = max(8, self.high_level_threshold - 3)  # Lower threshold
            print(f"🚀 Breakthrough mode enabled! Boost: {self.plateau_boost}x, Threshold: {self.high_level_threshold}")
        else:
            self.plateau_boost = 1.0
            self.high_level_threshold = 15  # Reset to normal
            print("📈 Breakthrough mode disabled")
