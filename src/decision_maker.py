#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for making poker decisions based on game state and hand analysis.
"""

import logging
import random
from typing import Dict
from game_state import GameState

logger = logging.getLogger(__name__)


class Decision:
    """Class representing a poker decision."""
    
    ACTION_FOLD = "fold"
    ACTION_CHECK = "check"
    ACTION_CALL = "call"
    ACTION_BET = "bet"
    ACTION_RAISE = "raise"
    
    def __init__(self, action, amount=0.0):
        """
        Initialize a Decision object.
        
        Args:
            action (str): The action to take (fold, check, call, bet, raise)
            amount (float): The amount to bet or raise (if applicable)
        """
        self.action = action
        self.amount = amount
        
    def __str__(self):
        if self.action in [self.ACTION_BET, self.ACTION_RAISE]:
            return f"{self.action} {self.amount}"
        return self.action


class DecisionMaker:
    """Class for making poker decisions based on game state and hand analysis."""
    
    def __init__(self, config=None):
        """Initialize the decision maker with optional configuration."""
        self.config = config or {}
        # Define aggression level (0-1), where 0 is tight-passive and 1 is loose-aggressive
        self.aggression = self.config.get('aggression', 0.5)
        # Bluff frequency (0-1)
        self.bluff_frequency = self.config.get('bluff_frequency', 0.1)
        # Minimum win probability to continue with the hand
        self.min_continue_threshold = self.config.get('min_continue_threshold', 0.3)
        logger.info(f"Decision maker initialized (aggression: {self.aggression:.2f}, "
                   f"bluff frequency: {self.bluff_frequency:.2f})")
        
    def decide(self, game_state: GameState, hand_analysis: Dict) -> Decision:
        """
        Make a decision based on game state and hand analysis.
        
        Args:
            game_state (GameState): Current game state
            hand_analysis (Dict): Analysis of the current hand
            
        Returns:
            Decision: The decision to make
        """
        if not game_state.is_our_turn:
            logger.warning("Asked to make decision but it's not our turn")
            return Decision(Decision.ACTION_CHECK)
            
        if not game_state.available_actions:
            logger.warning("No available actions detected")
            return Decision(Decision.ACTION_CHECK)
            
        try:
            # Extract key information
            win_probability = hand_analysis.get('win_probability', 0)
            hand_strength = hand_analysis.get('strength', 0)
            hand_type = hand_analysis.get('hand_type', 'unknown')
            
            # Log the hand analysis being used for decision making
            logger.info(f"Making decision based on hand: {hand_type} "
                       f"(strength: {hand_strength:.2f}, win prob: {win_probability:.2%})")
            
            # Determine if we should bluff
            should_bluff = self._should_bluff(game_state)
            
            # Make decision based on game stage (pre-flop, flop, turn, river)
            community_card_count = len(game_state.community_cards)
            
            if community_card_count == 0:
                # Pre-flop decision
                return self._decide_preflop(game_state, hand_analysis, should_bluff)
            elif community_card_count == 3:
                # Flop decision
                return self._decide_postflop(game_state, hand_analysis, should_bluff, "flop")
            elif community_card_count == 4:
                # Turn decision
                return self._decide_postflop(game_state, hand_analysis, should_bluff, "turn")
            elif community_card_count == 5:
                # River decision
                return self._decide_postflop(game_state, hand_analysis, should_bluff, "river")
            else:
                logger.warning(f"Unexpected number of community cards: {community_card_count}")
                return self._default_decision(game_state)
                
        except Exception as e:
            logger.exception(f"Error making decision: {e}")
            return self._default_decision(game_state)
    
    def _should_bluff(self, game_state: GameState) -> bool:
        """
        Determine if we should bluff based on various factors.
        
        Args:
            game_state (GameState): Current game state
            
        Returns:
            bool: True if we should bluff, False otherwise
        """
        # Base bluff chance on our configured frequency
        bluff_chance = self.bluff_frequency
        
        # Adjust based on position (more likely to bluff in late position)
        if game_state.position == "button" or game_state.position == "late":
            bluff_chance *= 1.5
        elif game_state.position == "early":
            bluff_chance *= 0.5
            
        # Adjust based on pot size (less likely to bluff for large pots)
        pot_to_stack_ratio = game_state.pot_size / max(1, game_state.player_stack)
        if pot_to_stack_ratio > 0.25:
            bluff_chance *= 0.7
            
        # Randomly determine if we should bluff
        return random.random() < bluff_chance
    
    def _decide_preflop(self, game_state: GameState, hand_analysis: Dict, should_bluff: bool) -> Decision:
        """
        Make a pre-flop decision.
        
        Args:
            game_state (GameState): Current game state
            hand_analysis (Dict): Analysis of the current hand
            should_bluff (bool): Whether we should bluff
            
        Returns:
            Decision: The preflop decision
        """
        hand_strength = hand_analysis.get('strength', 0)
        
        # Adjust hand strength by position
        position_adjusted_strength = hand_strength
        if game_state.position == "late" or game_state.position == "button":
            position_adjusted_strength *= 1.2
        elif game_state.position == "early":
            position_adjusted_strength *= 0.8
            
        # Decision thresholds
        fold_threshold = 0.2 - (self.aggression * 0.15)
        call_threshold = 0.4 - (self.aggression * 0.1)
        raise_threshold = 0.6 - (self.aggression * 0.1)
        
        # If we're bluffing, pretend we have a stronger hand
        if should_bluff:
            position_adjusted_strength += 0.3
            logger.info("Bluffing pre-flop")
            
        # Basic preflop strategy
        if position_adjusted_strength < fold_threshold:
            return Decision(Decision.ACTION_FOLD)
        elif position_adjusted_strength < call_threshold:
            # Check if possible, otherwise fold
            if Decision.ACTION_CHECK in game_state.available_actions:
                return Decision(Decision.ACTION_CHECK)
            else:
                return Decision(Decision.ACTION_FOLD)
        elif position_adjusted_strength < raise_threshold:
            # Call if possible
            if Decision.ACTION_CALL in game_state.available_actions:
                return Decision(Decision.ACTION_CALL)
            elif Decision.ACTION_CHECK in game_state.available_actions:
                return Decision(Decision.ACTION_CHECK)
            else:
                return Decision(Decision.ACTION_FOLD)
        else:
            # Strong hand - raise if possible
            if Decision.ACTION_RAISE in game_state.available_actions:
                bet_amount = self._calculate_bet_amount(game_state, position_adjusted_strength)
                return Decision(Decision.ACTION_RAISE, bet_amount)
            elif Decision.ACTION_BET in game_state.available_actions:
                bet_amount = self._calculate_bet_amount(game_state, position_adjusted_strength)
                return Decision(Decision.ACTION_BET, bet_amount)
            elif Decision.ACTION_CALL in game_state.available_actions:
                return Decision(Decision.ACTION_CALL)
            else:
                return Decision(Decision.ACTION_CHECK)
    
    def _decide_postflop(self, game_state: GameState, hand_analysis: Dict, should_bluff: bool, stage: str) -> Decision:
        """
        Make a post-flop decision.
        
        Args:
            game_state (GameState): Current game state
            hand_analysis (Dict): Analysis of the current hand
            should_bluff (bool): Whether we should bluff
            stage (str): Game stage ('flop', 'turn', 'river')
            
        Returns:
            Decision: The postflop decision
        """
        win_probability = hand_analysis.get('win_probability', 0)
        
        # If we're bluffing, pretend we have a better chance of winning
        if should_bluff:
            bluff_factor = 0.3 if stage == 'river' else 0.2
            win_probability += bluff_factor
            logger.info(f"Bluffing on {stage}")
            
        # Adjust thresholds based on game stage
        # On later streets, we need stronger hands to continue
        stage_factor = 0.0
        if stage == 'turn':
            stage_factor = 0.05
        elif stage == 'river':
            stage_factor = 0.1
            
        # Decision thresholds
        fold_threshold = 0.3 + stage_factor - (self.aggression * 0.1)
        call_threshold = 0.45 + stage_factor - (self.aggression * 0.1)
        raise_threshold = 0.6 + stage_factor - (self.aggression * 0.1)
        
        # Post-flop strategy
        if win_probability < fold_threshold:
            # Weak hand - check if possible, otherwise fold
            if Decision.ACTION_CHECK in game_state.available_actions:
                return Decision(Decision.ACTION_CHECK)
            else:
                return Decision(Decision.ACTION_FOLD)
        elif win_probability < call_threshold:
            # Marginal hand - call small bets, fold to large ones
            if game_state.current_bet > game_state.player_stack * 0.25:
                return Decision(Decision.ACTION_FOLD)
            elif Decision.ACTION_CALL in game_state.available_actions:
                return Decision(Decision.ACTION_CALL)
            else:
                return Decision(Decision.ACTION_CHECK)
        elif win_probability < raise_threshold:
            # Good hand - call or make a moderate bet
            if Decision.ACTION_BET in game_state.available_actions:
                bet_amount = self._calculate_bet_amount(game_state, win_probability * 0.7)
                return Decision(Decision.ACTION_BET, bet_amount)
            elif Decision.ACTION_CALL in game_state.available_actions:
                return Decision(Decision.ACTION_CALL)
            else:
                return Decision(Decision.ACTION_CHECK)
        else:
            # Strong hand - bet/raise aggressively
            if Decision.ACTION_RAISE in game_state.available_actions:
                bet_amount = self._calculate_bet_amount(game_state, win_probability)
                return Decision(Decision.ACTION_RAISE, bet_amount)
            elif Decision.ACTION_BET in game_state.available_actions:
                bet_amount = self._calculate_bet_amount(game_state, win_probability)
                return Decision(Decision.ACTION_BET, bet_amount)
            elif Decision.ACTION_CALL in game_state.available_actions:
                return Decision(Decision.ACTION_CALL)
            else:
                return Decision(Decision.ACTION_CHECK)
    
    def _calculate_bet_amount(self, game_state: GameState, strength_factor: float) -> float:
        """
        Calculate bet amount based on pot size, stack size, and hand strength.
        
        Args:
            game_state (GameState): Current game state
            strength_factor (float): Hand strength factor (0-1)
            
        Returns:
            float: Amount to bet
        """
        pot_size = max(1, game_state.pot_size)
        stack_size = max(1, game_state.player_stack)
        
        # Base bet size as a percentage of the pot
        base_pot_percentage = 0.5 + (strength_factor * 0.5)  # 50% to 100% of pot
        
        # Make sure bet doesn't exceed our stack
        max_bet = min(pot_size * base_pot_percentage, stack_size * 0.8)
        
        # Add some randomization (+/- 15%)
        randomization = random.uniform(-0.15, 0.15)
        bet_amount = max_bet * (1 + randomization)
        
        # Round to common bet sizing
        min_bet = pot_size * 0.25  # Minimum 1/4 pot bet
        bet_amount = max(min_bet, bet_amount)
        
        # Ensure bet is not more than our stack
        bet_amount = min(bet_amount, stack_size)
        
        return round(bet_amount, 2)
    
    def _default_decision(self, game_state: GameState) -> Decision:
        """
        Make a safe default decision when uncertain.
        
        Args:
            game_state (GameState): Current game state
            
        Returns:
            Decision: A safe default decision
        """
        # Consider position even when we don't know our cards
        position_factor = 0.0
        if game_state.position == "late" or game_state.position == "button":
            position_factor = 0.3  # More aggressive in late position
        elif game_state.position == "middle":
            position_factor = 0.15
            
        # Consider pot size
        pot_factor = min(0.2, game_state.pot_size / 10.0)
        
        # Random factor to prevent being too predictable
        random_factor = random.uniform(0, 0.2)
        
        # Aggregate decision factor
        decision_factor = position_factor + pot_factor + random_factor
        
        # Make a more strategic default decision based on position and randomness
        if decision_factor > 0.5 and Decision.ACTION_BET in game_state.available_actions:
            # Small bet when we're in good position and feeling aggressive
            bet_amount = game_state.pot_size * 0.5  # Half pot bet
            bet_amount = min(bet_amount, game_state.player_stack * 0.1)  # Max 10% of stack
            logger.info(f"Making default bet from good position: {bet_amount:.2f}")
            return Decision(Decision.ACTION_BET, round(bet_amount, 2))
            
        if decision_factor > 0.3 and Decision.ACTION_CALL in game_state.available_actions:
            # Call small bets in good position
            if game_state.current_bet <= game_state.player_stack * 0.05:  # Only call if bet is small
                logger.info("Making default call from decent position")
                return Decision(Decision.ACTION_CALL)
                
        if Decision.ACTION_CHECK in game_state.available_actions:
            return Decision(Decision.ACTION_CHECK)
        elif Decision.ACTION_FOLD in game_state.available_actions:
            return Decision(Decision.ACTION_FOLD)
        else:
            # Fallback to check
            return Decision(Decision.ACTION_CHECK)