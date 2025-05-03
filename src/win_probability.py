#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for estimating poker hand win probability.
"""

import logging
from typing import List
from hand_evaluator import calculate_hand_strength

logger = logging.getLogger(__name__)


def estimate_win_probability(player_cards: List[str], community_cards: List[str], position: str) -> float:
    """
    Estimate the probability of winning with the current hand.
    
    Args:
        player_cards (List): List of player's hole cards
        community_cards (List): List of community cards
        position (str): Player's position
        
    Returns:
        float: Estimated win probability (0-1)
    """
    # This is a simplified implementation
    # In a real application, you would use poker odds calculators
    # or Monte Carlo simulations to get more accurate probabilities
    
    # If we have a hand strength, use it as a starting point
    if player_cards:
        hand_type, hand_strength = calculate_hand_strength(player_cards, community_cards)
        
        # Adjust based on position
        position_factor = 0.0
        if position == "late":
            position_factor = 0.1
        elif position == "middle":
            position_factor = 0.05
        elif position == "early":
            position_factor = 0.0
        elif position == "button":
            position_factor = 0.15
            
        # Number of opponents would also be a factor here
        
        # Adjust for number of community cards (more uncertainty with fewer cards)
        uncertainty_factor = max(0, (5 - len(community_cards)) * 0.05)
        
        # For straight flush, be more conservative with win probability estimate
        # A straight flush is extremely rare and often overestimated
        if hand_type == "straight flush":
            # More realistic win probability, especially with fewer community cards
            base_prob = 0.85  # High but not guaranteed
            card_count_factor = min(1.0, len(community_cards) / 5)  # Scale by number of community cards
            win_prob = base_prob * card_count_factor
        else:
            # Combine factors for other hand types
            win_prob = max(0, min(1, hand_strength - uncertainty_factor + position_factor))
            
        return win_prob
        
    return 0.0