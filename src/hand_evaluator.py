#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for evaluating poker hand strength.
"""

import logging
from typing import List, Tuple
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards
from card_utils import get_rank_value
from hand_validators import (
    can_form_straight_flush, can_form_four_of_a_kind, can_form_full_house,
    can_form_flush, can_form_straight, can_form_three_of_a_kind,
    can_form_two_pair, can_form_pair, verify_hand_evaluation
)

logger = logging.getLogger(__name__)


def calculate_hand_strength(player_cards: List[str], community_cards: List[str]) -> Tuple[str, float]:
    """
    Calculate the strength and type of the hand.
    
    Args:
        player_cards (List[str]): List of player's hole cards in pypokerengine format
        community_cards (List[str]): List of community cards in pypokerengine format
        
    Returns:
        Tuple[str, float]: Hand type and strength score
    """
    if len(player_cards) < 2:
        return "incomplete", 0.0
        
    if len(community_cards) == 0:
        # Pre-flop evaluation - evaluate hole cards only
        return evaluate_hole_cards(player_cards)
        
    try:
        # Convert string format to pypokerengine Card objects using gen_cards
        hole_cards = gen_cards(player_cards)
        community_cards_obj = gen_cards(community_cards)
        
        # Use HandEvaluator to get the score
        score = HandEvaluator.eval_hand(hole_cards, community_cards_obj)
        
        # Log the raw score for debugging
        logger.debug(f"Raw hand evaluation score: {score}")
        
        # Map the score to a hand type and strength between 0 and 1
        # PyPokerEngine scores range from high card (1) to straight flush (8000+)
        if score >= 8000 and can_form_straight_flush(player_cards, community_cards):  
            # Straight flush - extra validation to ensure it's real
            hand_type = "straight flush"
            hand_strength = 1.0
        elif score >= 7000 and can_form_four_of_a_kind(player_cards, community_cards):  
            # Four of a kind - with extra validation
            hand_type = "four of a kind"
            hand_strength = 0.9
        elif score >= 6000 and can_form_full_house(player_cards, community_cards):  
            # Full house - with extra validation
            hand_type = "full house" 
            hand_strength = 0.8
        elif score >= 5000 and can_form_flush(player_cards, community_cards):  
            # Flush - with extra validation
            hand_type = "flush"
            hand_strength = 0.7
        elif score >= 4000 and can_form_straight(player_cards, community_cards):  
            # Straight - with extra validation
            hand_type = "straight"
            hand_strength = 0.6
        elif score >= 3000 and can_form_three_of_a_kind(player_cards, community_cards):  
            # Three of a kind - with extra validation
            hand_type = "three of a kind"
            hand_strength = 0.5
        elif score >= 2000 and can_form_two_pair(player_cards, community_cards):  
            # Two pair - with extra validation
            hand_type = "two pair"
            hand_strength = 0.4
        elif score >= 1000 and can_form_pair(player_cards, community_cards):  
            # Pair - with extra validation
            hand_type = "pair"
            hand_strength = 0.3
        else:  # High card
            hand_type = "high card"
            hand_strength = 0.1 + (score / 1000.0) * 0.2  # Scale from 0.1 to 0.3
        
        # Double-check hand evaluation for common mistakes
        verify_hand_evaluation(player_cards, community_cards, hand_type)
        
        # Log the identified hand for debugging
        logger.debug(f"Identified hand: {hand_type} with strength {hand_strength}")
        
        return hand_type, hand_strength
    except Exception as e:
        logger.exception(f"Error calculating hand strength: {e}")
        return "error", 0.0


def evaluate_hole_cards(player_cards: List[str]) -> Tuple[str, float]:
    """
    Evaluate hole cards (pre-flop).
    
    Args:
        player_cards (List[str]): List of player's hole cards
        
    Returns:
        Tuple[str, float]: Hand type and strength score
    """
    if len(player_cards) != 2:
        return "incomplete", 0.0
        
    # Extract ranks and suits
    ranks = [card[1] for card in player_cards]
    suits = [card[0] for card in player_cards]
    
    # Check if we have a pair
    if ranks[0] == ranks[1]:
        # Pair strength based on rank
        rank_value = get_rank_value(ranks[0])
        # Scale from 0.5 (pair of 2s) to 0.75 (pair of Aces)
        strength = 0.5 + ((rank_value - 2) / (14 - 2)) * 0.25
        return "pair", strength
        
    # Check if suited
    suited = suits[0] == suits[1]
    
    # Get rank values
    rank_values = [get_rank_value(rank) for rank in ranks]
    rank_values.sort(reverse=True)
    
    # Calculate gap between ranks
    gap = abs(rank_values[0] - rank_values[1]) - 1
    
    # Basic strength calculation for non-pairs
    # Higher cards and connected cards get more value
    # Scale from 0 (lowest) to 0.49 (highest non-paired hand)
    high_card_factor = (rank_values[0] - 2) / (14 - 2) * 0.2
    low_card_factor = (rank_values[1] - 2) / (14 - 2) * 0.1
    gap_factor = max(0, (5 - gap)) / 5 * 0.1
    suited_factor = 0.09 if suited else 0
    
    strength = high_card_factor + low_card_factor + gap_factor + suited_factor
    
    # Determine hand type
    if suited and gap == 0:
        return "suited connectors", strength
    elif suited:
        return "suited", strength
    elif gap == 0:
        return "connectors", strength
    else:
        return "high card", strength