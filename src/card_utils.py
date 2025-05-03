#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility module for card operations in the poker bot.
"""

import logging
from typing import List
from models import Card

logger = logging.getLogger(__name__)


def convert_to_poker_cards(cards: List[Card]) -> List[str]:
    """
    Convert our Card objects to pypokerengine format.
    
    Args:
        cards (List[Card]): List of our Card objects
        
    Returns:
        List[str]: List of cards in pypokerengine format
    """
    poker_cards = []
    rank_mapping = {
        '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
        '10': 'T', 'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'
    }
    
    suit_mapping = {
        'h': 'H', 'd': 'D', 'c': 'C', 's': 'S'  # hearts, diamonds, clubs, spades
    }
    
    for card in cards:
        rank = rank_mapping.get(card.rank.lower(), card.rank.upper())
        suit = suit_mapping.get(card.suit.lower(), card.suit.upper())
        try:
            # PyPokerEngine uses format like "SA" for Ace of Spades
            poker_card = suit + rank
            poker_cards.append(poker_card)
        except ValueError as e:
            logger.error(f"Invalid card: {card.rank}{card.suit}, error: {e}")
            
    return poker_cards


def validate_cards(cards: List[str]) -> None:
    """
    Validate the cards to ensure they're legitimate.
    
    Args:
        cards (List[str]): List of cards to validate
    """
    valid_ranks = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'}
    valid_suits = {'H', 'D', 'C', 'S'}
    
    # Check that each card has a valid rank and suit
    for card in cards:
        if len(card) != 2 or card[0] not in valid_suits or card[1] not in valid_ranks:
            logger.warning(f"Invalid card detected: {card}")
            
    # Check for duplicates
    if len(cards) != len(set(cards)):
        duplicates = [card for card in cards if cards.count(card) > 1]
        logger.warning(f"Duplicate cards detected: {duplicates}")


def get_rank_value(rank: str) -> int:
    """
    Convert card rank to numerical value.
    
    Args:
        rank (str): Card rank
        
    Returns:
        int: Numerical value of the rank (2-14)
    """
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    return rank_values.get(rank, 2)