#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for validating poker hand types.
"""

import logging
from typing import List
from card_utils import get_rank_value

logger = logging.getLogger(__name__)


def can_form_straight_flush(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form a straight flush with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if a straight flush is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 5 cards total to form a straight flush
    if len(all_cards) < 5:
        return False
        
    # Group cards by suit
    cards_by_suit = {'H': [], 'D': [], 'C': [], 'S': []}
    
    for card in all_cards:
        suit = card[0]
        rank = card[1]
        cards_by_suit[suit].append(rank)
        
    # Check if any suit has at least 5 cards
    for suit, ranks in cards_by_suit.items():
        if len(ranks) >= 5:
            # Convert ranks to numerical values
            rank_values = [get_rank_value(rank) for rank in ranks]
            rank_values.sort()
            
            # Check for straight (5 consecutive values)
            # Special case for A-5 straight (A can be low)
            if 14 in rank_values:  # Ace
                rank_values.append(1)  # Add low ace
            
            # Check for 5 consecutive cards
            for i in range(len(rank_values) - 4):
                if rank_values[i:i+5] == list(range(rank_values[i], rank_values[i] + 5)):
                    return True
                    
    return False


def can_form_four_of_a_kind(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form four of a kind with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if four of a kind is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 4 cards total to form four of a kind
    if len(all_cards) < 4:
        logger.warning(f"Cannot form four of a kind with only {len(all_cards)} cards")
        return False
        
    # Count occurrences of each rank
    rank_counts = {}
    for card in all_cards:
        rank = card[1]
        if rank not in rank_counts:
            rank_counts[rank] = 0
        rank_counts[rank] += 1
        
    # Check if any rank appears exactly 4 times
    has_four = any(count >= 4 for count in rank_counts.values())
    
    if not has_four:
        logger.warning(f"Hand incorrectly identified as four of a kind. Rank distribution: {rank_counts}")
        
    return has_four


def can_form_full_house(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form a full house with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if a full house is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 5 cards total to form a full house
    if len(all_cards) < 5:
        return False
        
    # Count occurrences of each rank
    rank_counts = {}
    for card in all_cards:
        rank = card[1]
        if rank not in rank_counts:
            rank_counts[rank] = 0
        rank_counts[rank] += 1
        
    # Sort ranks by count (descending)
    sorted_counts = sorted(rank_counts.values(), reverse=True)
    
    # For a full house, we need at least one rank with 3+ cards and another with 2+ cards
    if len(sorted_counts) >= 2 and sorted_counts[0] >= 3 and sorted_counts[1] >= 2:
        return True
        
    return False


def can_form_flush(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form a flush with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if a flush is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 5 cards total to form a flush
    if len(all_cards) < 5:
        return False
        
    # Count occurrences of each suit
    suit_counts = {'H': 0, 'D': 0, 'C': 0, 'S': 0}
    for card in all_cards:
        suit = card[0]
        suit_counts[suit] += 1
        
    # Check if any suit appears at least 5 times
    has_flush = any(count >= 5 for count in suit_counts.values())
    
    return has_flush


def can_form_straight(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form a straight with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if a straight is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 5 cards total to form a straight
    if len(all_cards) < 5:
        return False
        
    # Get unique ranks as values
    rank_values = set(get_rank_value(card[1]) for card in all_cards)
    
    # Special case for A-5 straight
    if 14 in rank_values:  # Ace
        rank_values.add(1)  # Add Ace as 1
        
    # Sort rank values
    rank_values = sorted(rank_values)
    
    # Check for 5 consecutive ranks
    for i in range(len(rank_values) - 4):
        if rank_values[i:i+5] == list(range(rank_values[i], rank_values[i] + 5)):
            return True
            
    return False


def can_form_three_of_a_kind(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form three of a kind with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if three of a kind is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 3 cards total to form three of a kind
    if len(all_cards) < 3:
        return False
        
    # Count occurrences of each rank
    rank_counts = {}
    for card in all_cards:
        rank = card[1]
        if rank not in rank_counts:
            rank_counts[rank] = 0
        rank_counts[rank] += 1
        
    # Check if any rank appears at least 3 times
    has_three = any(count >= 3 for count in rank_counts.values())
    
    return has_three


def can_form_two_pair(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form two pairs with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if two pairs are possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 4 cards total to form two pairs
    if len(all_cards) < 4:
        return False
        
    # Count occurrences of each rank
    rank_counts = {}
    for card in all_cards:
        rank = card[1]
        if rank not in rank_counts:
            rank_counts[rank] = 0
        rank_counts[rank] += 1
        
    # Count how many pairs we have
    pairs = sum(1 for count in rank_counts.values() if count >= 2)
    
    return pairs >= 2


def can_form_pair(player_cards: List[str], community_cards: List[str]) -> bool:
    """
    Check if it's possible to form a pair with the given cards.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        
    Returns:
        bool: True if a pair is possible, False otherwise
    """
    all_cards = player_cards + community_cards
    
    # We need at least 2 cards total to form a pair
    if len(all_cards) < 2:
        return False
        
    # Count occurrences of each rank
    rank_counts = {}
    for card in all_cards:
        rank = card[1]
        if rank not in rank_counts:
            rank_counts[rank] = 0
        rank_counts[rank] += 1
        
    # Check if any rank appears at least twice
    has_pair = any(count >= 2 for count in rank_counts.values())
    
    return has_pair


def verify_hand_evaluation(player_cards: List[str], community_cards: List[str], hand_type: str):
    """
    Double-check the hand evaluation for common mistakes.
    
    Args:
        player_cards (List[str]): Player hole cards
        community_cards (List[str]): Community cards
        hand_type (str): Currently identified hand type
    """
    # Special validation for straight flush which is being incorrectly detected
    if hand_type == "straight flush":
        all_cards = player_cards + community_cards
        
        # Check if we have the minimum necessary cards
        if len(all_cards) < 5:
            logger.warning("Hand incorrectly identified as straight flush with fewer than 5 cards")
            return
        
        # Detailed logging for debugging straight flush detection
        logger.debug(f"Verifying straight flush: player cards {player_cards}, community cards {community_cards}")
        
        # Count cards by suit
        suits_count = {'H': 0, 'D': 0, 'C': 0, 'S': 0}
        for card in all_cards:
            suits_count[card[0]] += 1
        
        # Check if any suit has at least 5 cards
        has_five_of_same_suit = any(count >= 5 for count in suits_count.values())
        
        if not has_five_of_same_suit:
            logger.warning(f"Hand incorrectly identified as straight flush: no suit has 5+ cards. Suits count: {suits_count}")
            
        # For debugging, log which suit appeared to form the straight flush
        for suit, count in suits_count.items():
            if count >= 5:
                cards_of_suit = [card for card in all_cards if card[0] == suit]
                ranks = [card[1] for card in cards_of_suit]
                rank_values = [get_rank_value(rank) for rank in ranks]
                rank_values.sort()
                logger.debug(f"Potential straight flush with {suit}: {cards_of_suit}, rank values: {rank_values}")
                
                # Check for 5 consecutive cards
                has_straight = False
                if 14 in rank_values:  # Ace
                    # Also consider Ace as 1 for A-5 straight
                    rank_values.append(1)
                    
                for i in range(len(rank_values) - 4):
                    if rank_values[i:i+5] == list(range(rank_values[i], rank_values[i] + 5)):
                        has_straight = True
                        logger.debug(f"Found straight in suit {suit}: {rank_values[i:i+5]}")
                        break
                        
                if not has_straight:
                    logger.warning(f"Hand incorrectly identified as straight flush: cards of suit {suit} don't form a straight")