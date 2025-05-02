#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for analyzing poker hands and calculating their strength.
"""

import logging
import random
# Replacing poker with pypokerengine
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards
from typing import List, Dict, Tuple
from game_state import Card, GameState

logger = logging.getLogger(__name__)


class HandAnalyzer:
    """Class for analyzing poker hands and calculating their strength."""
    
    def __init__(self, config=None):
        """Initialize the hand analyzer with optional configuration."""
        self.config = config or {}
        logger.info("Hand analyzer initialized")
        
    def analyze(self, game_state: GameState) -> Dict:
        """
        Analyze the current hand based on player cards and community cards.
        
        Args:
            game_state (GameState): Current game state with player and community cards
            
        Returns:
            Dict: Hand analysis with strength score, hand type, and win probability
        """
        if not game_state.player_cards:
            logger.warning("Cannot analyze hand: No player cards detected")
            return {
                'hand_type': 'unknown',
                'strength': 0,
                'win_probability': 0.0
            }
            
        try:
            # Convert our Card objects to pypokerengine format
            player_cards = self._convert_to_poker_cards(game_state.player_cards)
            community_cards = self._convert_to_poker_cards(game_state.community_cards)
            
            # Get hand type and strength
            hand_type, hand_strength = self._calculate_hand_strength(player_cards, community_cards)
            
            # Estimate win probability
            win_probability = self._estimate_win_probability(
                player_cards, 
                community_cards,
                game_state.position
            )
            
            result = {
                'hand_type': hand_type,
                'strength': hand_strength,
                'win_probability': win_probability
            }
            
            logger.info(f"Hand analysis: {result['hand_type']} (strength: {result['strength']:.2f}, "
                       f"win probability: {result['win_probability']:.2%})")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error analyzing hand: {e}")
            return {
                'hand_type': 'error',
                'strength': 0,
                'win_probability': 0.0
            }
    
    def _convert_to_poker_cards(self, cards: List[Card]) -> List[str]:
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
    
    def _calculate_hand_strength(self, player_cards, community_cards):
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
            return self._evaluate_hole_cards(player_cards)
            
        try:
            # Validate the detected cards before evaluation
            self._validate_cards(player_cards + community_cards)
            
            # Convert string format to pypokerengine Card objects using gen_cards
            hole_cards = gen_cards(player_cards)
            community_cards_obj = gen_cards(community_cards)
            
            # Use HandEvaluator to get the score
            score = HandEvaluator.eval_hand(hole_cards, community_cards_obj)
            
            # Log the raw score for debugging
            logger.debug(f"Raw hand evaluation score: {score}")
            
            # Map the score to a hand type and strength between 0 and 1
            # PyPokerEngine scores range from high card (1) to straight flush (8000+)
            if score >= 8000 and self._can_form_straight_flush(player_cards, community_cards):  
                # Straight flush - extra validation to ensure it's real
                hand_type = "straight flush"
                hand_strength = 1.0
            elif score >= 7000 and self._can_form_four_of_a_kind(player_cards, community_cards):  
                # Four of a kind - with extra validation
                hand_type = "four of a kind"
                hand_strength = 0.9
            elif score >= 6000 and self._can_form_full_house(player_cards, community_cards):  
                # Full house - with extra validation
                hand_type = "full house" 
                hand_strength = 0.8
            elif score >= 5000 and self._can_form_flush(player_cards, community_cards):  
                # Flush - with extra validation
                hand_type = "flush"
                hand_strength = 0.7
            elif score >= 4000 and self._can_form_straight(player_cards, community_cards):  
                # Straight - with extra validation
                hand_type = "straight"
                hand_strength = 0.6
            elif score >= 3000 and self._can_form_three_of_a_kind(player_cards, community_cards):  
                # Three of a kind - with extra validation
                hand_type = "three of a kind"
                hand_strength = 0.5
            elif score >= 2000 and self._can_form_two_pair(player_cards, community_cards):  
                # Two pair - with extra validation
                hand_type = "two pair"
                hand_strength = 0.4
            elif score >= 1000 and self._can_form_pair(player_cards, community_cards):  
                # Pair - with extra validation
                hand_type = "pair"
                hand_strength = 0.3
            else:  # High card
                hand_type = "high card"
                hand_strength = 0.1 + (score / 1000.0) * 0.2  # Scale from 0.1 to 0.3
            
            # Double-check hand evaluation for common mistakes
            self._verify_hand_evaluation(player_cards, community_cards, hand_type)
            
            # Log the identified hand for debugging
            logger.debug(f"Identified hand: {hand_type} with strength {hand_strength}")
            
            return hand_type, hand_strength
        except Exception as e:
            logger.exception(f"Error calculating hand strength: {e}")
            return "error", 0.0
    
    def _validate_cards(self, cards):
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
            
    def _can_form_straight_flush(self, player_cards, community_cards):
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
                rank_values = [self._get_rank_value(rank) for rank in ranks]
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
    
    def _evaluate_hole_cards(self, player_cards):
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
            rank_value = self._get_rank_value(ranks[0])
            # Scale from 0.5 (pair of 2s) to 0.75 (pair of Aces)
            strength = 0.5 + ((rank_value - 2) / (14 - 2)) * 0.25
            return "pair", strength
            
        # Check if suited
        suited = suits[0] == suits[1]
        
        # Get rank values
        rank_values = [self._get_rank_value(rank) for rank in ranks]
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
    
    def _get_rank_value(self, rank):
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
    
    def _estimate_win_probability(self, player_cards, community_cards, position):
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
            hand_type, hand_strength = self._calculate_hand_strength(player_cards, community_cards)
            
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

    def _verify_hand_evaluation(self, player_cards, community_cards, hand_type):
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
                    rank_values = [self._get_rank_value(rank) for rank in ranks]
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
                        
    def _can_form_four_of_a_kind(self, player_cards, community_cards):
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
        
    def _can_form_full_house(self, player_cards, community_cards):
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
        
    def _can_form_flush(self, player_cards, community_cards):
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
        
    def _can_form_straight(self, player_cards, community_cards):
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
        rank_values = set(self._get_rank_value(card[1]) for card in all_cards)
        
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
        
    def _can_form_three_of_a_kind(self, player_cards, community_cards):
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
        
    def _can_form_two_pair(self, player_cards, community_cards):
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
        
    def _can_form_pair(self, player_cards, community_cards):
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