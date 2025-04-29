#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for analyzing poker hands and calculating their strength.
"""

import logging
import poker
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
            # Convert our Card objects to poker library Card objects
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
    
    def _convert_to_poker_cards(self, cards: List[Card]) -> List:
        """
        Convert our Card objects to poker library Card objects.
        
        Args:
            cards (List[Card]): List of our Card objects
            
        Returns:
            List: List of poker library Card objects
        """
        poker_cards = []
        rank_mapping = {
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            '10': 'T', 'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'
        }
        
        suit_mapping = {
            'h': 'h', 'd': 'd', 'c': 'c', 's': 's'  # hearts, diamonds, clubs, spades
        }
        
        for card in cards:
            rank = rank_mapping.get(card.rank.lower(), card.rank)
            suit = suit_mapping.get(card.suit.lower(), card.suit)
            try:
                poker_card = poker.Card(rank + suit)
                poker_cards.append(poker_card)
            except ValueError as e:
                logger.error(f"Invalid card: {card.rank}{card.suit}, error: {e}")
                
        return poker_cards
    
    def _calculate_hand_strength(self, player_cards, community_cards):
        """
        Calculate the strength and type of the hand.
        
        Args:
            player_cards (List): List of player's hole cards as poker library Card objects
            community_cards (List): List of community cards as poker library Card objects
            
        Returns:
            Tuple[str, float]: Hand type and strength score
        """
        if len(player_cards) < 2:
            return "incomplete", 0.0
            
        all_cards = player_cards + community_cards
        
        if len(all_cards) < 5:
            # Pre-flop evaluation - evaluate hole cards only
            return self._evaluate_hole_cards(player_cards)
            
        # Create a poker Hand object
        hand = poker.Hand(all_cards)
        hand_type = str(hand).split()[0].lower()  # e.g., "Pair", "Two Pair", "Flush"
        
        # Map hand rank to a strength score from 0 to 1
        # The poker library ranks hands from 1 (High Card) to 9 (Straight Flush)
        hand_rank = hand.rank
        # Normalize to 0-1 scale, where 1 is the best possible hand
        hand_strength = (hand_rank - 1) / 8.0
        
        return hand_type, hand_strength
    
    def _evaluate_hole_cards(self, player_cards):
        """
        Evaluate hole cards (pre-flop).
        
        Args:
            player_cards (List): List of player's hole cards
            
        Returns:
            Tuple[str, float]: Hand type and strength score
        """
        if len(player_cards) != 2:
            return "incomplete", 0.0
            
        # Extract ranks and suits
        ranks = [card.rank for card in player_cards]
        suits = [card.suit for card in player_cards]
        
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
        return rank_values.get(rank.upper(), 2)
    
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
            _, hand_strength = self._calculate_hand_strength(player_cards, community_cards)
            
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
            
            # Combine factors
            win_prob = max(0, min(1, hand_strength - uncertainty_factor + position_factor))
            return win_prob
            
        return 0.0