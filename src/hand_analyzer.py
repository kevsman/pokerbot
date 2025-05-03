#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for analyzing poker hands and calculating their strength.
"""

import logging
import random
from typing import List, Dict
from models import Card
from game_state import GameState
from card_utils import convert_to_poker_cards, validate_cards
from hand_evaluator import calculate_hand_strength
from win_probability import estimate_win_probability

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
            player_cards = convert_to_poker_cards(game_state.player_cards)
            community_cards = convert_to_poker_cards(game_state.community_cards)
            
            # Validate the cards
            validate_cards(player_cards + community_cards)
            
            # Get hand type and strength
            hand_type, hand_strength = calculate_hand_strength(player_cards, community_cards)
            
            # Estimate win probability
            win_probability = estimate_win_probability(
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