#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting poker game state from screenshots.
"""

import logging
import cv2
import numpy as np
import pytesseract
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from models import Card

# Import the detector classes
from game_detector import GameDetector
from player_card_detector import PlayerCardDetector
from community_card_detector import CommunityCardDetector
from pot_detector import PotDetector
from bet_detector import BetDetector
from stack_detector import StackDetector
from position_detector import PositionDetector
from turn_detector import TurnDetector
from action_detector import ActionDetector

logger = logging.getLogger(__name__)

# Configure pytesseract path - update this with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class GameState:
    """Class representing the current poker game state."""
    is_active: bool = False
    player_cards: List[Card] = None
    community_cards: List[Card] = None
    pot_size: float = 0.0
    current_bet: float = 0.0
    player_stack: float = 0.0
    position: str = ""
    is_our_turn: bool = False
    available_actions: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.player_cards is None:
            self.player_cards = []
        if self.community_cards is None:
            self.community_cards = []
        if self.available_actions is None:
            self.available_actions = []


class GameStateDetector:
    """Class for detecting the current state of a poker game from screenshots."""
    
    def __init__(self, config=None):
        """Initialize the game state detector with optional configuration."""
        self.config = config or {}
        self.card_templates = {}
        self.button_templates = {}
        
        # Initialize the game detector
        self.game_detector = GameDetector(config)
        
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)
        self.last_debug_time = 0  # To limit debug image saving frequency
        
        # Initialize the position detector
        self.position_detector = PositionDetector(config)
        
        # Initialize the card identifier
        from card_identifier import CardIdentifier
        self.card_identifier = CardIdentifier(debug_mode=self.debug_mode)
        
        # Initialize the player card detector
        self.player_card_detector = PlayerCardDetector(config)
        self.player_card_detector.set_card_identifier(self.card_identifier)
        
        # Initialize the community card detector
        self.community_card_detector = CommunityCardDetector(
            card_identifier=self.card_identifier, 
            debug_mode=self.debug_mode
        )
        
        # Initialize the pot detector
        self.pot_detector = PotDetector(config, debug_mode=self.debug_mode)
        
        # Initialize the bet detector
        self.bet_detector = BetDetector(config, debug_mode=self.debug_mode)
        
        # Initialize the stack detector
        self.stack_detector = StackDetector(config, debug_mode=self.debug_mode)
        
        # Initialize the turn detector
        self.turn_detector = TurnDetector(config)
        
        # Initialize the action detector
        self.action_detector = ActionDetector(config)
        
        logger.info("Game state detector initialized")
        if self.debug_mode:
            logger.info("Debug mode is enabled - will save debug images")
        else:
            logger.info("Debug mode is disabled - will not save debug images")
        
    def detect_state(self, screenshot):
        """
        Detect the current game state from a screenshot.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            GameState: The detected game state.
        """
        if screenshot is None:
            logger.error("Cannot detect game state from None screenshot")
            return GameState()
            
        game_state = GameState()
        
        try:
            # Log screenshot size for debugging
            if self.debug_mode:
                h, w = screenshot.shape[:2]
                logger.info(f"Screenshot size: {w}x{h}")
                
            # Check if the game is active (poker table is visible)
            game_state.is_active = self.game_detector.detect_active_game(screenshot)
            
            if game_state.is_active:
                # Detect player cards
                game_state.player_cards = self.player_card_detector.detect_player_cards(screenshot)
                
                # Detect community cards using the new detector
                game_state.community_cards = self.community_card_detector.detect_community_cards(screenshot)
                
                # Detect pot size with the new detector
                game_state.pot_size = self.pot_detector.detect_pot_size(screenshot)
                
                # Detect current bet with the new detector
                game_state.current_bet = self.bet_detector.detect_current_bet(screenshot)
                
                # Detect player stack with the new detector
                game_state.player_stack = self.stack_detector.detect_player_stack(screenshot)
                
                # Detect position using the position detector
                game_state.position = self.position_detector.detect_position(screenshot)
                
                # Check if it's our turn using the turn detector
                game_state.is_our_turn = self.turn_detector.detect_is_our_turn(screenshot)
                
                # Detect available actions using the action detector
                game_state.available_actions = self.action_detector.detect_available_actions(screenshot)
                
                logger.info(f"Game state detected: {len(game_state.player_cards)} player cards, "
                          f"{len(game_state.community_cards)} community cards, "
                          f"pot: {game_state.pot_size}, is_our_turn: {game_state.is_our_turn}")
            else:
                logger.info("No active game detected")
                
            return game_state
            
        except Exception as e:
            logger.exception(f"Error detecting game state: {e}")
            return GameState()
    
    def _preprocess_for_ocr(self, img):
        """Preprocess image for better OCR results."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh