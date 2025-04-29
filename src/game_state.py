#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting poker game state from screenshots.
"""

import logging
import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configure pytesseract path - update this with your Tesseract installation path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class Card:
    """Class representing a playing card."""
    rank: str
    suit: str
    
    def __str__(self):
        return f"{self.rank}{self.suit}"


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
        self.load_templates()
        logger.info("Game state detector initialized")
        
    def load_templates(self):
        """Load card and button templates from resources directory."""
        try:
            # In a real implementation, you would load all card images
            # For now, we'll just log that templates would be loaded
            logger.info("Templates would be loaded from resources directory")
            
            # Example of how template loading might be implemented:
            # for suit in ['h', 'd', 'c', 's']:  # hearts, diamonds, clubs, spades
            #     for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']:
            #         template_path = f"resources/cards/{rank}{suit}.png"
            #         template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            #         if template is not None:
            #             self.card_templates[f"{rank}{suit}"] = template
            
        except Exception as e:
            logger.exception(f"Error loading templates: {e}")
            
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
            # Check if the game is active (poker table is visible)
            game_state.is_active = self._detect_active_game(screenshot)
            
            if game_state.is_active:
                # Detect player cards
                game_state.player_cards = self._detect_player_cards(screenshot)
                
                # Detect community cards
                game_state.community_cards = self._detect_community_cards(screenshot)
                
                # Detect pot size
                game_state.pot_size = self._detect_pot_size(screenshot)
                
                # Detect current bet
                game_state.current_bet = self._detect_current_bet(screenshot)
                
                # Detect player stack
                game_state.player_stack = self._detect_player_stack(screenshot)
                
                # Detect position
                game_state.position = self._detect_position(screenshot)
                
                # Check if it's our turn
                game_state.is_our_turn = self._detect_is_our_turn(screenshot)
                
                # Detect available actions
                game_state.available_actions = self._detect_available_actions(screenshot)
                
                logger.info(f"Game state detected: {len(game_state.player_cards)} player cards, "
                          f"{len(game_state.community_cards)} community cards, "
                          f"pot: {game_state.pot_size}, is_our_turn: {game_state.is_our_turn}")
            else:
                logger.info("No active game detected")
                
            return game_state
            
        except Exception as e:
            logger.exception(f"Error detecting game state: {e}")
            return GameState()
            
    def _detect_active_game(self, screenshot):
        """Check if a poker game is active in the screenshot."""
        # In a real implementation, you would look for poker table elements
        # For now, we'll assume a game is active if we can find some key elements
        # like the poker table color or logo
        
        # Placeholder implementation - in a real project, you'd implement actual detection
        return True
        
    def _detect_player_cards(self, screenshot):
        """Detect player's hole cards from the screenshot."""
        # Placeholder - in a real implementation, you would use template matching or ML
        return []
        
    def _detect_community_cards(self, screenshot):
        """Detect community cards from the screenshot."""
        # Placeholder - in a real implementation, you would use template matching or ML
        return []
        
    def _detect_pot_size(self, screenshot):
        """Detect the current pot size from the screenshot using OCR."""
        # Placeholder - in a real implementation, you would:
        # 1. Crop the region where pot size is displayed
        # 2. Preprocess the image for better OCR
        # 3. Use pytesseract to extract text
        # 4. Parse the text to get the pot size as a float
        return 0.0
        
    def _detect_current_bet(self, screenshot):
        """Detect the current bet from the screenshot."""
        # Placeholder implementation
        return 0.0
        
    def _detect_player_stack(self, screenshot):
        """Detect the player's chip stack from the screenshot."""
        # Placeholder implementation
        return 100.0
        
    def _detect_position(self, screenshot):
        """Detect the player's position (early, middle, late, blinds)."""
        # Placeholder implementation
        return "unknown"
        
    def _detect_is_our_turn(self, screenshot):
        """Check if it's currently our turn to act."""
        # Placeholder implementation
        return False
        
    def _detect_available_actions(self, screenshot):
        """Detect available actions (fold, check, call, bet, raise)."""
        # Placeholder implementation
        return ["fold", "check", "bet"]
        
    def _preprocess_for_ocr(self, img):
        """Preprocess image for better OCR results."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh