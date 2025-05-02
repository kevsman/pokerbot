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

# Import the new detector classes
from game_detector import GameDetector
from player_card_detector import PlayerCardDetector
from community_card_detector import CommunityCardDetector
from pot_detector import PotDetector
from bet_detector import BetDetector
from stack_detector import StackDetector

logger = logging.getLogger(__name__)

# Configure pytesseract path - update this with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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
        
        # Initialize the game detector
        self.game_detector = GameDetector(config)
        
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)
        self.last_debug_time = 0  # To limit debug image saving frequency
        
        # Initialize the position detector
        from position_detector import PositionDetector
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
        
    def _detect_position(self, screenshot):
        """
        Detect the player's position based on dealer button location.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            str: The player's position ('early', 'middle', 'late', 'sb', 'bb', 'dealer').
        """
        # Use the position detector to detect the position
        return self.position_detector.detect_position(screenshot)
        
    def _detect_is_our_turn(self, screenshot):
        """
        Check if it's currently our turn to act by looking for active action buttons.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            bool: True if it's our turn to act, False otherwise.
        """
        try:
            h, w = screenshot.shape[:2]
            
            # Define region where action buttons are typically located
            # Usually at the bottom of the screen
            
            # For stasjonær
            roi_x = int(w * 0.3)
            roi_y = int(h * 0.78)
            roi_w = int(w * 0.2)
            roi_h = int(h * 0.15)
            
            # Add debugging info
            logger.info(f"[TURN DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[TURN DEBUG] Action button ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
            
            # Create a debug image for visualization
            debug_img = screenshot.copy()
            
            # Draw a rectangle around the ROI we're analyzing
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Detect blue buttons
            lower_blue = np.array([90, 40, 40])
            upper_blue = np.array([150, 255, 255])
            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            
            # Detect orange buttons - using color code #ffa604 (which is RGB: 255, 166, 4)
            # Convert RGB to HSV: Hue ~30, high Saturation, high Value
            lower_orange = np.array([15, 150, 150])  # Lower bound for orange color
            upper_orange = np.array([35, 255, 255])  # Upper bound for orange color
            orange_mask = cv2.inRange(hsv_roi, lower_orange, upper_orange)
            
            # Add a debug message for the color detection
            logger.info(f"[TURN DEBUG] Looking for orange buttons with color similar to #ffa604")
            
            # Combine blue and orange masks
            combined_mask = cv2.bitwise_or(blue_mask, orange_mask)
            
            # Find contours of potential buttons
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find button-like shapes
            active_buttons = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (buttons should be within a reasonable size range)
                min_button_area = (roi_w * roi_h) * 0.01  # At least 1% of ROI
                max_button_area = (roi_w * roi_h) * 0.15  # At most 15% of ROI
                
                logger.info(f"[TURN DEBUG] Contour area: {area}, min: {min_button_area}, max: {max_button_area}")
                
                if min_button_area < area < max_button_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (buttons are typically wider than tall)
                    aspect_ratio = w / h
                    logger.info(f"[TURN DEBUG] Button aspect ratio: {aspect_ratio}")
                    
                    if 1.5 < aspect_ratio < 5:
                        active_buttons.append((x, y, w, h))
                        
                        # Draw the button on the debug image
                        button_x = x + roi_x
                        button_y = y + roi_y
                        button_w = w
                        button_h = h
                        
                        # Draw rectangle around the button
                        cv2.rectangle(debug_img, (button_x, button_y), 
                                     (button_x + button_w, button_y + button_h), 
                                     (0, 255, 255), 2)
                        
                        # Check if the button is from the orange mask
                        button_mask = np.zeros_like(orange_mask)
                        cv2.drawContours(button_mask, [contour], 0, 255, -1)
                        orange_pixels = cv2.countNonZero(cv2.bitwise_and(orange_mask, button_mask))
                        button_type = "Orange" if orange_pixels > 0 else "Blue"
                        
                        # Add a label with button type
                        cv2.putText(debug_img, f"{button_type} Button {len(active_buttons)}", 
                                   (button_x, button_y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # If we found button-like shapes with active colors, it's likely our turn
            is_our_turn = len(active_buttons) > 0
            
            # Always save debug images to track detection quality
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            original_path = os.path.join(debug_dir, f'turn_roi_{timestamp}.png')
            blue_mask_path = os.path.join(debug_dir, f'turn_blue_mask_{timestamp}.png')
            orange_mask_path = os.path.join(debug_dir, f'turn_orange_mask_{timestamp}.png')
            combined_mask_path = os.path.join(debug_dir, f'turn_combined_mask_{timestamp}.png')
            debug_path = os.path.join(debug_dir, f'turn_debug_{timestamp}.png')
            
            cv2.imwrite(original_path, roi)
            cv2.imwrite(blue_mask_path, blue_mask)
            cv2.imwrite(orange_mask_path, orange_mask)
            cv2.imwrite(combined_mask_path, combined_mask)
            
            # Add turn status text to debug image
            cv2.rectangle(debug_img, (10, 10), (300, 50), (0, 0, 0), -1)  # Black background for text
            if is_our_turn:
                cv2.putText(debug_img, "STATUS: IT'S OUR TURN", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(debug_img, "STATUS: NOT OUR TURN", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw information about detected buttons
            y_pos = 80
            cv2.rectangle(debug_img, (10, 50), (300, 50 + 30 * (len(active_buttons) + 1)), (0, 0, 0), -1)
            cv2.putText(debug_img, f"Detected {len(active_buttons)} buttons", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
            
            for i, (x, y, w, h) in enumerate(active_buttons):
                # Check which mask this button belongs to
                button_mask = np.zeros_like(orange_mask)
                button_contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
                cv2.drawContours(button_mask, [button_contour], 0, 255, -1)
                orange_pixels = cv2.countNonZero(cv2.bitwise_and(orange_mask, button_mask))
                color_text = "Orange" if orange_pixels > 0 else "Blue"
                
                cv2.putText(debug_img, f"{color_text} Button {i+1}: ({x},{y}) {w}x{h}", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_pos += 30
                
            # Save the full debug image
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"[TURN DEBUG] Saved debug images to {debug_dir}")
            
            # Log the result
            if is_our_turn:
                logger.info("[TURN DEBUG] It's our turn to act")
            else:
                logger.info("[TURN DEBUG] It's not our turn to act")
                
            return is_our_turn
            
        except Exception as e:
            logger.exception(f"Error detecting if it's our turn: {e}")
            return False
        
    def _detect_available_actions(self, screenshot):
        """
        Detect available actions (fold, check, call, bet, raise) based on visible buttons.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            list: List of available action strings.
        """
        available_actions = []
        
        try:
            h, w = screenshot.shape[:2]
            
            # Define the region where action buttons are typically located           
            # For stasjonær
            roi_x = int(w * 0.3)
            roi_y = int(h * 0.81)
            roi_w = int(w * 0.2)
            roi_h = int(h * 0.11)
            
            # Add debugging info
            logger.info(f"[ACTIONS DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[ACTIONS DEBUG] Action buttons ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
            
            # Create a debug image for visualization
            debug_img = screenshot.copy()
            
            # Draw a rectangle around the ROI we're analyzing
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Create a copy of the ROI for debugging
            roi_debug = roi.copy()
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Detect blue buttons
            lower_blue = np.array([90, 40, 40])
            upper_blue = np.array([150, 255, 255])
            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            
            # Detect orange buttons - using color code #ffa604 (which is RGB: 255, 166, 4)
            lower_orange = np.array([15, 150, 150])
            upper_orange = np.array([35, 255, 255])
            orange_mask = cv2.inRange(hsv_roi, lower_orange, upper_orange)
            
            # Combine masks for all action buttons
            combined_mask = cv2.bitwise_or(blue_mask, orange_mask)
            
            # Define masks for button regions (left, middle, right)
            third_width = roi_w // 3
            
            left_mask = np.zeros_like(blue_mask)
            left_mask[:, :third_width] = 255
            
            middle_mask = np.zeros_like(blue_mask)
            middle_mask[:, third_width:2*third_width] = 255
            
            right_mask = np.zeros_like(blue_mask)
            right_mask[:, 2*third_width:] = 255
            
            # Draw region dividers on the ROI debug image
            cv2.line(roi_debug, (third_width, 0), (third_width, roi_h), (0, 255, 255), 2)
            cv2.line(roi_debug, (2*third_width, 0), (2*third_width, roi_h), (0, 255, 255), 2)
            
            # Label the regions
            cv2.putText(roi_debug, "FOLD", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(roi_debug, "CHECK/CALL", (third_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(roi_debug, "BET/RAISE", (2*third_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Find active buttons in each region using the combined mask
            # Left button (typically fold)
            left_buttons = cv2.bitwise_and(combined_mask, left_mask)
            left_button_active = cv2.countNonZero(left_buttons) > 50
            if left_button_active:
                available_actions.append("fold")
                logger.info("[ACTIONS DEBUG] Detected FOLD button")
            
            # Middle button (typically check/call)
            middle_buttons = cv2.bitwise_and(combined_mask, middle_mask)
            middle_button_active = cv2.countNonZero(middle_buttons) > 50
            
            check_or_call = "unknown"
            if middle_button_active:
                # Try to determine if it's check or call using OCR
                # Crop the middle section
                middle_roi = roi[:, third_width:2*third_width]
                
                # Preprocess for better OCR
                middle_gray = cv2.cvtColor(middle_roi, cv2.COLOR_BGR2GRAY)
                _, middle_binary = cv2.threshold(middle_gray, 150, 255, cv2.THRESH_BINARY)
                
                # Save the middle button crop for debugging
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
                middle_button_path = os.path.join(debug_dir, f'actions_middle_button_{timestamp}.png')
                middle_binary_path = os.path.join(debug_dir, f'actions_middle_binary_{timestamp}.png')
                cv2.imwrite(middle_button_path, middle_roi)
                cv2.imwrite(middle_binary_path, middle_binary)
                
                # Use OCR to read the text
                middle_text = pytesseract.image_to_string(middle_binary).lower()
                logger.info(f"[ACTIONS DEBUG] Middle button OCR text: '{middle_text}'")
                
                if 'check' in middle_text:
                    available_actions.append("check")
                    check_or_call = "check"
                    logger.info("[ACTIONS DEBUG] Detected CHECK button")
                elif 'call' in middle_text:
                    available_actions.append("call")
                    check_or_call = "call"
                    logger.info("[ACTIONS DEBUG] Detected CALL button")
                else:
                    # If we can't determine, add both possibilities
                    available_actions.append("check")
                    available_actions.append("call")
                    check_or_call = "check/call"
                    logger.info("[ACTIONS DEBUG] Detected button but couldn't determine if CHECK or CALL")
            
            # Right button (typically bet/raise)
            right_buttons = cv2.bitwise_and(combined_mask, right_mask)
            right_button_active = cv2.countNonZero(right_buttons) > 50
            
            bet_or_raise = "unknown"
            if right_button_active:
                # Try to determine if it's bet or raise
                right_roi = roi[:, 2*third_width:]
                
                # Preprocess for better OCR
                right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
                _, right_binary = cv2.threshold(right_gray, 150, 255, cv2.THRESH_BINARY)
                
                # Save the right button crop for debugging
                right_button_path = os.path.join(debug_dir, f'actions_right_button_{timestamp}.png')
                right_binary_path = os.path.join(debug_dir, f'actions_right_binary_{timestamp}.png')
                cv2.imwrite(right_button_path, right_roi)
                cv2.imwrite(right_binary_path, right_binary)
                
                # Use OCR to read text
                right_text = pytesseract.image_to_string(right_binary).lower()
                logger.info(f"[ACTIONS DEBUG] Right button OCR text: '{right_text}'")
                
                if 'bet' in right_text:
                    available_actions.append("bet")
                    bet_or_raise = "bet"
                    logger.info("[ACTIONS DEBUG] Detected BET button")
                elif 'raise' in right_text:
                    available_actions.append("raise")
                    bet_or_raise = "raise"
                    logger.info("[ACTIONS DEBUG] Detected RAISE button")
                else:
                    # If we can't determine, add both possibilities
                    available_actions.append("bet")
                    available_actions.append("raise")
                    bet_or_raise = "bet/raise"
                    logger.info("[ACTIONS DEBUG] Detected button but couldn't determine if BET or RAISE")
            
            # If no specific actions were detected but we previously determined it's our turn,
            # include default actions as a fallback
            if not available_actions:
                is_our_turn = self._detect_is_our_turn(screenshot)
                if is_our_turn:
                    available_actions = ["fold", "check", "bet"]
                    logger.info("[ACTIONS DEBUG] No buttons detected but it's our turn, using default actions")
                else:
                    logger.info("[ACTIONS DEBUG] No buttons detected and it's not our turn")
            
            # Highlight detected buttons on the debug image
            if left_button_active:
                cv2.rectangle(roi_debug, (0, 30), (third_width-1, roi_h-10), (0, 0, 255), 2)
                cv2.putText(roi_debug, "FOLD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if middle_button_active:
                cv2.rectangle(roi_debug, (third_width, 30), (2*third_width-1, roi_h-10), (0, 0, 255), 2)
                cv2.putText(roi_debug, check_or_call.upper(), (third_width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if right_button_active:
                cv2.rectangle(roi_debug, (2*third_width, 30), (roi_w-1, roi_h-10), (0, 0, 255), 2)
                cv2.putText(roi_debug, bet_or_raise.upper(), (2*third_width + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add detected actions to main debug image
            cv2.rectangle(debug_img, (10, 10), (350, 50 + 20 * len(available_actions)), (0, 0, 0), -1)  # Black background
            cv2.putText(debug_img, "Available Actions:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            for i, action in enumerate(available_actions):
                cv2.putText(debug_img, f"- {action.upper()}", (30, 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Save all debug images
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            roi_path = os.path.join(debug_dir, f'actions_roi_{timestamp}.png')
            roi_debug_path = os.path.join(debug_dir, f'actions_roi_debug_{timestamp}.png')
            blue_mask_path = os.path.join(debug_dir, f'actions_blue_mask_{timestamp}.png')
            orange_mask_path = os.path.join(debug_dir, f'actions_orange_mask_{timestamp}.png')
            combined_mask_path = os.path.join(debug_dir, f'actions_combined_mask_{timestamp}.png')
            debug_path = os.path.join(debug_dir, f'actions_debug_{timestamp}.png')
            
            cv2.imwrite(roi_path, roi)
            cv2.imwrite(roi_debug_path, roi_debug)
            cv2.imwrite(blue_mask_path, blue_mask)
            cv2.imwrite(orange_mask_path, orange_mask)
            cv2.imwrite(combined_mask_path, combined_mask)
            cv2.imwrite(debug_path, debug_img)
            
            logger.info(f"[ACTIONS DEBUG] Saved debug images to {debug_dir}")
            logger.info(f"[ACTIONS DEBUG] Detected available actions: {available_actions}")
                
        except Exception as e:
            logger.exception(f"Error detecting available actions: {e}")
            # Provide default actions as fallback
            available_actions = ["fold", "check", "bet"]
            
        return available_actions
    
    def _preprocess_for_ocr(self, img):
        """Preprocess image for better OCR results."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh