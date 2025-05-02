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

# Add import for window detection
try:
    import pygetwindow as gw
    WINDOW_DETECTION_AVAILABLE = True
except ImportError:
    WINDOW_DETECTION_AVAILABLE = False
    print("pygetwindow not installed. Window detection will not be available.")
    print("To enable window detection, install with: pip install pygetwindow")

logger = logging.getLogger(__name__)

# Configure pytesseract path - update this with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# List of possible poker client window titles
POKER_CLIENT_TITLES = [
    "Poker", "PokerStars", "GGPoker", "888Poker", "PartyPoker",
    "Winamax", "Unibet", "ACR Poker", "Ignition Poker", "Bovada"
]


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
        self.client_templates = {}
        self.resources_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')
        # Also check images folder as fallback
        self.images_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
        self.load_templates()
        # Lower the match threshold for better detection
        self.match_threshold = 0.6  # Changed from 0.7 to 0.6
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)
        self.last_debug_time = 0  # To limit debug image saving frequency
        self.use_window_detection = config.get('use_window_detection', WINDOW_DETECTION_AVAILABLE)
        self.target_window_title = config.get('window_title', None)
        logger.info("Game state detector initialized")
        if self.use_window_detection:
            logger.info("Window title detection is enabled")
        if self.debug_mode:
            logger.info("Debug mode is enabled - will save debug images")
        else:
            logger.info("Debug mode is disabled - will not save debug images")
        
    def load_templates(self):
        """Load card and button templates from resources directory."""
        try:
            # Load client reference images from resources directory
            for i in range(1, 6):
                client_path = os.path.join(self.resources_path, f'client{i}.png' if i > 1 else 'client.png')
                if os.path.exists(client_path):
                    client_img = cv2.imread(client_path, cv2.IMREAD_COLOR)
                    if client_img is not None:
                        self.client_templates[f'client{i}'] = client_img
                        logger.info(f"Loaded client template {i} from resources")
                    else:
                        logger.warning(f"Failed to load client template {client_path}")
            
            # If no templates loaded from resources, try the images directory
            if not self.client_templates:
                logger.info("No templates found in resources, trying images directory")
                for i in range(1, 6):
                    client_path = os.path.join(self.images_path, f'client{i}.png' if i > 1 else 'client.png')
                    if os.path.exists(client_path):
                        client_img = cv2.imread(client_path, cv2.IMREAD_COLOR)
                        if client_img is not None:
                            self.client_templates[f'client{i}'] = client_img
                            logger.info(f"Loaded client template {i} from images")
                        else:
                            logger.warning(f"Failed to load client template {client_path}")
                        
            # Check if we loaded any client templates
            if not self.client_templates:
                logger.warning("No client templates were loaded")
            else:
                logger.info(f"Loaded {len(self.client_templates)} client templates")
                # Log template sizes to help with debugging
                for name, template in self.client_templates.items():
                    h, w = template.shape[:2]
                    logger.info(f"Template {name} size: {w}x{h}")
            
            # In a full implementation, we would also load:
            # 1. Card templates (for card recognition)
            # 2. Button templates (for action buttons)
            # 3. Position templates (for dealer button, etc.)
            
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
            # Log screenshot size for debugging
            if self.debug_mode:
                h, w = screenshot.shape[:2]
                logger.info(f"Screenshot size: {w}x{h}")
                
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
        """
        Check if a poker game is active using multiple detection methods.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            bool: True if a poker game is detected, False otherwise.
        """
        # Try window title detection first if enabled
        if self.use_window_detection and WINDOW_DETECTION_AVAILABLE:
            if self._detect_poker_window():
                logger.info("Poker window detected by window title")
                return True
                
        # Fall back to template matching if window detection failed or is disabled
        if self.client_templates:
            return self._detect_active_game_by_template(screenshot)
        else:
            logger.warning("No client templates available for game detection and window detection failed")
            return False
            
    def _detect_poker_window(self):
        """
        Check if a poker client window is currently open by window title.
        
        Returns:
            bool: True if a poker window is found, False otherwise.
        """
        if not WINDOW_DETECTION_AVAILABLE:
            return False
            
        try:
            # If a specific window title was configured, check only for that
            if self.target_window_title:
                windows = gw.getWindowsWithTitle(self.target_window_title)
                if windows and any(w.visible for w in windows):
                    logger.info(f"Target window '{self.target_window_title}' found and visible")
                    return True
            else:
                # Otherwise check for any known poker client
                for title in POKER_CLIENT_TITLES:
                    windows = gw.getWindowsWithTitle(title)
                    if windows and any(w.visible for w in windows):
                        logger.info(f"Poker client window '{title}' found and visible")
                        return True
                        
            logger.debug("No poker client window found by title detection")
            return False
        except Exception as e:
            logger.exception(f"Error detecting poker window: {e}")
            return False
            
    def _detect_active_game_by_template(self, screenshot):
        """
        Check if a poker game is active in the screenshot by matching with client templates.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            bool: True if a poker game is detected, False otherwise.
        """
        if not self.client_templates:
            logger.warning("No client templates available for game detection")
            return False
        
        # Try to match each client template
        best_match = 0
        best_client = None
        
        for client_name, template in self.client_templates.items():
            try:
                # Use template matching to find the client
                result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # Log match values for debugging
                logger.info(f"Template match for {client_name}: {max_val:.4f}")
                
                # Keep track of the best match
                if max_val > best_match:
                    best_match = max_val
                    best_client = client_name
                
                # If we have a good match, consider the game active
                if max_val >= self.match_threshold:
                    logger.info(f"Poker client detected ({client_name}), match score: {max_val:.4f}")
                    
                    # Save debug image if in debug mode (but limit frequency)
                    if self.debug_mode and time.time() - self.last_debug_time > 30:  # Save at most every 30 seconds
                        self.last_debug_time = time.time()
                        
                        # Draw rectangle around detected client
                        debug_img = screenshot.copy()
                        h, w = template.shape[:2]
                        top_left = max_loc
                        bottom_right = (top_left[0] + w, top_left[1] + h)
                        cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
                        
                        # Save the debug image
                        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_path = os.path.join(debug_dir, f'match_{int(time.time())}.png')
                        cv2.imwrite(debug_path, debug_img)
                        logger.info(f"Saved debug image to {debug_path}")
                    
                    return True
                    
            except Exception as e:
                logger.exception(f"Error matching client template {client_name}: {e}")
        
        logger.info(f"No active game detected by template matching. Best match: {best_client} with score {best_match:.4f} (threshold: {self.match_threshold})")
        
        # If the best match is close to the threshold, save a debug image
        if self.debug_mode and best_match > self.match_threshold - 0.1 and time.time() - self.last_debug_time > 30:
            self.last_debug_time = time.time()
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save the screenshot
            screenshot_path = os.path.join(debug_dir, f'no_match_{int(time.time())}.png')
            cv2.imwrite(screenshot_path, screenshot)
            logger.info(f"Saved near-miss screenshot to {screenshot_path}")
        
        return False
        
    def _detect_player_cards(self, screenshot):
        """
        Detect player's hole cards from the screenshot using color and shape analysis.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            List[Card]: List of detected cards.
        """
        cards = []
        try:
            # Common region for player cards (typically bottom center of the screen)
            h, w = screenshot.shape[:2]
            
            # Define region of interest (ROI) where player cards are usually located
            # These values need to be calibrated based on the specific poker client
            roi_x = int(w * 0.3)  # Start at 40% from the left
            roi_y = int(h * 0.6)  # Start at 60% from the top
            roi_w = int(w * 0.2)  # Width is 20% of the screen width
            roi_h = int(h * 0.10)  # Height is 15% of the screen height

            # Add debugging info
            logger.info(f"[CARD DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[CARD DEBUG] Player card ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")

            # Create a debug image for visualization
            debug_img = screenshot.copy()
            
            # Draw a rectangle around the ROI we're analyzing
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            # Draw crosshairs at the center of the ROI
            center_x = roi_x + roi_w // 2
            center_y = roi_y + roi_h // 2
            cv2.line(debug_img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(debug_img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
            
            # Draw coordinate text
            cv2.putText(debug_img, f"ROI: ({roi_x},{roi_y})", (roi_x, roi_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Convert to HSV color space which is better for color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Look for card-like shapes based on color thresholds
            # White/gray areas for card backgrounds
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
            
            # Find contours of potential cards
            contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logger.info(f"[CARD DEBUG] Found {len(contours)} potential card contours")
            
            # Filter contours to find card-like shapes
            card_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by area (cards should be reasonably sized)
                min_card_area = (roi_w * roi_h) * 0.03  # Cards take at least 3% of ROI
                max_card_area = (roi_w * roi_h) * 0.3   # Cards take at most 30% of ROI
                
                logger.info(f"[CARD DEBUG] Contour area: {area}, min: {min_card_area}, max: {max_card_area}")
                
                if min_card_area < area < max_card_area:
                    # Get bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if aspect ratio is card-like
                    # UPDATED: Now accepts aspect ratios from 0.8 to 1.6 to include your client's cards (around 0.9)
                    aspect_ratio = h / w
                    logger.info(f"[CARD DEBUG] Contour aspect ratio: {aspect_ratio}")
                    
                    if 0.8 < aspect_ratio < 1.6:  # Widened range to include aspect ratios around 0.9
                        card_contours += 1
                        # Draw the contour in the debug image
                        cv2.drawContours(debug_img, [np.array([[x+roi_x, y+roi_y], 
                                                            [x+w+roi_x, y+roi_y],
                                                            [x+w+roi_x, y+h+roi_y],
                                                            [x+roi_x, y+h+roi_y]])], 0, (255, 0, 0), 2)
                        
                        # Add rectangle and text label
                        cv2.rectangle(debug_img, (x+roi_x, y+roi_y), (x+w+roi_x, y+h+roi_y), (0, 255, 255), 2)
                        cv2.putText(debug_img, f"Card #{card_contours}", (x+roi_x, y+roi_y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Extract the card image
                        card_img = roi[y:y+h, x:x+w]
                        
                        # Get the rank and suit based on colors and patterns
                        rank, suit = self._identify_card(card_img)
                        
                        if rank and suit:
                            cards.append(Card(rank, suit))
                            logger.info(f"[CARD DEBUG] Detected player card: {rank}{suit}")
                            
                            # Display the rank and suit on the debug image
                            # Convert suit symbol to text representation for display
                            suit_text = suit
                            suit_color = (0, 0, 0)  # Default black
                            
                            if suit == 'h': 
                                suit_text = "♥"  # hearts
                                suit_color = (0, 0, 255)  # Red
                            elif suit == 'd': 
                                suit_text = "♦"  # diamonds
                                suit_color = (0, 0, 255)  # Red
                            elif suit == 'c': 
                                suit_text = "♣"  # clubs
                                suit_color = (0, 0, 0)  # Black
                            elif suit == 's': 
                                suit_text = "♠"  # spades
                                suit_color = (0, 0, 0)  # Black
                            
                            # Draw the card value with a bold, clearly visible font
                            card_text = f"{rank}{suit_text}"
                            text_x = x + roi_x + 5
                            text_y = y + roi_y + h + 20  # Position below the card
                            
                            # Draw background for better visibility
                            text_size = cv2.getTextSize(card_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(debug_img, 
                                         (text_x - 5, text_y - text_size[1] - 5), 
                                         (text_x + text_size[0] + 5, text_y + 5), 
                                         (255, 255, 255), -1)
                            
                            # Draw the text
                            cv2.putText(debug_img, card_text, (text_x, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, suit_color, 2)
                                      
                            # Draw a larger suit symbol for better visibility
                            symbol_x = x + roi_x + w//2
                            symbol_y = y + roi_y + h + 50
                            
                            # Draw large suit symbol
                            if suit == 'h':  # Heart
                                # Draw heart shape
                                pts = np.array([[symbol_x, symbol_y-15], 
                                              [symbol_x-10, symbol_y], 
                                              [symbol_x, symbol_y+15], 
                                              [symbol_x+10, symbol_y]])
                                cv2.fillPoly(debug_img, [pts], (0, 0, 255))
                            elif suit == 'd':  # Diamond
                                # Draw diamond shape
                                pts = np.array([[symbol_x, symbol_y-15], 
                                              [symbol_x-10, symbol_y], 
                                              [symbol_x, symbol_y+15], 
                                              [symbol_x+10, symbol_y]])
                                cv2.fillPoly(debug_img, [pts], (0, 0, 255))
                            elif suit == 'c':  # Club
                                # Draw club shape (simplified)
                                cv2.circle(debug_img, (symbol_x-5, symbol_y-5), 6, (0, 0, 0), -1)
                                cv2.circle(debug_img, (symbol_x+5, symbol_y-5), 6, (0, 0, 0), -1)
                                cv2.circle(debug_img, (symbol_x, symbol_y+2), 6, (0, 0, 0), -1)
                                cv2.rectangle(debug_img, (symbol_x-2, symbol_y), (symbol_x+2, symbol_y+12), (0, 0, 0), -1)
                            elif suit == 's':  # Spade
                                # Draw spade shape
                                pts = np.array([[symbol_x, symbol_y-15], 
                                              [symbol_x-10, symbol_y], 
                                              [symbol_x, symbol_y+5], 
                                              [symbol_x+10, symbol_y]])
                                cv2.fillPoly(debug_img, [pts], (0, 0, 0))
                                # Add stem
                                cv2.rectangle(debug_img, (symbol_x-2, symbol_y+5), (symbol_x+2, symbol_y+15), (0, 0, 0), -1)
                        else:
                            logger.info(f"[CARD DEBUG] Failed to identify card rank/suit")
                            # Display that the card couldn't be identified
                            cv2.putText(debug_img, "Unknown", (x+roi_x, y+roi_y+h+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            logger.info(f"[CARD DEBUG] Total card-like contours found: {card_contours}")
            
            # Always save the debug image in this modified version
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f'player_cards_debug_{int(time.time())}.png')
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"[CARD DEBUG] Saved debug image to {debug_path}")
                
            # Also save the white mask for debugging
            mask_path = os.path.join(debug_dir, f'player_cards_mask_{int(time.time())}.png')
            cv2.imwrite(mask_path, mask_white)
            logger.info(f"[CARD DEBUG] Saved card mask to {mask_path}")
                    
        except Exception as e:
            logger.exception(f"Error detecting player cards: {e}")
            
        return cards
        
    def _identify_card(self, card_img):
        """
        Identify the rank and suit of a card from its image using a more robust approach.
        
        Args:
            card_img (numpy.ndarray): Image of a single card.
            
        Returns:
            tuple: (rank, suit) of the card, or (None, None) if not identified.
        """
        try:
            # Resize the card image for more consistent processing
            h, w = card_img.shape[:2]
            resized = cv2.resize(card_img, (100, int(100 * h/w)))
            
            # Extract the top-left corner where the rank and suit are usually displayed
            # Increased corner_h from 40 to 50 to ensure we capture the full rank character
            corner_h, corner_w = min(50, resized.shape[0]//2), min(30, resized.shape[1]//2)
            corner = resized[0:corner_h, 0:corner_w]
            
            # Save corner image for debugging
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                corner_path = os.path.join(debug_dir, f'card_corner_{int(time.time())}.png')
                cv2.imwrite(corner_path, corner)
                logger.info(f"[CARD DEBUG] Saved card corner image to {corner_path}")
            
            # Enhanced color analysis for suits
            # Convert to multiple color spaces for better analysis
            hsv_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
            
            # Red detection (for hearts and diamonds) with improved thresholds
            # In HSV, red is at both ends of the hue spectrum
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100]) 
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv_corner, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_corner, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Black detection (for clubs and spades) with improved thresholds
            # In HSV, black has low V (value/brightness)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 100, 70])
            black_mask = cv2.inRange(hsv_corner, lower_black, upper_black)
            
            # Count red and black pixels
            red_pixels = cv2.countNonZero(red_mask)
            black_pixels = cv2.countNonZero(black_mask)
            
            # Save masks for debugging
            if self.debug_mode:
                cv2.imwrite(os.path.join(debug_dir, f'red_mask_{int(time.time())}.png'), red_mask)
                cv2.imwrite(os.path.join(debug_dir, f'black_mask_{int(time.time())}.png'), black_mask)
                logger.info(f"[CARD DEBUG] Red pixels: {red_pixels}, Black pixels: {black_pixels}")
            
            # Determine if card is red or black
            is_red = red_pixels > black_pixels and red_pixels > 10
            
            # Enhanced OCR for rank detection
            # Convert to grayscale with better contrast
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold for better text extraction
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Dilate to connect broken parts of characters
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            # Save thresholded image for debugging
            if self.debug_mode:
                cv2.imwrite(os.path.join(debug_dir, f'rank_thresh_{int(time.time())}.png'), thresh)
            
            # Use tesseract with specific configurations for card rank detection
            # Use --psm 8 for single character recognition and an improved whitelist
            rank_config = r'--psm 8 -c tessedit_char_whitelist=23456789TJQKA10'
            detected_text = pytesseract.image_to_string(thresh, config=rank_config).strip()
            
            # Improved mapping of OCR results to card ranks
            rank_map = {
                '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', 
                '8': '8', '9': '9', '1': '10', 'T': '10', 'J': 'J', 
                'Q': 'Q', 'K': 'K', 'A': 'A', 'l': '1', 'I': '1',
                'O': '10', 'o': '10', '0': '10', 'L': 'J', 'Z': '2',
                't': '10', 'i': '1', '!': '1', '[': 'J', ']': 'J',
                # Added common misidentifications
                'S': '5', 'B': '8', 'G': '6', 'g': '9',
                'U': 'J', 'V': 'A', 'Y': 'A'
            }
            
            # Process OCR text for rank
            rank = None
            logger.info(f"[CARD DEBUG] Raw OCR text: '{detected_text}'")
            
            # Clean up OCR results
            detected_text = ''.join(c for c in detected_text if c.isalnum())
            
            # If we found something like '10' directly
            if detected_text in ['10', '1O', 'IO', 'To', 'T0']:
                rank = '10'
            else:
                # Try to match the first character to a rank
                if detected_text and detected_text[0] in rank_map:
                    rank = rank_map[detected_text[0]]
                
                # Special case for 10
                if detected_text and len(detected_text) > 1:
                    if detected_text[:2] in ['10', '1O', 'IO', 'To', 'T0']:
                        rank = '10'
            
            # If OCR failed, try an alternative approach with shape analysis
            if not rank and corner_h > 10 and corner_w > 10:
                # Focus on the very top-left where rank is typically located
                rank_roi = corner[0:min(20, corner_h), 0:min(20, corner_w)]
                
                # Try multiple shape detection methods
                if self._check_for_A_shape(rank_roi):
                    rank = 'A'
                elif self._check_for_K_shape(rank_roi):
                    rank = 'K'
                elif self._check_for_Q_shape(rank_roi):
                    rank = 'Q'
                elif self._check_for_J_shape(rank_roi):
                    rank = 'J'
                elif self._check_for_9_shape(rank_roi):
                    rank = '9'
            
            # Use a pattern-based approach as a last resort
            if not rank:
                rank = self._rank_by_pattern_matching(thresh)
            
            # Determine suit based on shape analysis, not just color
            suit = None
            
            # For red cards (hearts and diamonds)
            if is_red:
                # Apply morphological operations to better detect shape features
                kernel = np.ones((2,2), np.uint8)
                red_processed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours in the red parts
                contours, _ = cv2.findContours(red_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Heart detection: hearts typically have a "V" shape at the bottom
                    heart_score = self._detect_heart_shape(contours, red_processed.shape)
                    
                    # Diamond detection: diamonds typically have sharp corners forming a rhombus
                    diamond_score = self._detect_diamond_shape(contours, red_processed.shape)
                    
                    logger.info(f"[CARD DEBUG] Heart score: {heart_score}, Diamond score: {diamond_score}")
                    
                    # Determine suit based on which score is higher
                    if heart_score > diamond_score:
                        suit = 'h'
                    else:
                        suit = 'd'
                else:
                    # If contour analysis fails, default to diamond as it's more common in online poker
                    suit = 'd'
            else:
                # For black cards (clubs and spades)
                kernel = np.ones((2,2), np.uint8)
                black_processed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(black_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Spade detection: spades typically have a triangular shape at the top
                    spade_score = self._detect_spade_shape(contours, black_processed.shape)
                    
                    # Club detection: clubs typically have three circular lobes
                    club_score = self._detect_club_shape(contours, black_processed.shape)
                    
                    logger.info(f"[CARD DEBUG] Spade score: {spade_score}, Club score: {club_score}")
                    
                    # Determine suit based on which score is higher
                    if spade_score > club_score:
                        suit = 's'
                    else:
                        suit = 'c'
                else:
                    # If contour analysis fails, default to spade as it's more common
                    suit = 's'
            
            # If we couldn't detect either rank or suit, return None
            if not rank or not suit:
                logger.info(f"[CARD DEBUG] Failed to identify card: rank={rank}, suit={suit}")
                return None, None
                
            logger.info(f"[CARD DEBUG] Successfully identified card: {rank}{suit}, is_red={is_red}")
            return rank, suit
            
        except Exception as e:
            logger.exception(f"Error identifying card: {e}")
            return None, None
            
    def _rank_by_pattern_matching(self, img):
        """Try to identify a card rank by its pattern of white pixels."""
        h, w = img.shape[:2]
        
        # Count white pixels in different regions
        top_left = cv2.countNonZero(img[0:h//3, 0:w//3])
        top_middle = cv2.countNonZero(img[0:h//3, w//3:2*w//3])
        top_right = cv2.countNonZero(img[0:h//3, 2*w//3:w])
        
        middle_left = cv2.countNonZero(img[h//3:2*h//3, 0:w//3])
        middle_middle = cv2.countNonZero(img[h//3:2*h//3, w//3:2*w//3])
        middle_right = cv2.countNonZero(img[h//3:2*h//3, 2*w//3:w])
        
        bottom_left = cv2.countNonZero(img[2*h//3:h, 0:w//3])
        bottom_middle = cv2.countNonZero(img[2*h//3:h, w//3:2*w//3])
        bottom_right = cv2.countNonZero(img[2*h//3:h, 2*w//3:w])
        
        # Very simplified pattern recognition based on the distribution of white pixels
        total_pixels = top_left + top_middle + top_right + middle_left + middle_middle + middle_right + bottom_left + bottom_middle + bottom_right
        if total_pixels < 10:  # Too few pixels to classify
            return None
            
        # A very rough estimation based on typical card rank shapes
        if top_middle > top_left and top_middle > top_right and bottom_middle > bottom_left and bottom_middle > bottom_right:
            # Central vertical line pattern suggests 1 or T
            return '10'
        elif top_left > top_right and bottom_right > bottom_left:
            # Diagonal pattern may suggest K
            return 'K'
        elif top_left > 0 and top_right > 0 and bottom_middle > 0:
            # U shape pattern suggests J
            return 'J'
        elif (top_left > 0 and top_right > 0 and 
              middle_left > 0 and middle_right > 0 and 
              bottom_left > 0 and bottom_right > 0):
            # O shape pattern suggests Q or 0
            return 'Q'
        elif top_middle > 0 and middle_middle > 0 and bottom_left > 0 and bottom_right > 0:
            # Top vertical with bottom horizontals suggests A
            return 'A'
        else:
            return None
            
    def _check_for_9_shape(self, img):
        """Check for '9' characteristic shape (circle with tail)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
            
        # 9 typically has a circular top part
        top_half = binary[0:h//2, :]
        contours, _ = cv2.findContours(top_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_circle = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5:  # Ignore tiny contours
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity > 0.4:  # Circle-like shape
                has_circle = True
                break
                
        # Check for tail in bottom right
        bottom_right = binary[h//2:h, w//2:w]
        bottom_right_pixels = cv2.countNonZero(bottom_right)
        
        # Characteristic of '9': circle top and tail at bottom
        if has_circle and bottom_right_pixels > 2:
            return True
            
        return False
        
    def _check_for_A_shape(self, img):
        """Check for 'A' characteristic shape (a peak with diverging lines)"""
        # Convert to binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Look for white pixels forming a triangular distribution
        h, w = binary.shape
        if h < 10 or w < 10:  # Too small to analyze
            return False
            
        # Check for a concentration of white pixels in the upper middle 
        # and diverging pattern toward bottom
        upper_mid = binary[0:h//2, w//4:3*w//4]
        upper_mid_pixels = cv2.countNonZero(upper_mid)
        
        bottom_left = binary[h//2:h, 0:w//2]
        bottom_left_pixels = cv2.countNonZero(bottom_left)
        
        bottom_right = binary[h//2:h, w//2:w]
        bottom_right_pixels = cv2.countNonZero(bottom_right)
        
        # Characteristic of 'A': strong presence in upper middle and both bottom corners
        if (upper_mid_pixels > 5 and
            bottom_left_pixels > 5 and
            bottom_right_pixels > 5):
            return True
        return False
        
    def _check_for_K_shape(self, img):
        """Check for 'K' characteristic shape (vertical line with diagonal branches)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
            
        # 'K' typically has a strong vertical line on the left
        left_col = binary[:, 0:w//4]
        left_pixels = cv2.countNonZero(left_col)
        
        # And diagonal elements from middle to right
        mid_right = binary[:, w//3:w]
        mid_right_pixels = cv2.countNonZero(mid_right)
        
        # Characteristic of 'K': strong left vertical and diagonal components
        if (left_pixels > h/2 and mid_right_pixels > 5):
            return True
        return False
        
    def _check_for_Q_shape(self, img):
        """Check for 'Q' characteristic shape (circular with a tail)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
        
        # 'Q' typically has a circular pattern in the top and middle
        top_and_middle = binary[0:3*h//4, :]
        
        # And a distinct diagonal tail specifically in the bottom right
        bottom_right = binary[2*h//3:h, 2*w//3:w]
        bottom_right_pixels = cv2.countNonZero(bottom_right)
        
        # Check middle left and right for O-like shape (should have pixels on both sides)
        middle_left = binary[h//4:3*h//4, 0:w//3]
        middle_right = binary[h//4:3*h//4, 2*w//3:w]
        middle_left_pixels = cv2.countNonZero(middle_left)
        middle_right_pixels = cv2.countNonZero(middle_right)
        
        # Q-specific: Should have a gap in the middle center (open circle) 
        # and pixels on both left and right sides
        has_side_pixels = middle_left_pixels > 3 and middle_right_pixels > 3
        
        # Check for circle-like contour in top and middle area
        contours, _ = cv2.findContours(top_and_middle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Check if contour is approximately circular
            area = cv2.contourArea(cnt)
            if area < 5:  # Ignore tiny contours
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 'Q' shape: circular top and middle, plus diagonal tail in bottom right
            # Must have side pixels on both left and right in middle region to be "O"-like
            if circularity > 0.5 and bottom_right_pixels > 2 and has_side_pixels:
                return True
                
        return False
        
    def _check_for_J_shape(self, img):
        """Check for 'J' characteristic shape (vertical with hook at bottom)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
            
        # 'J' typically has vertical component in the middle
        middle_col = binary[:, w//4:3*w//4]
        middle_pixels = cv2.countNonZero(middle_col)
        
        # And a hook at the bottom left
        bottom_left = binary[3*h//4:h, 0:w//2]
        bottom_left_pixels = cv2.countNonZero(bottom_left)
        
        # Characteristic of 'J': middle vertical and bottom left hook
        if (middle_pixels > h/3 and bottom_left_pixels > 3):
            return True
        return False
        
    def _detect_heart_shape(self, contours, shape):
        """Return a score indicating how heart-like the contours are"""
        score = 0
        h, w = shape
        
        # Heart characteristics: 
        # 1. Two circular bumps at top
        # 2. Pointed bottom
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Check for pointed bottom - safely extract y coordinates
                cnt_reshaped = cnt.reshape(-1, 2)  # Reshape to 2D array of [x,y] coordinates
                y_coords = cnt_reshaped[:, 1]      # Get all y coordinates
                bottom_y = np.max(y_coords)
                
                # Find points near the bottom
                bottom_points = cnt_reshaped[y_coords >= bottom_y - 2]
                
                # A pointed bottom will have relatively few points at the max y-coordinate
                if len(bottom_points) < 5:
                    score += 5
                
                # Check for two bumps at top using convexity defects
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                cnt_area = cv2.contourArea(cnt)
                
                # Hearts have concavities, so contour area is significantly less than hull area
                if cnt_area > 0 and hull_area / cnt_area > 1.2:
                    score += 10
            except Exception as e:
                logger.debug(f"Error in heart shape detection: {e}")
                
        return score
        
    def _detect_diamond_shape(self, contours, shape):
        """Return a score indicating how diamond-like the contours are"""
        score = 0
        h, w = shape
        
        # Diamond characteristics: 
        # 1. Four corners with similar angles
        # 2. Symmetrical shape
        # 3. Approximates a rhombus
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Check for polygon approximation with 4 points (square/diamond)
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) == 4:
                    score += 15  # Strong indicator of diamond
                
                # Check if width/height ratio is close to 1 (diamond is typically symmetric)
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 0 and h > 0 and 0.7 < w/h < 1.3:
                    score += 5
            except Exception as e:
                logger.debug(f"Error in diamond shape detection: {e}")
                
        return score
        
    def _detect_spade_shape(self, contours, shape):
        """Return a score indicating how spade-like the contours are"""
        score = 0
        h, w = shape
        
        # Spade characteristics:
        # 1. Triangular top
        # 2. Small stem at bottom
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Reshape contour to 2D array for easier indexing
                cnt_reshaped = cnt.reshape(-1, 2)
                
                # Get the topmost point - find the minimum y-coordinate
                y_coords = cnt_reshaped[:, 1]
                topmost_idx = np.argmin(y_coords)
                topmost = tuple(cnt_reshaped[topmost_idx])
                
                # Check for triangular top
                # Get points in the top half
                top_half_indices = np.where(cnt_reshaped[:, 1] < h//2)[0]
                if len(top_half_indices) >= 3:
                    top_half = cnt_reshaped[top_half_indices]
                    
                    # Try to fit a triangle to top half points
                    top_hull = cv2.convexHull(top_half.reshape(-1, 1, 2))
                    epsilon = 0.04 * cv2.arcLength(top_hull, True)
                    approx = cv2.approxPolyDP(top_hull, epsilon, True)
                    
                    if len(approx) == 3:
                        score += 10  # Strong indicator of spade
                        
                # Check for narrow stem at bottom
                # Get points in the bottom third
                bottom_indices = np.where(cnt_reshaped[:, 1] > 2*h//3)[0]
                if len(bottom_indices) > 0:
                    bottom_part = cnt_reshaped[bottom_indices]
                    
                    if len(bottom_part) > 0:
                        # Calculate width of bottom part
                        x_coords = bottom_part[:, 0]
                        bottom_width = np.max(x_coords) - np.min(x_coords)
                        
                        if bottom_width < w//2:
                            score += 5  # Narrow stem is typical for spades
            except Exception as e:
                logger.debug(f"Error in spade shape detection: {e}")
                    
        return score
        
    def _detect_club_shape(self, contours, shape):
        """Return a score indicating how club-like the contours are"""
        score = 0
        h, w = shape
        
        # Club characteristics:
        # 1. Multiple circular lobes (typically 3)
        # 2. Small stem at bottom
        
        # First, check if there are multiple distinct contours (club lobes)
        if len(contours) >= 2:
            score += 10  # Increase the weight for multiple contours (strong club indicator)
            
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Reshape contour to 2D array
                cnt_reshaped = cnt.reshape(-1, 2)
                
                # Check if contour is approximately circular (club lobes are circular)
                area = cv2.contourArea(cnt)
                if area < 5:  # Ignore tiny contours
                    continue
                    
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.6:
                    score += 15  # Increase weight for circular shapes (strong club indicator)
                    
                # Check for small stem at bottom
                bottom_indices = np.where(cnt_reshaped[:, 1] > 2*h//3)[0]
                if len(bottom_indices) > 0:
                    bottom_part = cnt_reshaped[bottom_indices]
                    
                    if len(bottom_part) > 0:
                        # Calculate width of bottom part
                        x_coords = bottom_part[:, 0]
                        bottom_width = np.max(x_coords) - np.min(x_coords)
                        
                        if bottom_width < w//3:
                            score += 5  # Narrow stem is typical for clubs
            except Exception as e:
                logger.debug(f"Error in club shape detection: {e}")
                    
        return score
        
    def _detect_community_cards(self, screenshot):
        """
        Detect community cards on the table using color analysis and shape detection.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            List[Card]: List of detected community cards.
        """
        cards = []
        try:
            # Common region for community cards (middle area of the screen)
            h, w = screenshot.shape[:2]
            
            # Define region of interest (ROI) where community cards are typically located
            roi_x = int(w * 0.25)  # Start at 30% from the left
            roi_y = int(h * 0.3)  # Start at 40% from the top
            roi_w = int(w * 0.4)  # Width is 40% of the screen width
            roi_h = int(h * 0.15)  # Height is 15% of the screen height
            
            # Add debugging info
            logger.info(f"[COMM CARD DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[COMM CARD DEBUG] Community card ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
            
            # Create a debug image for visualization
            debug_img = screenshot.copy()
            
            # Draw a rectangle around the ROI we're analyzing
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            # Draw crosshairs at the center of the ROI
            center_x = roi_x + roi_w // 2
            center_y = roi_y + roi_h // 2
            cv2.line(debug_img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(debug_img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
            
            # Draw coordinate text
            cv2.putText(debug_img, f"ROI: ({roi_x},{roi_y})", (roi_x, roi_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Convert to HSV color space for better color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define color range for white/gray card backgrounds
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
            
            # Find contours of potential cards
            contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Log number of contours found
            logger.info(f"[COMM CARD DEBUG] Found {len(contours)} potential community card contours")
            
            # Sort contours from left to right (as community cards are typically laid out)
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # Filter contours to find card-like shapes
            detected_cards = []
            card_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (cards should be within a reasonable size range)
                min_card_area = (roi_w * roi_h) * 0.02  # Cards take at least 2% of ROI
                max_card_area = (roi_w * roi_h) * 0.15  # Cards take at most 15% of ROI
                
                logger.info(f"[COMM CARD DEBUG] Contour area: {area}, min: {min_card_area}, max: {max_card_area}")
                
                if min_card_area < area < max_card_area:
                    # Get bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if aspect ratio is card-like (height:width ratio ~1.4:1)
                    aspect_ratio = h / w
                    logger.info(f"[COMM CARD DEBUG] Contour aspect ratio: {aspect_ratio}")
                    
                    if 1.2 < aspect_ratio < 1.6:
                        card_contours += 1
                        
                        # Draw the contour in the debug image
                        cv2.drawContours(debug_img, [np.array([[x+roi_x, y+roi_y], 
                                                            [x+w+roi_x, y+roi_y],
                                                            [x+w+roi_x, y+h+roi_y],
                                                            [x+roi_x, y+h+roi_y]])], 0, (255, 0, 0), 2)
                        
                        # Add rectangle and text label
                        cv2.rectangle(debug_img, (x+roi_x, y+roi_y), (x+w+roi_x, y+h+roi_y), (0, 255, 255), 2)
                        cv2.putText(debug_img, f"Card #{card_contours}", (x+roi_x, y+roi_y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Extract the card image
                        card_img = roi[y:y+h, x:x+w]
                        detected_cards.append((x, y, w, h, card_img))
            
            logger.info(f"[COMM CARD DEBUG] Total card-like contours found: {card_contours}")
            
            # Process detected cards (maximum of 5 for community cards)
            processed_count = 0
            for x, y, w, h, card_img in detected_cards[:5]:
                # Get the rank and suit
                rank, suit = self._identify_card(card_img)
                
                if rank and suit:
                    cards.append(Card(rank, suit))
                    processed_count += 1
                    logger.info(f"[COMM CARD DEBUG] Detected community card: {rank}{suit}")
                    
                    # Display the rank and suit on the debug image
                    # Convert suit symbol to text representation for display
                    suit_text = suit
                    if suit == 'h': suit_text = "♥"  # hearts
                    elif suit == 'd': suit_text = "♦"  # diamonds
                    elif suit == 'c': suit_text = "♣"  # clubs
                    elif suit == 's': suit_text = "♠"  # spades
                    
                    # Draw rank and suit text at a position below the card rectangle
                    card_text = f"{rank}{suit_text}"
                    text_x = x + roi_x + 5
                    text_y = y + roi_y + h + 20  # Position below the card
                    
                    # Draw the card value with a bold, clearly visible font
                    # First draw a black background for better visibility
                    cv2.putText(debug_img, card_text, (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                    
                    # Then overlay the text with color based on suit
                    if suit in ['h', 'd']:  # Red for hearts and diamonds
                        cv2.putText(debug_img, card_text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:  # Black for clubs and spades
                        cv2.putText(debug_img, card_text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    logger.info(f"[COMM CARD DEBUG] Failed to identify card rank/suit at position {x},{y}")
                    # Display that the card couldn't be identified
                    cv2.putText(debug_img, "Unknown", (x+roi_x, y+roi_y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            if not cards:
                logger.debug("No community cards detected")
            else:
                logger.info(f"[COMM CARD DEBUG] Successfully detected {len(cards)} community cards")
            
            # Always save the debug image - this helps with troubleshooting
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f'community_cards_debug_{int(time.time())}.png')
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"[COMM CARD DEBUG] Saved debug image to {debug_path}")
            
            # Also save the white mask for debugging
            mask_path = os.path.join(debug_dir, f'community_cards_mask_{int(time.time())}.png')
            cv2.imwrite(mask_path, mask_white)
            logger.info(f"[COMM CARD DEBUG] Saved card mask to {mask_path}")
            
        except Exception as e:
            logger.exception(f"Error detecting community cards: {e}")
            
        return cards
        
    def _detect_pot_size(self, screenshot):
        """
        Detect the current pot size from the screenshot using OCR.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            float: The detected pot size.
        """
        try:
            h, w = screenshot.shape[:2]
            
            # Define region where pot size is typically displayed (center-top of the table)
            roi_x = int(w * 0.4)
            roi_y = int(h * 0.3)
            roi_w = int(w * 0.2)
            roi_h = int(h * 0.08)
            
            # Add debugging info
            logger.info(f"[POT DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[POT DEBUG] Pot size ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
            
            # Create a debug image for visualization
            debug_img = screenshot.copy()
            
            # Draw a rectangle around the ROI we're analyzing
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
            # Draw crosshairs at the center of the ROI
            center_x = roi_x + roi_w // 2
            center_y = roi_y + roi_h // 2
            cv2.line(debug_img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(debug_img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
            
            # Draw coordinate text
            cv2.putText(debug_img, f"ROI: ({roi_x},{roi_y})", (roi_x, roi_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Preprocess the image for better OCR
            preprocessed = self._preprocess_for_ocr(roi)
            
            # Use OCR to extract text
            text = pytesseract.image_to_string(
                preprocessed,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,$'
            )
            
            logger.info(f"[POT DEBUG] Raw OCR text: '{text}'")
            
            # Clean and parse the text
            pot_size = self._parse_money_value(text)
            
            # Save both the original ROI and the preprocessed version
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            original_path = os.path.join(debug_dir, f'pot_roi_{timestamp}.png')
            preprocessed_path = os.path.join(debug_dir, f'pot_preprocessed_{timestamp}.png')
            debug_path = os.path.join(debug_dir, f'pot_debug_{timestamp}.png')
            
            cv2.imwrite(original_path, roi)
            cv2.imwrite(preprocessed_path, preprocessed)
            logger.info(f"[POT DEBUG] Saved original ROI to {original_path}")
            logger.info(f"[POT DEBUG] Saved preprocessed image to {preprocessed_path}")
            
            # Add OCR results to the debug image
            cv2.rectangle(debug_img, (10, 10), (350, 80), (0, 0, 0), -1)  # Black background for text
            cv2.putText(debug_img, f"OCR Text: '{text}'", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(debug_img, f"Parsed Value: ${pot_size:.2f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Save the full debug image
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"[POT DEBUG] Saved debug image to {debug_path}")
            
            if pot_size > 0:
                logger.info(f"[POT DEBUG] Detected pot size: ${pot_size:.2f}")
                return pot_size
            else:
                logger.info("[POT DEBUG] No pot size detected or pot size is zero")
                return 0.0
                
        except Exception as e:
            logger.exception(f"Error detecting pot size: {e}")
            return 0.0
            
    def _parse_money_value(self, text):
        """
        Parse a text string to extract a monetary value.
        
        Args:
            text (str): Text to parse.
            
        Returns:
            float: Extracted monetary value.
        """
        if not text:
            return 0.0
            
        # Remove non-numeric characters except decimal point
        # First, check if there's a specific pattern like "Pot: $123.45"
        import re
        
        # Look for patterns like "pot: $123.45" or "$123.45"
        pot_pattern = re.search(r'(?:pot:?\s*)?[$]?(\d+(?:\.\d+)?)', text.lower())
        if pot_pattern:
            try:
                return float(pot_pattern.group(1))
            except ValueError:
                pass
        
        # If no pattern matched, try to extract any number
        digits_only = ''.join(c for c in text if c.isdigit() or c == '.')
        
        # Handle multiple decimal points
        parts = digits_only.split('.')
        if len(parts) > 2:
            # Keep only the first decimal point
            digits_only = parts[0] + '.' + ''.join(parts[1:]).replace('.', '')
        
        try:
            return float(digits_only) if digits_only else 0.0
        except ValueError:
            return 0.0
            
    def _detect_current_bet(self, screenshot):
        """
        Detect the current bet amount from the screenshot using OCR.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            float: The detected current bet amount.
        """
        try:
            h, w = screenshot.shape[:2]
            
            # Define region where the current bet is typically displayed
            # Usually in the center-bottom area of the table
            roi_x = int(w * 0.4)
            roi_y = int(h * 0.55)
            roi_w = int(w * 0.2)
            roi_h = int(h * 0.05)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Preprocess the image for better OCR
            preprocessed = self._preprocess_for_ocr(roi)
            
            # Use OCR to extract text
            text = pytesseract.image_to_string(
                preprocessed,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,$'
            )
            
            # Parse the bet amount
            bet_amount = self._parse_money_value(text)
            
            if bet_amount > 0:
                logger.info(f"Detected current bet: ${bet_amount:.2f}")
                
                # Save debug image
                if self.debug_mode and time.time() - self.last_debug_time > 30:
                    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    debug_path = os.path.join(debug_dir, f'current_bet_{int(time.time())}.png')
                    cv2.imwrite(debug_path, roi)
                    logger.info(f"Saved current bet debug image to {debug_path}")
                
                return bet_amount
            else:
                logger.debug("No current bet detected or bet is zero")
                return 0.0
                
        except Exception as e:
            logger.exception(f"Error detecting current bet: {e}")
            return 0.0
        
    def _detect_player_stack(self, screenshot):
        """
        Detect the player's chip stack from the screenshot using OCR.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            float: The detected player stack amount.
        """
        try:
            h, w = screenshot.shape[:2]
            
            # Define region where player stack is typically displayed
            # Usually near the bottom of the screen, in front of the player
            roi_x = int(w * 0.45)
            roi_y = int(h * 0.8)
            roi_w = int(w * 0.1)
            roi_h = int(h * 0.05)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Preprocess the image for better OCR
            preprocessed = self._preprocess_for_ocr(roi)
            
            # Use OCR to extract text
            text = pytesseract.image_to_string(
                preprocessed,
                config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,$'
            )
            
            # Parse the stack amount
            stack_amount = self._parse_money_value(text)
            
            # If no stack was detected or the value is unreasonably small,
            # use a default value (this should be configured)
            if stack_amount <= 0:
                logger.debug("Could not detect player stack, using default value")
                return self.config.get('default_stack', 100.0)
                
            logger.info(f"Detected player stack: ${stack_amount:.2f}")
            
            # Save debug image
            if self.debug_mode and time.time() - self.last_debug_time > 30:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                debug_path = os.path.join(debug_dir, f'player_stack_{int(time.time())}.png')
                cv2.imwrite(debug_path, roi)
                logger.info(f"Saved player stack debug image to {debug_path}")
                
            return stack_amount
                
        except Exception as e:
            logger.exception(f"Error detecting player stack: {e}")
            return self.config.get('default_stack', 100.0)
        
    def _detect_position(self, screenshot):
        """
        Detect the player's position based on dealer button location.
        
        Args:
            screenshot (numpy.ndarray): The screenshot to analyze.
            
        Returns:
            str: The player's position ('early', 'middle', 'late', 'sb', 'bb', 'dealer').
        """
        try:
            h, w = screenshot.shape[:2]
            
            # Define region where the dealer button could be located
            # This typically spans the whole table area
            roi_x = int(w * 0.2)
            roi_y = int(h * 0.3)
            roi_w = int(w * 0.6)
            roi_h = int(h * 0.4)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Look for dealer button (typically white/gray or bright colored circular object)
            # Define color ranges for common dealer button colors
            # White/gray button
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            
            # Yellow button (some clients use yellow)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours for potential dealer buttons
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for circular/oval shapes that could be the dealer button
            dealer_button_x = None
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (dealer button is typically small)
                min_area = (roi_w * roi_h) * 0.001  # At least 0.1% of ROI
                max_area = (roi_w * roi_h) * 0.01   # At most 1% of ROI
                
                if min_area < area < max_area:
                    # Check if the shape is approximately circular
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Circles have circularity close to 1.0
                    if circularity > 0.7:  # Allow some tolerance for oval shapes
                        # Get the center of the contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            
                            # Update the dealer button x-coordinate (using leftmost if multiple detected)
                            if dealer_button_x is None or cx < dealer_button_x:
                                dealer_button_x = cx
                                
                                # Save debug image
                                if self.debug_mode and time.time() - self.last_debug_time > 30:
                                    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                                    os.makedirs(debug_dir, exist_ok=True)
                                    
                                    debug_img = roi.copy()
                                    cv2.drawContours(debug_img, [contour], 0, (0, 255, 0), 2)
                                    
                                    debug_path = os.path.join(debug_dir, f'dealer_button_{int(time.time())}.png')
                                    cv2.imwrite(debug_path, debug_img)
                                    logger.info(f"Saved dealer button debug image to {debug_path}")
            
            if dealer_button_x is None:
                logger.debug("Could not detect dealer button")
                return "unknown"
                
            # Determine player position based on the dealer button location
            # This is a simplified approach and assumes 6-max table
            # For more accurate results, we'd need to detect all players and their relative positions
            
            # Divide the table into regions (left to right)
            third_width = roi_w / 3
            
            if dealer_button_x < third_width:  # Left third of the table
                # If dealer is on the left, we're in late position
                position = "late"
                logger.info(f"Detected position: {position} (dealer on left)")
            elif dealer_button_x < 2 * third_width:  # Middle third
                position = "middle"
                logger.info(f"Detected position: {position} (dealer in middle)")
            else:  # Right third
                # If dealer is on the right, we're in early position
                position = "early"
                logger.info(f"Detected position: {position} (dealer on right)")
                
            return position
            
        except Exception as e:
            logger.exception(f"Error detecting player position: {e}")
            return "unknown"
        
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
            roi_x = int(w * 0.3)
            roi_y = int(h * 0.8)
            roi_w = int(w * 0.4)
            roi_h = int(h * 0.15)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # MODIFIED: Focus only on blue buttons since all buttons in the client are blue
            # Use a wider range for blue to ensure detection
            lower_blue = np.array([90, 40, 40])  # Slightly expanded range
            upper_blue = np.array([150, 255, 255])  # Slightly expanded range
            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            
            # Use only blue mask for buttons
            combined_mask = blue_mask
            
            # Find contours of potential buttons
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find button-like shapes
            active_buttons = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (buttons should be within a reasonable size range)
                min_button_area = (roi_w * roi_h) * 0.01  # At least 1% of ROI
                max_button_area = (roi_w * roi_h) * 0.15  # At most 15% of ROI
                
                if min_button_area < area < max_button_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (buttons are typically wider than tall)
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 5:
                        active_buttons.append((x, y, w, h))
            
            # If we found button-like shapes with active colors, it's likely our turn
            is_our_turn = len(active_buttons) > 0
            
            # Save debug image
            if self.debug_mode and (is_our_turn or time.time() - self.last_debug_time > 30):
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                debug_img = roi.copy()
                for x, y, w, h in active_buttons:
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                debug_path = os.path.join(debug_dir, f'turn_detection_{int(time.time())}.png')
                cv2.imwrite(debug_path, debug_img)
                logger.info(f"Saved turn detection debug image to {debug_path}")
            
            if is_our_turn:
                logger.info("It's our turn to act")
            else:
                logger.debug("It's not our turn to act")
                
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
            # If it's not our turn, no actions are available
            if not self._detect_is_our_turn(screenshot):
                return []
                
            h, w = screenshot.shape[:2]
            
            # Define the region where action buttons are typically located
            roi_x = int(w * 0.3)
            roi_y = int(h * 0.8)
            roi_w = int(w * 0.4)
            roi_h = int(h * 0.15)
            
            # Extract the region of interest
            roi = screenshot[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # MODIFIED: Use only blue mask for all buttons since all buttons in the client are blue
            lower_blue = np.array([90, 40, 40])  # Slightly expanded range
            upper_blue = np.array([150, 255, 255])  # Slightly expanded range
            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            
            # Define masks for button regions (left, middle, right)
            third_width = roi_w // 3
            
            left_mask = np.zeros_like(blue_mask)
            left_mask[:, :third_width] = 255
            
            middle_mask = np.zeros_like(blue_mask)
            middle_mask[:, third_width:2*third_width] = 255
            
            right_mask = np.zeros_like(blue_mask)
            right_mask[:, 2*third_width:] = 255
            
            # Find active buttons in each region using only blue color
            # Left button (typically fold)
            left_blue = cv2.bitwise_and(blue_mask, left_mask)
            if cv2.countNonZero(left_blue) > 50:
                available_actions.append("fold")
            
            # Middle button (typically check/call)
            middle_blue = cv2.bitwise_and(blue_mask, middle_mask)
            if cv2.countNonZero(middle_blue) > 50:
                # Try to determine if it's check or call using OCR
                # Crop the middle section
                middle_roi = roi[:, third_width:2*third_width]
                gray = cv2.cvtColor(middle_roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                # Use OCR to read the text
                middle_text = pytesseract.image_to_string(binary).lower()
                
                if 'check' in middle_text:
                    available_actions.append("check")
                elif 'call' in middle_text:
                    available_actions.append("call")
                else:
                    # If we can't determine, add both possibilities
                    # The decision maker can handle this ambiguity
                    available_actions.append("check")
                    available_actions.append("call")
            
            # Right button (typically bet/raise)
            right_blue = cv2.bitwise_and(blue_mask, right_mask)
            if cv2.countNonZero(right_blue) > 50:
                # Try to determine if it's bet or raise
                right_roi = roi[:, 2*third_width:]
                gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                # Use OCR to read text
                right_text = pytesseract.image_to_string(binary).lower()
                
                if 'bet' in right_text:
                    available_actions.append("bet")
                elif 'raise' in right_text:
                    available_actions.append("raise")
                else:
                    # If we can't determine, add both possibilities
                    available_actions.append("bet")
                    available_actions.append("raise")
            
            # If no specific actions were detected but it's our turn,
            # include default actions as a fallback
            if not available_actions and self._detect_is_our_turn(screenshot):
                available_actions = ["fold", "check", "bet"]
            
            logger.info(f"Detected available actions: {available_actions}")
            
            # Save debug image
            if self.debug_mode and available_actions and time.time() - self.last_debug_time > 30:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                debug_img = roi.copy()
                
                # Visualize button regions
                cv2.line(debug_img, (third_width, 0), (third_width, roi_h), (0, 255, 255), 2)
                cv2.line(debug_img, (2*third_width, 0), (2*third_width, roi_h), (0, 255, 255), 2)
                
                # Add text for detected actions
                y = 20
                for action in available_actions:
                    cv2.putText(debug_img, action, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 20
                
                debug_path = os.path.join(debug_dir, f'available_actions_{int(time.time())}.png')
                cv2.imwrite(debug_path, debug_img)
                logger.info(f"Saved available actions debug image to {debug_path}")
                
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