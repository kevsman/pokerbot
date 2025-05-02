#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting active poker games from screenshots.
"""

import logging
import cv2
import os
import time
import numpy as np

# Add import for window detection
try:
    import pygetwindow as gw
    WINDOW_DETECTION_AVAILABLE = True
except ImportError:
    WINDOW_DETECTION_AVAILABLE = False
    print("pygetwindow not installed. Window detection will not be available.")
    print("To enable window detection, install with: pip install pygetwindow")

logger = logging.getLogger(__name__)

# List of possible poker client window titles
POKER_CLIENT_TITLES = [
    "Poker", "PokerStars", "GGPoker", "888Poker", "PartyPoker",
    "Winamax", "Unibet", "ACR Poker", "Ignition Poker", "Bovada"
]


class GameDetector:
    """Class for detecting active poker games from screenshots."""
    
    def __init__(self, config=None):
        """Initialize the game detector with optional configuration."""
        self.config = config or {}
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
        
        logger.info("Game detector initialized")
        if self.use_window_detection:
            logger.info("Window title detection is enabled")
        if self.debug_mode:
            logger.info("Debug mode is enabled - will save debug images")
        else:
            logger.info("Debug mode is disabled - will not save debug images")
            
    def load_templates(self):
        """Load client templates from resources directory."""
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
            
        except Exception as e:
            logger.exception(f"Error loading templates: {e}")
            
    def detect_active_game(self, screenshot):
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