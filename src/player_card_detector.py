#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting player cards from screenshots in poker games.
"""

import logging
import cv2
import numpy as np
import os
import time
from typing import List
from game_state import Card

logger = logging.getLogger(__name__)


class PlayerCardDetector:
    """Class for detecting player's hole cards from screenshots."""
    
    def __init__(self, config=None):
        """Initialize the player card detector with optional configuration."""
        self.config = config or {}
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)
        self.last_debug_time = 0  # To limit debug image saving frequency
    
    def set_card_identifier(self, card_identifier):
        """Set the card identifier instance to use for card recognition."""
        self.card_identifier = card_identifier
        
    def detect_player_cards(self, screenshot):
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
            # For bærbar
            # roi_x = int(w * 0.3)  # Start at 40% from the left
            # roi_y = int(h * 0.6)  # Start at 60% from the top
            # roi_w = int(w * 0.2)  # Width is 20% of the screen width
            # roi_h = int(h * 0.10)  # Height is 15% of the screen height

            #For stasjonær
            roi_x = int(w * 0.22)  # Start at 40% from the left
            roi_y = int(h * 0.6)  # Start at 60% from the top
            roi_w = int(w * 0.07)  # Width is 20% of the screen width
            roi_h = int(h * 0.08)  # Height is 15% of the screen height

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
                        rank, suit = self.card_identifier.identify_card(card_img)
                        
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