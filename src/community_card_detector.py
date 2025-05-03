#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting community cards on a poker table.
"""

import logging
import cv2
import numpy as np
import os
import time
from typing import List

from models import Card

logger = logging.getLogger(__name__)

class CommunityCardDetector:
    """Class for detecting community cards on the poker table."""
    
    def __init__(self, card_identifier, debug_mode=False):
        """
        Initialize the community card detector.
        
        Args:
            card_identifier: CardIdentifier instance to identify cards
            debug_mode: Whether to save debug images
        """
        self.card_identifier = card_identifier
        self.debug_mode = debug_mode
        
    def detect_community_cards(self, screenshot):
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
            # For bærbar
            # roi_x = int(w * 0.25)  # Start at 30% from the left
            # roi_y = int(h * 0.3)  # Start at 40% from the top
            # roi_w = int(w * 0.4)  # Width is 40% of the screen width
            # roi_h = int(h * 0.15)  # Height is 15% of the screen height

            # For stasjonær
            roi_x = int(w * 0.15)  # Start at 30% from the left
            roi_y = int(h * 0.4)  # Start at 40% from the top
            roi_w = int(w * 0.2)  # Width is 40% of the screen width
            roi_h = int(h * 0.10)  # Height is 15% of the screen height
            
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
                rank, suit = self.card_identifier.identify_card(card_img)
                
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