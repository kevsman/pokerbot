#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting when it's the player's turn in poker games.
"""

import logging
import cv2
import numpy as np
import os
import time

logger = logging.getLogger(__name__)

class TurnDetector:
    """Class for detecting when it's the player's turn to act in poker games."""
    
    def __init__(self, config=None):
        """Initialize the turn detector with optional configuration."""
        self.config = config or {}
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)

    def detect_is_our_turn(self, screenshot):
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