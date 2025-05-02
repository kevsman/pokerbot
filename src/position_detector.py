#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting poker player position based on dealer button location.
"""

import logging
import cv2
import numpy as np
import os
import time

logger = logging.getLogger(__name__)

class PositionDetector:
    """Class for detecting player position in poker games."""
    
    def __init__(self, config=None):
        """Initialize the position detector with optional configuration."""
        self.config = config or {}
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)

    def detect_position(self, screenshot):
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
            #Bærbar
            # roi_x = int(w * 0.2)
            # roi_y = int(h * 0.3)
            # roi_w = int(w * 0.6)
            # roi_h = int(h * 0.4)

            #Stasjonær
            roi_x = int(w * 0.1)
            roi_y = int(h * 0.3)
            roi_w = int(w * 0.4)
            roi_h = int(h * 0.4)
            
            # Add debugging info
            logger.info(f"[POSITION DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[POSITION DEBUG] Position/dealer button ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
            
            # Create a debug image for visualization
            debug_img = screenshot.copy()
            
            # Draw a rectangle around the ROI we're analyzing
            cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
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
            dealer_button_contour = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size (dealer button is typically small)
                min_area = (roi_w * roi_h) * 0.001  # At least 0.1% of ROI
                max_area = (roi_w * roi_h) * 0.01   # At most 1% of ROI
                
                logger.info(f"[POSITION DEBUG] Contour area: {area}, min: {min_area}, max: {max_area}")
                
                if min_area < area < max_area:
                    # Check if the shape is approximately circular
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    logger.info(f"[POSITION DEBUG] Contour circularity: {circularity}")
                    
                    # Circles have circularity close to 1.0
                    if circularity > 0.7:  # Allow some tolerance for oval shapes
                        # Get the center of the contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Update the dealer button x-coordinate (using leftmost if multiple detected)
                            if dealer_button_x is None or cx < dealer_button_x:
                                dealer_button_x = cx
                                dealer_button_contour = contour
                                
                                # Draw the detected dealer button contour
                                cv2.drawContours(debug_img, [np.array([[x+roi_x, y+roi_y] for x, y in contour.reshape(-1, 2)])], 0, (0, 0, 255), 2)
                                
                                # Draw crosshair at button center
                                button_center_x = cx + roi_x
                                button_center_y = cy + roi_y
                                cv2.line(debug_img, (button_center_x - 10, button_center_y), (button_center_x + 10, button_center_y), (255, 0, 0), 2)
                                cv2.line(debug_img, (button_center_x, button_center_y - 10), (button_center_x, button_center_y + 10), (255, 0, 0), 2)
                                
                                # Label the dealer button
                                cv2.putText(debug_img, "Dealer", (button_center_x + 15, button_center_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Always save debug images to track detection quality
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            original_path = os.path.join(debug_dir, f'position_roi_{timestamp}.png')
            mask_path = os.path.join(debug_dir, f'position_mask_{timestamp}.png')
            debug_path = os.path.join(debug_dir, f'position_debug_{timestamp}.png')
            
            cv2.imwrite(original_path, roi)
            cv2.imwrite(mask_path, combined_mask)
            
            if dealer_button_x is None:
                logger.debug("[POSITION DEBUG] Could not detect dealer button")
                cv2.putText(debug_img, "No dealer button detected", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                position = "unknown"
            else:
                # Determine player position based on the dealer button location
                # This is a simplified approach and assumes 6-max table
                
                # Draw dividing lines showing position regions
                third_width = roi_w / 3
                cv2.line(debug_img, (int(roi_x + third_width), roi_y), 
                         (int(roi_x + third_width), roi_y + roi_h), (255, 255, 0), 1)
                cv2.line(debug_img, (int(roi_x + 2 * third_width), roi_y), 
                         (int(roi_x + 2 * third_width), roi_y + roi_h), (255, 255, 0), 1)
                
                # Label the regions
                cv2.putText(debug_img, "LATE", (roi_x + 10, roi_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(debug_img, "MIDDLE", (int(roi_x + third_width + 10), roi_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(debug_img, "EARLY", (int(roi_x + 2 * third_width + 10), roi_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if dealer_button_x < third_width:  # Left third of the table
                    # If dealer is on the left, we're in late position
                    position = "late"
                    logger.info(f"[POSITION DEBUG] Detected position: {position} (dealer on left)")
                elif dealer_button_x < 2 * third_width:  # Middle third
                    position = "middle"
                    logger.info(f"[POSITION DEBUG] Detected position: {position} (dealer in middle)")
                else:  # Right third
                    # If dealer is on the right, we're in early position
                    position = "early"
                    logger.info(f"[POSITION DEBUG] Detected position: {position} (dealer on right)")
                
                # Add position text to debug image
                cv2.rectangle(debug_img, (10, 10), (200, 50), (0, 0, 0), -1)  # Black background for text
                cv2.putText(debug_img, f"Position: {position.upper()}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Save the full debug image
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"[POSITION DEBUG] Saved debug images to {debug_dir}")
            
            return position
            
        except Exception as e:
            logger.exception(f"Error detecting player position: {e}")
            return "unknown"