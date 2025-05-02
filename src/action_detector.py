#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting available poker actions.
"""

import logging
import cv2
import numpy as np
import pytesseract
import os
import time

logger = logging.getLogger(__name__)

class ActionDetector:
    """Class for detecting available poker actions."""
    
    def __init__(self, config=None):
        """Initialize the action detector with optional configuration."""
        self.config = config or {}
        # Enable debug mode unless concise_logging is set to True in config
        self.debug_mode = not self.config.get('concise_logging', False)
        
        # Configure pytesseract path if not already set in the environment
        if not hasattr(pytesseract.pytesseract, 'tesseract_cmd') or not pytesseract.pytesseract.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def detect_available_actions(self, screenshot):
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
                # Import here to avoid circular imports
                from turn_detector import TurnDetector
                turn_detector = TurnDetector(self.config)
                is_our_turn = turn_detector.detect_is_our_turn(screenshot)
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