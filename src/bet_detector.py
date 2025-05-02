#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting current bet from poker screenshots.
"""

import logging
import cv2
import numpy as np
import pytesseract
import os
import time
import re

logger = logging.getLogger(__name__)

# Configure pytesseract path - update this with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class BetDetector:
    """Class for detecting current bet from poker screenshots."""
    
    def __init__(self, config=None, debug_mode=False):
        """Initialize the bet detector with optional configuration."""
        self.config = config or {}
        self.debug_mode = debug_mode
        logger.info("Bet detector initialized")
    
    def detect_current_bet(self, screenshot):
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
            # Stasjonær
            roi_x = int(w * 0.22)  # Start at 22% from the left
            roi_y = int(h * 0.55)  # Start at 55% from the top
            roi_w = int(w * 0.06)  # Width is 6% of the screen width
            roi_h = int(h * 0.05)  # Height is 5% of the screen height
            
            # Add debugging info
            logger.info(f"[BET DEBUG] Screenshot size: {w}x{h}")
            logger.info(f"[BET DEBUG] Current bet ROI: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")
            
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
            
            logger.info(f"[BET DEBUG] Raw OCR text: '{text}'")
            
            # Parse the bet amount
            bet_amount = self._parse_money_value(text)
            
            # Always save debug images to track detection quality
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            original_path = os.path.join(debug_dir, f'current_bet_roi_{timestamp}.png')
            preprocessed_path = os.path.join(debug_dir, f'current_bet_preprocessed_{timestamp}.png')
            debug_path = os.path.join(debug_dir, f'current_bet_debug_{timestamp}.png')
            
            cv2.imwrite(original_path, roi)
            cv2.imwrite(preprocessed_path, preprocessed)
            
            # Add OCR results to the debug image
            cv2.rectangle(debug_img, (10, 10), (350, 80), (0, 0, 0), -1)  # Black background for text
            cv2.putText(debug_img, f"OCR Text: '{text}'", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(debug_img, f"Parsed Value: ${bet_amount:.2f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Save the full debug image
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"[BET DEBUG] Saved debug images to {debug_dir}")
            
            if bet_amount > 0:
                logger.info(f"[BET DEBUG] Detected current bet: ${bet_amount:.2f}")
                return bet_amount
            else:
                logger.debug("[BET DEBUG] No current bet detected or bet is zero")
                return 0.0
                
        except Exception as e:
            logger.exception(f"Error detecting current bet: {e}")
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
            
        # Look for patterns like "$123.45"
        money_pattern = re.search(r'[$]?(\d+(?:\.\d+)?)', text.lower())
        if money_pattern:
            try:
                return float(money_pattern.group(1))
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
    
    def _preprocess_for_ocr(self, img):
        """Preprocess image for better OCR results."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh