#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for detecting pot size from poker screenshots.
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

class PotDetector:
    """Class for detecting pot size from poker screenshots."""
    
    def __init__(self, config=None, debug_mode=False):
        """Initialize the pot detector with optional configuration."""
        self.config = config or {}
        self.debug_mode = debug_mode
        logger.info("Pot detector initialized")
    
    def detect_pot_size(self, screenshot):
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
            #For stasjonær
            roi_x = int(w * 0.22)
            roi_y = int(h * 0.36)
            roi_w = int(w * 0.07)
            roi_h = int(h * 0.05)
            
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
    
    def _preprocess_for_ocr(self, img):
        """Preprocess image for better OCR results."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh