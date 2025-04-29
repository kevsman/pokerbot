#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for capturing and processing screenshots of the poker game.
"""

import logging
import cv2
import numpy as np
import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Class for capturing and processing screenshots of the poker game."""
    
    def __init__(self, region=None):
        """
        Initialize the screen capture with optional region.
        
        Args:
            region (tuple, optional): Region to capture (left, top, width, height).
                                     If None, captures the entire screen.
        """
        self.region = region
        logger.info("Screen capture initialized")
        
    def capture(self):
        """
        Capture a screenshot of the specified region or entire screen.
        
        Returns:
            numpy.ndarray: The captured screenshot as a numpy array in BGR format.
        """
        try:
            if self.region:
                screenshot = pyautogui.screenshot(region=self.region)
            else:
                screenshot = pyautogui.screenshot()
                
            # Convert PIL Image to OpenCV format (numpy array, BGR)
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            logger.debug("Screenshot captured successfully")
            return screenshot
            
        except Exception as e:
            logger.exception(f"Error capturing screenshot: {e}")
            return None
            
    def save_screenshot(self, screenshot, filename="screenshot.png"):
        """
        Save the screenshot to a file.
        
        Args:
            screenshot (numpy.ndarray): Screenshot to save.
            filename (str): Filename to save the screenshot to.
        """
        try:
            cv2.imwrite(filename, screenshot)
            logger.debug(f"Screenshot saved to {filename}")
        except Exception as e:
            logger.exception(f"Error saving screenshot: {e}")
            
    def find_template(self, screenshot, template_path, threshold=0.8):
        """
        Find a template image within the screenshot using template matching.
        
        Args:
            screenshot (numpy.ndarray): Screenshot to search in.
            template_path (str): Path to the template image.
            threshold (float): Matching threshold (0-1).
            
        Returns:
            tuple: (x, y, w, h) coordinates of match, or None if not found.
        """
        try:
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                logger.error(f"Could not load template: {template_path}")
                return None
                
            result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                w, h = template.shape[1], template.shape[0]
                return (max_loc[0], max_loc[1], w, h)
            else:
                logger.debug(f"Template {template_path} not found in screenshot (max match: {max_val:.2f})")
                return None
                
        except Exception as e:
            logger.exception(f"Error finding template: {e}")
            return None