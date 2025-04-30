#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for placing bets and executing actions in a poker game.
"""

import logging
import time
import pyautogui
import cv2
import numpy as np
import os
import random  # Added proper random module import
from decision_maker import Decision
from screen_capture import ScreenCapture

logger = logging.getLogger(__name__)


class BetPlacer:
    """Class for placing bets and executing actions in a poker game."""
    
    def __init__(self, config=None):
        """
        Initialize the bet placer with optional configuration.
        
        Args:
            config (dict, optional): Configuration settings
        """
        self.config = config or {}
        
        # Default button locations (these would need to be calibrated for each poker client)
        self.button_locations = self.config.get('button_locations', {
            'fold': None,
            'check': None,
            'call': None,
            'bet': None,
            'raise': None,
            'bet_input': None,
            'confirm': None
        })
        
        # Delay between actions to avoid detection/rate limiting
        self.action_delay = self.config.get('action_delay', 0.5)
        
        # Movement randomization to avoid detection
        self.randomize_movement = self.config.get('randomize_movement', True)
        
        # Safety timeout before all actions
        self.safety_timeout = self.config.get('safety_timeout', 1.0)
        
        # Resource path for button templates
        self.resources_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')
        
        # Client templates for button detection
        self.client_templates = {}
        
        # Load client templates
        self._load_client_templates()
        
        logger.info("Bet placer initialized")
    
    def _load_client_templates(self):
        """Load client templates for button detection."""
        try:
            # Load client reference images
            for i in range(1, 6):
                client_path = os.path.join(self.resources_path, f'client{i}.png' if i > 1 else 'client.png')
                if os.path.exists(client_path):
                    client_img = cv2.imread(client_path, cv2.IMREAD_COLOR)
                    if client_img is not None:
                        self.client_templates[f'client{i}'] = client_img
                        logger.info(f"Loaded client template {i} for button detection")
                    else:
                        logger.warning(f"Failed to load client template {client_path}")
                        
            if not self.client_templates:
                logger.warning("No client templates were loaded for button detection")
        except Exception as e:
            logger.exception(f"Error loading client templates: {e}")
    
    def execute_action(self, decision: Decision) -> bool:
        """
        Execute a poker action based on the decision.
        
        Args:
            decision (Decision): The decision to execute
            
        Returns:
            bool: True if action was executed successfully, False otherwise
        """
        try:
            logger.info(f"Executing action: {decision}")
            
            # Safety timeout before any action
            time.sleep(self.safety_timeout)
            
            if decision.action == Decision.ACTION_FOLD:
                return self._click_button('fold')
                
            elif decision.action == Decision.ACTION_CHECK:
                return self._click_button('check')
                
            elif decision.action == Decision.ACTION_CALL:
                return self._click_button('call')
                
            elif decision.action == Decision.ACTION_BET:
                return self._place_bet(decision.amount)
                
            elif decision.action == Decision.ACTION_RAISE:
                return self._place_raise(decision.amount)
                
            else:
                logger.warning(f"Unknown action: {decision.action}")
                return False
                
        except Exception as e:
            logger.exception(f"Error executing action: {e}")
            return False
            
    def _click_button(self, button_name: str) -> bool:
        """
        Click a button on the poker client.
        
        Args:
            button_name (str): Name of the button to click
            
        Returns:
            bool: True if clicked successfully, False otherwise
        """
        button_location = self.button_locations.get(button_name)
        
        if not button_location:
            logger.warning(f"Button location for '{button_name}' not defined")
            return False
            
        try:
            # Move mouse to button location
            x, y = button_location
            
            # Add some randomization to avoid detection
            if self.randomize_movement:
                x += random.uniform(-5, 5)  # Using standard random module
                y += random.uniform(-3, 3)  # Using standard random module
                
            # Move mouse to location
            pyautogui.moveTo(x, y, duration=0.3)
            time.sleep(0.1)
            
            # Click the button
            pyautogui.click(x, y)
            
            logger.info(f"Clicked {button_name} button at ({x}, {y})")
            
            # Add delay after action
            time.sleep(self.action_delay)
            
            return True
            
        except Exception as e:
            logger.exception(f"Error clicking {button_name} button: {e}")
            return False
            
    def _enter_bet_amount(self, amount: float) -> bool:
        """
        Enter a bet amount in the betting input field.
        
        Args:
            amount (float): Amount to bet
            
        Returns:
            bool: True if amount was entered successfully, False otherwise
        """
        input_location = self.button_locations.get('bet_input')
        
        if not input_location:
            logger.warning("Bet input location not defined")
            return False
            
        try:
            # Move mouse to input field
            x, y = input_location
            
            # Add some randomization to avoid detection
            if self.randomize_movement:
                x += random.uniform(-3, 3)  # Using standard random module
                y += random.uniform(-2, 2)  # Using standard random module
                
            # Click the input field
            pyautogui.moveTo(x, y, duration=0.3)
            time.sleep(0.1)
            pyautogui.click(x, y)
            time.sleep(0.1)
            
            # Clear the input field (select all + delete)
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.1)
            pyautogui.press('delete')
            time.sleep(0.1)
            
            # Type the amount
            formatted_amount = f"{amount:.2f}"
            pyautogui.write(formatted_amount, interval=0.05)
            
            logger.info(f"Entered bet amount: {formatted_amount}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error entering bet amount: {e}")
            return False
            
    def _place_bet(self, amount: float) -> bool:
        """
        Place a bet of a specific amount.
        
        Args:
            amount (float): Amount to bet
            
        Returns:
            bool: True if bet was placed successfully, False otherwise
        """
        try:
            # First enter the bet amount
            if not self._enter_bet_amount(amount):
                return False
                
            # Click the bet button
            if not self._click_button('bet'):
                return False
                
            # Some poker clients may require a confirmation click
            confirm_location = self.button_locations.get('confirm')
            if confirm_location:
                time.sleep(0.2)
                if not self._click_button('confirm'):
                    return False
                    
            logger.info(f"Successfully placed bet: {amount:.2f}")
            return True
            
        except Exception as e:
            logger.exception(f"Error placing bet: {e}")
            return False
            
    def _place_raise(self, amount: float) -> bool:
        """
        Place a raise of a specific amount.
        
        Args:
            amount (float): Amount to raise to
            
        Returns:
            bool: True if raise was placed successfully, False otherwise
        """
        try:
            # First enter the raise amount
            if not self._enter_bet_amount(amount):
                return False
                
            # Click the raise button
            if not self._click_button('raise'):
                return False
                
            # Some poker clients may require a confirmation click
            confirm_location = self.button_locations.get('confirm')
            if confirm_location:
                time.sleep(0.2)
                if not self._click_button('confirm'):
                    return False
                    
            logger.info(f"Successfully placed raise: {amount:.2f}")
            return True
            
        except Exception as e:
            logger.exception(f"Error placing raise: {e}")
            return False

    def calibrate_button_locations(self):
        """
        Calibrate button locations interactively or automatically using template matching.
        """
        logger.info("Starting button calibration")
        
        # First check if we can detect buttons automatically using template matching
        if self._try_automatic_calibration():
            logger.info("Automatic button calibration successful")
            return True
            
        # If automatic calibration fails, fallback to interactive calibration
        return self._interactive_calibration()
    
    def _try_automatic_calibration(self) -> bool:
        """
        Try to automatically detect button locations using template matching.
        
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        try:
            if not self.client_templates:
                logger.warning("No client templates available for automatic calibration")
                return False
                
            logger.info("Attempting automatic button calibration")
            
            # Take a screenshot of the current screen
            screen_capture = ScreenCapture()
            screenshot = screen_capture.capture()
            if screenshot is None:
                logger.error("Failed to capture screenshot for calibration")
                return False
                
            # Try to identify which client we're using
            client_match = None
            client_match_score = 0
            client_match_loc = None
            
            for client_name, template in self.client_templates.items():
                try:
                    # Use template matching to find the client
                    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > client_match_score and max_val >= 0.7:
                        client_match = client_name
                        client_match_score = max_val
                        client_match_loc = max_loc
                        client_template = template
                        
                except Exception as e:
                    logger.exception(f"Error matching client template {client_name}: {e}")
            
            if client_match is None:
                logger.warning("Could not match any client template")
                return False
                
            logger.info(f"Detected client: {client_match} (score: {client_match_score:.2f})")
            
            # Based on the detected client, we can compute the relative positions of buttons
            # This would require calibration data for each client
            # For example, if we know that the fold button is 100px below and 50px to the left of the top-left corner
            # of the client template, we can compute its absolute position
            
            # This is a simplified example and should be customized based on the actual client UI
            client_x, client_y = client_match_loc
            
            # These offsets would need to be determined for each client template
            # For demonstration, we'll use placeholder values
            if client_match == 'client1':
                self.button_locations = {
                    'fold': (client_x + 100, client_y + 300),
                    'check': (client_x + 200, client_y + 300),
                    'call': (client_x + 200, client_y + 300),  # Same as check in many clients
                    'bet': (client_x + 300, client_y + 300),
                    'raise': (client_x + 300, client_y + 300),  # Same as bet in many clients
                    'bet_input': (client_x + 250, client_y + 250),
                    'confirm': (client_x + 350, client_y + 350)
                }
            elif client_match == 'client2':
                # Different offsets for client2
                self.button_locations = {
                    'fold': (client_x + 120, client_y + 320),
                    'check': (client_x + 220, client_y + 320),
                    'call': (client_x + 220, client_y + 320),
                    'bet': (client_x + 320, client_y + 320),
                    'raise': (client_x + 320, client_y + 320),
                    'bet_input': (client_x + 270, client_y + 270),
                    'confirm': (client_x + 370, client_y + 370)
                }
            # Add more client-specific offsets here
            
            # For now, log the calibrated positions
            for button, position in self.button_locations.items():
                logger.info(f"Calibrated {button} button at {position}")
                
            return True
            
        except Exception as e:
            logger.exception(f"Error during automatic calibration: {e}")
            return False
    
    def _interactive_calibration(self) -> bool:
        """
        Interactively calibrate button locations by asking the user to click on each button.
        
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        try:
            logger.info("Starting interactive button calibration")
            print("\n=== POKER BOT CALIBRATION ===")
            print("Please follow the instructions to calibrate the poker bot.")
            print("You'll need to click on each button when prompted.")
            
            # For each button type
            for button in ['fold', 'check/call', 'bet/raise', 'bet_input', 'confirm']:
                print(f"\nMove your mouse to the {button} button and press Enter...")
                input()  # Wait for user to press Enter
                
                # Get current mouse position
                x, y = pyautogui.position()
                
                if button == 'check/call':
                    self.button_locations['check'] = (x, y)
                    self.button_locations['call'] = (x, y)
                    logger.info(f"Set check and call button at ({x}, {y})")
                elif button == 'bet/raise':
                    self.button_locations['bet'] = (x, y)
                    self.button_locations['raise'] = (x, y)
                    logger.info(f"Set bet and raise button at ({x}, {y})")
                else:
                    self.button_locations[button] = (x, y)
                    logger.info(f"Set {button} button at ({x}, {y})")
                
                print(f"{button} button position recorded: ({x}, {y})")
            
            print("\nCalibration complete!")
            
            # Save the calibration data to the configuration
            self.config['button_locations'] = self.button_locations
            
            # In a real implementation, you would also save this to a configuration file
            # But for simplicity, we'll just keep it in memory for now
            
            return True
            
        except Exception as e:
            logger.exception(f"Error during interactive calibration: {e}")
            return False