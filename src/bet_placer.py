#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for placing bets and executing actions in a poker game.
"""

import logging
import time
import pyautogui
from decision_maker import Decision

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
        
        logger.info("Bet placer initialized")
    
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
                x += pyautogui.random.uniform(-5, 5)
                y += pyautogui.random.uniform(-3, 3)
                
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
                x += pyautogui.random.uniform(-3, 3)
                y += pyautogui.random.uniform(-2, 2)
                
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
        Calibrate button locations interactively. This would prompt the user
        to click on each button in the poker client to learn their locations.
        
        Note: This is a placeholder for a real implementation.
        """
        logger.info("Button calibration would happen here")
        # In a real implementation, you would:
        # 1. Display instructions to the user
        # 2. Wait for the user to press a key when ready
        # 3. Record mouse positions when the user clicks on each button
        # 4. Save these positions to config