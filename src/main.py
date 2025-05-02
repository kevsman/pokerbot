#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the poker automation application.
"""

import time
import logging
import os
import argparse
from screen_capture import ScreenCapture
from game_state import GameStateDetector
from hand_analyzer import HandAnalyzer
from decision_maker import DecisionMaker
from bet_placer import BetPlacer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Fixed levelname typo
    handlers=[
        logging.FileHandler("poker_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PokerBot:
    """Main class that orchestrates the poker automation process."""
    
    def __init__(self, config=None):
        """Initialize the poker bot with optional configuration."""
        self.config = config or {}
        self.screen_capture = ScreenCapture()
        # Pass config to GameStateDetector for window detection settings
        self.game_detector = GameStateDetector(config)
        self.hand_analyzer = HandAnalyzer()
        self.decision_maker = DecisionMaker()
        self.bet_placer = BetPlacer()
        self.running = False
        logger.info("Poker Bot initialized")
        
        # Log the debug mode setting
        if self.config.get('concise_logging', False):
            logger.info("Running in concise logging mode (no debug images)")
        
    def calibrate(self):
        """Calibrate the poker bot before starting."""
        logger.info("Starting poker bot calibration")
        print("\n=== Poker Bot Calibration ===")
        print("Calibrating the bot to recognize the poker client and UI elements.")
        
        # First, try automatic calibration using the client templates
        print("\nAttempting automatic calibration...")
        if self.bet_placer.calibrate_button_locations():
            print("Automatic calibration successful!")
            return True
            
        print("\nAutomatic calibration failed. Falling back to manual calibration.")
        print("Please make sure your poker client is visible on screen.")
        input("Press Enter to continue with manual calibration...")
        
        # Fall back to manual calibration if automatic fails
        return self.bet_placer.calibrate_button_locations()
        
    def start(self, calibrate_first=True):
        """
        Start the poker bot main loop.
        
        Args:
            calibrate_first (bool): Whether to calibrate before starting
        """
        # First, calibrate the bot if requested
        if calibrate_first:
            if not self.calibrate():
                logger.error("Calibration failed. Cannot start the bot.")
                return False
        
        self.running = True
        logger.info("Poker Bot started")
        
        try:
            while self.running:
                # Step 1: Capture the screen
                screenshot = self.screen_capture.capture()
                
                # Step 2: Detect game state
                game_state = self.game_detector.detect_state(screenshot)
                
                if not game_state.is_active:
                    logger.info("No active game detected, waiting...")
                    time.sleep(2)
                    continue
                
                # Step 3: Analyze current hand
                hand_strength = self.hand_analyzer.analyze(game_state)
                
                # Step 4: Make decision based on hand and game state
                decision = self.decision_maker.decide(game_state, hand_strength)
                
                # Step 5: Place bet or take action - now passing is_our_turn flag
                self.bet_placer.execute_action(decision, game_state.is_our_turn)
                
                # Enhanced concise logging when in concise mode
                if self.config.get('concise_logging', False) and game_state.is_our_turn:
                    player_cards = ', '.join(str(card) for card in game_state.player_cards)
                    community_cards = ', '.join(str(card) for card in game_state.community_cards)
                    logger.info(f"CONCISE LOG: Player cards: [{player_cards}] | "
                               f"Community cards: [{community_cards}] | "
                               f"Hand: {hand_strength['hand_type']} | "
                               f"Decision: {decision}")
                
                # Wait before next cycle
                time.sleep(self.config.get('cycle_delay', 1))
                
        except KeyboardInterrupt:
            logger.info("Poker Bot stopped by user")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            self.stop()
            
        return True
    
    def stop(self):
        """Stop the poker bot."""
        self.running = False
        logger.info("Poker Bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Poker Automation Bot')
    parser.add_argument('--no-calibration', action='store_true', help='Skip calibration step')
    parser.add_argument('--aggression', type=float, default=0.5, help='Aggression level (0-1)')
    parser.add_argument('--bluff-frequency', type=float, default=0.1, help='Bluff frequency (0-1)')
    
    # Add new arguments for window detection
    parser.add_argument('--use-window-detection', action='store_true', default=True, 
                        help='Use window title detection instead of template matching (default: True)')
    parser.add_argument('--window-title', type=str, 
                        help='Specific window title to look for (e.g., "My Poker Game")')
    parser.add_argument('--force-templates', action='store_true', 
                        help='Force using template matching even if window detection is available')
    
    # Add new argument for concise logging mode
    parser.add_argument('--concise-logging', action='store_true',
                        help='Enable concise logging (only log cards, hand rank and decisions; no debug images)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'cycle_delay': 1.5,
        'aggression': args.aggression,
        'bluff_frequency': args.bluff_frequency,
        
        # Add window detection settings
        'use_window_detection': args.use_window_detection and not args.force_templates,
        'window_title': args.window_title,
        
        # Add concise logging setting
        'concise_logging': args.concise_logging,
    }
    
    # Initialize and start the bot
    bot = PokerBot(config)
    bot.start(calibrate_first=not args.no_calibration)


if __name__ == "__main__":
    # Execute main function
    main()