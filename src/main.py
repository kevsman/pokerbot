#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the poker automation application.
"""

import time
import logging
from screen_capture import ScreenCapture
from game_state import GameStateDetector
from hand_analyzer import HandAnalyzer
from decision_maker import DecisionMaker
from bet_placer import BetPlacer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        self.game_detector = GameStateDetector()
        self.hand_analyzer = HandAnalyzer()
        self.decision_maker = DecisionMaker()
        self.bet_placer = BetPlacer()
        self.running = False
        logger.info("Poker Bot initialized")
        
    def start(self):
        """Start the poker bot main loop."""
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
                
                # Step 5: Place bet or take action
                self.bet_placer.execute_action(decision)
                
                # Wait before next cycle
                time.sleep(self.config.get('cycle_delay', 1))
                
        except KeyboardInterrupt:
            logger.info("Poker Bot stopped by user")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the poker bot."""
        self.running = False
        logger.info("Poker Bot stopped")


if __name__ == "__main__":
    # Example usage
    bot = PokerBot({
        'cycle_delay': 1.5,
        # Add other configuration options here
    })
    bot.start()