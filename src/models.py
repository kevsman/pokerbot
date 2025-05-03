#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing shared data models for the poker bot.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Card:
    """Class representing a playing card."""
    rank: str
    suit: str
    
    def __str__(self):
        return f"{self.rank}{self.suit}"