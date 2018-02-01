#!/usr/bin/env
""" file:xmeos/__init__.py
    author: Aaron S. Wolf
    date: Tuesday July 26, 2017

    description="Crystal-Melt Equation Of State Modeling in Python",
"""
# Load all core and models methods and place in xmeos namespace
from . import models

__all__ = [s for s in dir() if not s.startswith('_')]
