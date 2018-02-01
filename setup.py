#!/usr/bin/env python
""" file: setup.py
    modified: Aaron S. Wolf, University of Michigan
    date: July 26, 2017, revised July 26, 2017

    description: Distutils installer script for xmeos.
"""
from setuptools import find_packages, setup
setup(
    name="xmeos",
    version="0.1",
    description="Crystal-Melt Equation Of State Modeling in Python",
    author="Aaron S. Wolf",
    author_email='aswolf@umich.edu',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="BSD",
    url="https://github.com/aswolf/xmeos",
    # packages=[
    #       'xmeos',
    # ],
    packages=find_packages(),
)
