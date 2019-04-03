"""Argparse"""
# Standard dist imports
import argparse
import os

# Third party imports

# Project level imports
from . import main

# Module level constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('')

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__