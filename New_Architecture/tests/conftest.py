# New_Architecture/tests/conftest.py
import sys
import os
from pathlib import Path

# Add the New_Architecture directory to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))