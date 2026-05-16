"""Shared filesystem paths for the data package.

Examples:
--------
>>> from honey.data.config import CACHE_DIR
>>> CACHE_DIR.name
'cache'
"""

from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
