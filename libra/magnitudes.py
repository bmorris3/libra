import os
import json

magnitudes_path = os.path.join(os.path.dirname(__file__), 'data', 'mags.json')

__all__ = ['magnitudes']

magnitudes = json.load(open(magnitudes_path, 'r'))
