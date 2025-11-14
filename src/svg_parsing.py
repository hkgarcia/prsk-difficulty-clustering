# read .svg into a list of path objects / list of dictionaries of attributes
import os
from svgpathtools import svg2paths
import pandas as pd
from collections import defaultdict

# loading in .CSV data (song_id, difficulty, note_count, song_title, bpm, playback_time_minutes, playback_time_seconds)
csv_path = os.path.join(os.path.dirname(__file__), '../data/song_metadata.csv')
song_metadata = pd.read_csv(csv_path)
#print(song_metadata.head())

# loading .SVG file for a specific chart
# TODO: handle all charts
svg_path = os.path.join(os.path.dirname(__file__), '../data/example_charts/example1.svg')
paths, attributes = svg2paths(svg_path)

# filtering for relevant classes (lines) and playable notes (slide, slide-critical)
bar_lines    = [attr for attr in attributes if attr.get('class') == 'bar-line']
beat_lines   = [attr for attr in attributes if attr.get('class') == 'beat-line']
tick_lines   = [attr for attr in attributes if attr.get('class') == 'tick-line']
notes        = [attr for attr in attributes if attr.get('class') in ['slide', 'slide-critical']]

# map notes to timestamps (for later variables) based on tick-line positions, BPM, and playback time
# tick-line structure: {'class': 'tick-line', 'x1': '16', 'x2': '40', 'y1': '2176', 'y2': '2176'}

# using y1 or y2 (equal) to determine vertical position: lower y = earlier in the song (can be a negative value)
# y1/y2 values range from -1081 to 3989

# print all tick-lines to analyze their structure
tick_lines = [
    attr for attr in attributes 
    if attr.get('class') == 'tick-line'
]

# sort tick lines by their y1 value (ascending)
tick_lines = sorted(
    [attr for attr in attributes if attr.get('class') == 'tick-line'], 
    key=lambda x: float(x['y1'])
)

# # print tick lines (information)
# for tick in tick_lines:
#     print(tick)
    
