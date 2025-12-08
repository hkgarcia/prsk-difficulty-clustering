# source: https://github.com/mathandy/svgpathtools
# read .svg into a list of path objects / list of dictionaries of attributes
import os
from svgpathtools import svg2paths

### [ EXAMPLES USED FOR ANALYZING SVG FILES ] ###
# example1.svg: Glory Steady Go (166): MASTER, 875 total notes, 131 bpm, 1:39 mins
# example2.svg: NEO (366): APPEND, 1365 total notes, 190 bpm, 2:06 mins
# example3.svg: "BOSS/ULTIMATE" Medley/Rush (676): APPEND, 4181 total notes,, 2:06 mins

svg_path = os.path.join(os.path.dirname(__file__), '../../data/charts/166_master.svg')
paths, attributes = svg2paths(svg_path)

# finding unique classes of paths in the svg file
unique_classes = set()
for path_attributes in attributes:
    if 'class' in path_attributes:
        unique_classes.add(path_attributes['class'])

print("Unique classes in .SVG:")
for x in unique_classes:
    print(x)

### [ CLASS ANALYSIS RESULTS FOR .SVG FILES ] ###

## [ example1.svg ] ## 
# bar-count-flag: 64
# lane: 13
# slide: 107
# slide-critical: 34
# lane-line: 91
# meta: 1
# meta-line: 1
# tick-line: 1191 (note how total note count is 875 compared to the ticklines)
# background: 14
# bar-line: 65
# beat-line: 192
# event-flag: 130

## [ example2.svg ] ##
# released for project sekai's 3rd anniversary (9/30/23), NEO introduced new note types and the new APPEND mode
# speed-line: Markers for speed or tempo changes
# decoration: Background elements or decorative paths (unclickable)

## [ example3.svg ] ##
# released for the 5th anniversary, the master medley combined the most difficult parts of infamous songs into one long chart
# used to check for any new classes that may have been missed previously

# code used for counting and analyzing over each class type/path
# count = 0
# for decoration in attributes:
#     if decoration.get('class') == 'decoration':
#         count += 1
#         print(decoration)

# print(f"Total decoration paths: {count}")