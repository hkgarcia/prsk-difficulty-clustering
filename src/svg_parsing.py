# read .svg into a list of path objects / list of dictionaries of attributes
import os
from svgpathtools import svg2paths
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# --- [ helper functions ] ---
# function to get the vertical position (y-coordinate) of an SVG element; some elements use 'y1'(lines) and some use 'y' (shapes/rectangles)
def get_y(attr):
    return float(attr.get('y1', attr.get('y', 0)))


# --- [ loading in .CSV data (song_id, difficulty, note_count, song_title, bpm, playback_time_minutes, playback_time_seconds) ] ---
csv_path = os.path.join(os.path.dirname(__file__), '../data/song_metadata.csv')
song_metadata = pd.read_csv(csv_path)
#print(song_metadata.head())


# --- [ loading .SVG file for a specific chart ] ---
# TODO: handle all charts
svg_path = os.path.join(os.path.dirname(__file__), '../data/example_charts/example1.svg')
paths, attributes = svg2paths(svg_path)

print("*" * 40)
print(f"Reading in SVG file: {svg_path}")
print(f"Total paths in SVG: {len(paths)}")

# --- [ filtering for relevant classes (lines) and playable notes (slide, slide-critical) ] ---
lane_lines  = [attr for attr in attributes if attr.get('class') == 'lane-line']
bar_lines    = [attr for attr in attributes if attr.get('class') == 'bar-line']
beat_lines   = [attr for attr in attributes if attr.get('class') == 'beat-line']
tick_lines   = [attr for attr in attributes if attr.get('class') == 'tick-line']
notes        = [attr for attr in attributes if attr.get('class') in ['slide', 'slide-critical']]

print("*" * 40)
print(f"Total lane-lines: {len(lane_lines)}")
print(f"Total bar-lines: {len(bar_lines)}")
print(f"Total beat_lines: {len(beat_lines)}")
print(f"Total tick-lines: {len(tick_lines)}")
print(f"Total notes (slide, slide-critical): {len(notes)}")


### [ NOTES ] ###
# we use Y position to map notes to notes
# in the SVG, vertical position (Y) encodes timing: higher Y = earlier in song, lower Y = later in song -> Y always increases downards in SVG
# although the SVG increases Y downwards, the chart's timing is drawn such that earliest notes are at the bottom (largest Y value) and later notes are at the top (smallest Y value)
# sorting notes from highest Y to lowest Y recreates the correct temporal order of the song
# X position is used for lanes and has no timing information, so we only use Y for timing calculations

# --- [ combine all notes: slide, slide-critical, and tick-lines ] ---
# all notes are mapped to a bar, and we include tick lines to calculate relative timing
all_notes = notes + tick_lines

# sort all elements by descending Y position (higher Y = earlier in song)
all_notes.sort(key=get_y, reverse=True) 
bar_lines.sort(key=get_y, reverse=True)
beat_lines.sort(key=get_y, reverse=True)

print("*" * 40)
print(f"Total notes + tick-lines: {len(all_notes)}")
print(f"First 5 note Y positions: {[get_y(n) for n in all_notes[:5]]}")
print(f"First 5 bar-line Y positions: {[get_y(b) for b in bar_lines[:5]]}")
print(f"First 5 beat-line Y positions: {[get_y(b) for b in beat_lines[:5]]}")


### [ NOTES ] ###
# a bar line represents a measure (4 beats) in music
# the SVG may contain multiple bar-lines at the same Y position due to layering
# thus, we only keep one bar-line per unique Y position to represent the measure and to avoid zero-height bars
# and to avoid confusion when mapping notes to bars (since our bar lines and beat lines never change X position, unlike lanes)
# even if there are only 5 bar lines, we extend the first and last bars to include notes outside the bar range
# so, even a small number of bar lines can still cover all notes in the chart

# --- [ remove duplicate bar-lines ] ---
# due to the bars being stacked, we only keep unique Y-positions for bar-lines
unique_bar_lines = []
seen_y = set()
for b in bar_lines:
    y = get_y(b)
    if y not in seen_y:
        unique_bar_lines.append(b)
        seen_y.add(y)
bar_lines = unique_bar_lines

print(f"Total unique bar-lines: {len(bar_lines)}")


# --- [ create bars from bar-lines] ---
# each bar is defined by the space between two consecutive bar-lines (top Y to bottom Y)
# this gives us the vertical range of each bar for mapping notes

bars = []
for i in range(len(bar_lines)-1):
    top = get_y(bar_lines[i])
    bottom = get_y(bar_lines[i+1])
    
    # ensure bar has non-zero height to avoid zero-division errors
    if top == bottom:
        bottom = top - 1e-6
        
    # each bar is represented as (top_y, bottom_y)
    bars.append((top, bottom))


# --- [ extend first and last bars to include notes outside bar range ] ---
# some notes may be visually above the first bar or below the last bar - handling any missed values
min_y = min(get_y(n) for n in all_notes)
max_y = max(get_y(n) for n in all_notes)
if max_y > bars[0][0]:
    bars[0] = (max_y, bars[0][1])
if min_y < bars[-1][1]:
    bars[-1] = (bars[-1][0], min_y)

print(f"Total bars: {len(bars)}")


# --- [ map notes to bars ] ---
note_bars = []
for note in all_notes:
    y = get_y(note)
    # find the bar index where note Y fits between bar top and bottom
    bar_idx = next((i for i, (top, bottom) in enumerate(bars) if bottom <= y <= top), len(bars)-1)
    note_bars.append((note, bar_idx))

print("First 10 notes mapped to bars (Y, bar):", [(get_y(n), idx) for n, idx in note_bars[:10]])


# --- [ filter beat-lines to notes Y-range ] ---
# only keep beat-lines that fall within the min and max Y of all notes
filtered_beats = [b for b in beat_lines if min_y <= get_y(b) <= max_y]
print(f"Total beat-lines after filtering: {len(filtered_beats)}")
print("First 10 filtered beat Y positions:", [get_y(b) for b in filtered_beats[:10]])


# --- [ map filtered beat-lines to closest bars ] ---
# create a mapping of bar index -> list of beat Y-positions within that bar
# each bar may have multiple beat-lines representing quarter notes (subdivisions of a bar)
# for each beat line, we find the closest bar (by Y position) using the center of the bar (bar midpoints)
# this creates a dictionary: bar index -> list of beat Y-positions

bar_beats = defaultdict(list)
for beat in filtered_beats:
    y = get_y(beat)
    closest_bar_idx = min(
        range(len(bars)),
        key=lambda i: abs((bars[i][0] + bars[i][1])/2 - y)
    )
    if y not in bar_beats[closest_bar_idx]:
        bar_beats[closest_bar_idx].append(y)


# --- [ sort beats descending and handle single-beat bars ] ---
# for each note, we find the bar where its Y falls, resulting in every note being assigned a bar index

for bar_idx in bar_beats:
    bar_beats[bar_idx].sort(reverse=True)
       
     # if there is only 1 beat in a bar, create two very close beats to avoid zero-division later
    if len(bar_beats[bar_idx]) == 1:
        y = bar_beats[bar_idx][0]
        bar_beats[bar_idx] = [y + 1e-6, y - 1e-6] 

print("Bar -> number of beat-lines (first 10 bars with beats):")
for bar_idx in sorted(bar_beats.keys())[:10]:
    print(f"Bar {bar_idx}: {len(bar_beats[bar_idx])} beats, first 5 Y: {bar_beats[bar_idx][:5]}")


# --- [ calculate timestamps for notes using beat-lines and BPM ] ---
# TODO: handle reading in data
song_duration = 99  # seconds
bpm = 131
beats_per_bar = 4
seconds_per_beat = 60 / bpm
epsilon = 1e-6 # small value to handle edge cases

no_fallback_count = 0   # notes mapped using beat-lines
fallback_count = 0      # notes mapped using fallback linear mapping

note_times = []
for note, bar_idx in note_bars:
    note_y = get_y(note)
    bar_top, bar_bottom = bars[bar_idx]
    bar_height = bar_top - bar_bottom
    if bar_height == 0:
        bar_height = 1e-6

    beats_in_bar = bar_beats.get(bar_idx, [])
    assigned = False

    # --- [ map using beat-lines if available ] ---
    # for a note within a bar, find the two beat lines it is between
    # then, calculate its fractional/relative position within that beat segment
    # lastly, calculate the time in seconds to assign to the note
    if beats_in_bar and len(beats_in_bar) > 1:
        for i in range(len(beats_in_bar)-1):
            y_top_beat = beats_in_bar[i]
            y_bottom_beat = beats_in_bar[i+1]
            beat_height = y_top_beat - y_bottom_beat
            
            if beat_height == 0:
                beat_height = 1e-6
                
            # check if note Y falls within this beat segment
            if y_bottom_beat - epsilon <= note_y <= y_top_beat + epsilon:
                fraction = (y_top_beat - note_y) / beat_height      # relative position within the beat
                timestamp = (
                    bar_idx * beats_per_bar * seconds_per_beat +    # offset from previous bars
                    i * seconds_per_beat +                          # offset within current bar
                    fraction * seconds_per_beat                     # fraction within beat
                )
                
                note_times.append(timestamp)
                assigned = True
                no_fallback_count += 1
                
                # debugging print
                #print(f"Note Y={note_y} mapped to bar {bar_idx}, beat {i}, timestamp={timestamp:.3f}")
                
                break # stop after assigning

    # --- [ fallback: linear mapping within bar if no beats - less accurate ] ---
    # if no beat lines exist in this bar, we linearly map the note's Y within the bar
    if not assigned:
        fraction = (bar_top - note_y) / bar_height
        timestamp = bar_idx * beats_per_bar * seconds_per_beat + fraction * beats_per_bar * seconds_per_beat
        note_times.append(timestamp)
        fallback_count += 1
        
        # debugging print
        #print(f"Note Y={note_y} fallback mapped to bar {bar_idx}, timestamp={timestamp:.3f}")


# --- [ scale timestamps to fit the actual song duration ] ---
# we scale all timestamps so that the maximum timestamp matches the song duration/real playback time
min_time, max_time = min(note_times), max(note_times)
scale = song_duration / max_time if max_time > 0 else 1
note_times = [t * scale for t in note_times]

print("First 10 note timestamps:", note_times[:10])
print(f"Min timestamp: {min(note_times)}, Max timestamp: {max(note_times)}")
print(f"Notes mapped using beat-lines: {no_fallback_count}, fallback mappings: {fallback_count}")


# --- [ visualization ] ---
# separate notes by type for plotting
slide_notes = [i for i, (n, _) in enumerate(note_bars) if n.get('class') == 'slide']
slide_critical_notes = [i for i, (n, _) in enumerate(note_bars) if n.get('class') == 'slide-critical']
tick_notes = [i for i, (n, _) in enumerate(note_bars) if n.get('class') == 'tick-line']

note_ys = [get_y(n) for n, _ in note_bars]

plt.figure(figsize=(14, 10))

# jitter to avoid overplotting
jitter_y = 2.0      # SVG pixels
jitter_x = 0.15     # seconds

### [ NOTES ] ###
# slide notes appear very close together vertically due to their structure - they occur within the same beats or bar
# tick notes are more spread out since they represent timing points
# slide notes cluster due to them visually appearing in the same small vertical space, and since we have few bar lines

# plotting slide notes (very close together due to their structure)
plt.scatter(
    [note_times[i] + np.random.uniform(-jitter_x, jitter_x) for i in slide_notes],
    [note_ys[i] + np.random.uniform(-jitter_y, jitter_y) for i in slide_notes],
    s=25, alpha=0.8, c='green', label='Slide'
)

# plotting slide-critical notes (very close together due to their structure)
plt.scatter(
    [note_times[i] + np.random.uniform(-jitter_x, jitter_x) for i in slide_critical_notes],
    [note_ys[i] + np.random.uniform(-jitter_y, jitter_y) for i in slide_critical_notes],
    s=25, alpha=0.8, c='yellow', label='Slide-Critical'
)

# plotting tick-line notes
plt.scatter(
    [note_times[i] for i in tick_notes],
    [note_ys[i] for i in tick_notes],
    s=8, alpha=0.5, c='blue', label='Tick'
)

# draw horizontal lines for bars
for top, bottom in bars:
    plt.hlines(top, 0, song_duration, colors='red', linestyles='dashed', alpha=0.6)

# draw horizontal lines for beats
for bar_idx, beat_ys in bar_beats.items():
    for y in beat_ys:
        plt.hlines(y, 0, song_duration, colors='gray', linestyles='dotted', alpha=0.3)

# --- configure axes and title ---
plt.gca().invert_yaxis()  # match SVG orientation (top is first)
plt.xlabel("Time (seconds)")
plt.ylabel("Y position in SVG")
plt.title("All Notes with Tick, Slide, Slide-Critical, Beat, and Bar Lines")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()