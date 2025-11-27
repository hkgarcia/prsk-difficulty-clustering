# read .svg into a list of path objects / list of dictionaries of attributes
import os
from svgpathtools import svg2paths
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --- [ helper functions ] ---
# function to get the vertical position (y-coordinate) of an SVG element; some elements use 'y1'(lines) and some use 'y' (shapes/rectangles)
def get_y(attr):
    return float(attr.get('y1', attr.get('y', 0)))


# --- [ loading in .CSV data (song_id, difficulty, note_count, song_title, bpm, playback_time_minutes, playback_time_seconds) ] ---
csv_path = os.path.join(os.path.dirname(__file__), '../../data/song_metadata.csv')
song_metadata = pd.read_csv(csv_path)

# --- [ loading .SVG file for a specific chart ] ---
#svg_path = os.path.join(os.path.dirname(__file__), '../data/example_charts/601_master.svg')
# svg_path = os.path.join(os.path.dirname(__file__), '../data/example_charts/366_append.svg')
# svg_path = os.path.join(os.path.dirname(__file__), '../data/example_charts/166_master.svg')
svg_path = os.path.join(os.path.dirname(__file__), '../../data/charts/3_master.svg')

paths, attributes = svg2paths(svg_path)

# --- [ loading specific song data for a specific chart ] ---
# svg file: "1_master.svg"; return bpm, duration_seconds, note_count
def get_song_metadata_from_filename(svg_filename):
    base = os.path.basename(svg_filename)
    song_id_str, difficulty_ext = base.split("_", 1)
    difficulty = difficulty_ext.replace(".svg", "")
    
    song_id = int(song_id_str)
    
    # lookup metadata
    row = song_metadata[
        (song_metadata['song_id'] == song_id) &
        (song_metadata['difficulty'].str.lower() == difficulty.lower())
    ]
    
    if row.empty:
        raise ValueError(f"No metadata found for song_id={song_id}, difficulty={difficulty}")
    
    bpm = float(row['bpm'].values[0])
    duration_seconds = float(row['playback_time_seconds'].values[0])
    note_count = int(row['note_count'].values[0])
    
    return bpm, duration_seconds, note_count

bpm, song_duration, note_count_csv = get_song_metadata_from_filename(svg_path)

print("*" * 40)
print(f"Loaded chart: {svg_path}")
print(f"Metadata -> BPM: {bpm}, duration: {song_duration}s, note_count (CSV): {note_count_csv}")
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
song_duration = song_duration  # seconds
bpm = bpm  # beats per minute
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

print("*" * 40)
print("First 10 note timestamps:", note_times[:10])
print(f"Min timestamp: {min(note_times)}, Max timestamp: {max(note_times)}")
print(f"Notes mapped using beat-lines: {no_fallback_count}, fallback mappings: {fallback_count}")


# print number of notes per bar for diagnostics
notes_per_bar = Counter()
for _, bar_idx in note_bars:
    notes_per_bar[bar_idx] += 1
print("\nNotes per bar (first 10 bars):")

for bar_idx in sorted(notes_per_bar.keys())[:10]:
    print(f"Bar {bar_idx}: {notes_per_bar[bar_idx]} notes")


# --- [ visualization ] ---
# for initial testing purposes
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

# configure axes and title
plt.gca().invert_yaxis()  # match SVG orientation (top is first)
plt.xlabel("Time (seconds)")
plt.ylabel("Y position in SVG")
plt.title("All Notes with Tick, Slide, Slide-Critical, Beat, and Bar Lines")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
# plt.show()


# FEATURE EXTRACTION

# --- [ synthetic beat-grid generation per bar ] ---
# q layers based on tsujino et. al's work
# l_4: # of notes in the 4th layer / # of notes total (quarter notes)
# l_8: # of notes in the 8th layer / # of notes total (eighth notes)
# etc.

# keep track of other layers as 'oth' (other)
LAYER_QS = [4, 8, 12, 16, 24, 32]

# ensuring we have a scale factor that is consistent with earlier code
# note_times_local has timestamps of notes in seconds, but a song might have different timing in data or outside of expected range (due to parsing)
# scale_local normalizes note times so all bars fit within the song's duration
def ensure_scale_and_times(note_times_local, song_duration_local):
    max_time = max(note_times_local) if len(note_times_local) > 0 else 1.0      # avoid zero-division
    scale_local = song_duration_local / max_time if max_time > 0 else 1.0       # avoid zero-division
    return scale_local, max_time

# building synthetic beat grids for one bar and for multiple q subdivisions
# helps with mapping notes to positions relative to the bar in a structured, standardized way
# returns a dict q -> list of absolute times (seconds) that represent the q-grid
# seconds_per_beat is unscaled here (60/bpm); scaling is applied within the function
# scale used to align timestamps to song_duration
def build_synthetic_grid_for_bar(bar_idx, beats_per_bar, seconds_per_beat, scale, song_duration, q_list = LAYER_QS):
    scaled_spb = seconds_per_beat * scale       # seconds per beat after scaling to song_duration
    bar_duration = beats_per_bar * scaled_spb   # duration of the bar in seconds
    bar_start_time = bar_idx * bar_duration     # absolute start time of the bar in seconds

    grids = {}
    for q in q_list:
        interval = bar_duration / q                                         # interval between grid points in seconds
        grids[q] = [bar_start_time + k * interval for k in range(q + 1)]    # include endpoints
    return grids                                                            # dict q -> [times]

# building synthetic grids for all bars in the entire chart, bar by bar
# returns a dict bar_idx -> { q -> [times] }
def build_synthetic_grid_all_bars(num_bars, beats_per_bar, seconds_per_beat, scale, song_duration, q_list=LAYER_QS):
    all_grids = {}
    
    # call function for every bar
    for b in range(num_bars):
        all_grids[b] = build_synthetic_grid_for_bar( 
            b, beats_per_bar, seconds_per_beat, scale, song_duration, q_list
        )
    return all_grids


# --- [ per note layer classification into q-layers (4, 8, 12, 16, 24, 32) or oth ] ---
# classify a single note into a q-layer (subdivison) using the synthetic grid
# bar_idx: integer bar index
# seconds_per_beat: 60 / bpm (unscaled)
# scale: scaling used to align timestamps to song_duration
# tol_fraction: acceptable fraction of subdivision interval to consider 'aligned'; can be adjusted for strictness
# returns (4/8/12/...) or 'oth'
def classify_note_layer_for_single(note_time, bar_idx, beats_per_bar, seconds_per_beat, scale, q_list = LAYER_QS, tol_fraction=0.15):
    scaled_spb = seconds_per_beat * scale       # seconds per beat after scaling to song_duration
    bar_duration = beats_per_bar * scaled_spb   # duration of the bar in seconds
    bar_start = bar_idx * bar_duration          # absolute start time of the bar in seconds
    local = note_time - bar_start               # note position inside bar in seconds

    # robust to notes falling slightly outside due to numeric round/truncation
    if local < -1e-8 or local > bar_duration + 1e-8:
        # note is outside standard bar window - allow small tolerance, but still classify against nearest bar subdivision
        local_clamped = max(0.0, min(local, bar_duration))
    else:
        # note is within standard bar window
        local_clamped = local

    # try assigning each note to each subdivison, from smallest q -> largest q
    for q in sorted(q_list, reverse=False):
        interval = bar_duration / q                 # interval between grid points in seconds
        idx = int(round(local_clamped / interval))  # index of nearest grid point
        nearest_time = idx * interval               # nearest grid point time in seconds
        delta = abs(local_clamped - nearest_time)   # difference from note local time to nearest grid point
        if delta <= tol_fraction * interval:        # within tolerance
            return q
    
        
    # if no q matched, return 'oth'
    return 'oth'


# classify all notes into layers
# wrapper to classify an entire list of notes into layers, using our synthetic grid functions

# take in note_times_list (list of timestamps in seconds, scaled to song_duration), note_bars_list (tuples (note_attr, bar_idx) in same order as note_times_list), 
# bars_list (list of (top_y, bottom_y)), beats_per_bar, bpm, song_duration
# returns assignments (list of same length as notes with values q or 'oth'), counts of q occurences, ratios (dict of l_q ratios for q in q_list and 'oth')
def classify_notes_layers(note_times_list, note_bars_list, bars_list, beats_per_bar, bpm, song_duration, q_list = LAYER_QS, tol_fraction=0.15):
    assert len(note_times_list) == len(note_bars_list), "note_times and note_bars must be same length"      # check
    seconds_per_beat = 60.0 / bpm                                                                           # unscaled seconds per beat
    
    # compute scale from note_times -> song_duration for consistency
    scale, max_time = ensure_scale_and_times(note_times_list, song_duration)

    assignments = []
    for i, nt in enumerate(note_times_list):
        note_attr, bar_idx = note_bars_list[i]  # get bar index for this note
        
        # if bar_idx out of range, clamp to nearest valid bar index
        if bar_idx < 0:
            bar_idx = 0
        if bar_idx >= len(bars_list):
            bar_idx = len(bars_list) - 1

        # classify note into a q-layer
        q = classify_note_layer_for_single(nt, bar_idx, beats_per_bar, seconds_per_beat, scale, q_list=q_list, tol_fraction=tol_fraction)
        assignments.append(q)

    # compute counts and ratios
    counts = Counter(assignments)
    total_notes = len(assignments)
    ratios = {}
    for q in q_list:
        ratios[f"l_{q}"] = counts.get(q, 0) / total_notes
    ratios["l_oth"] = counts.get('oth', 0) / total_notes

    return assignments, counts, ratios

# function to compute and print all layer classification results
# mainly a testing/debugging wrapper function
# lots in layer 4 due to measure start, beat subdivisions, grid subdivisions, 1/12, 1/24, 1/48 mini-grid, sometimes helper grid from editor exports
def compute_and_print_layers(note_times, note_bars, bars, bpm, song_duration, beats_per_bar=4, tol_fraction=0.15):
    # classify notes into layers
    assignments, counts, ratios = classify_notes_layers(
        note_times_list=note_times,
        note_bars_list=note_bars,
        bars_list=bars,
        beats_per_bar=beats_per_bar,
        bpm=bpm,
        song_duration=song_duration,
        q_list=LAYER_QS,
        tol_fraction=tol_fraction
    )
    
    # print results
    print("*" * 40)
    print("Layer counts (sample):", counts.most_common()[:10])
    print("Layer ratios:")
    for q in sorted(LAYER_QS):
        print(f"  l_{q}: {ratios[f'l_{q}']:.4f}")
    print(f"  l_oth: {ratios['l_oth']:.4f}")
    
    return assignments, counts, ratios

# use note_times, note_bars, bars, bpm, song_duration:
# note_times from earlier timestamp calculation
# note_bars from earlier mapping
# bars from earlier bar construction
# bpm and song_duration from metadata

# classify notes into layers
assignments, counts, ratios = compute_and_print_layers(
    note_times=note_times,
    note_bars=note_bars,
    bars=bars,
    bpm=bpm,
    song_duration=song_duration
)    

# --- [ notes per second features ] ---
# computers notes-per-second statistics for the chart

# note_times: list of floats (seconds)
# song_duration: float seconds
# bin_size: seconds per bin (1.0 for 'per-second')
# returns: dict {n_l (minimum), n_m (maximum), n_mu (mean), n_var (variance), counts_array (array of counts per bin)}
def compute_notes_per_second(note_times, song_duration, bin_size=1.0):
    n_bins = int(np.ceil(song_duration / bin_size))
    counts = np.zeros(n_bins, dtype=int)
    
    # count notes in each bin
    for t in note_times:
        idx = int(np.floor(min(max(t, 0.0), song_duration - 1e-9) / bin_size))
        counts[idx] += 1
        
    # compute statistics
    n_l = int(counts.min())
    n_m = int(counts.max())
    n_mu = float(counts.mean())
    n_var = float(counts.var(ddof=0))
    
    return {"n_l": n_l, "n_m": n_m, "n_mu": n_mu, "n_var": n_var, "counts": counts}


# --- [ layer ratios ] ---
# compute layer ratios from assigned to each q-layer
# assignments: list of q-layer assignments (4/8/12/16/24/32 or 'oth')
# q_list: list of q-layers to compute ratios for
def compute_layer_ratios(assignments, q_list=[4,8,12,16,24,32]):
    total = len(assignments) if len(assignments) > 0 else 1
    count = Counter(assignments)
    ratios = {}
    
    # compute ratios
    for q in q_list:
        ratios[f"l_{q}"] = count.get(q, 0) / total
    ratios["l_oth"] = count.get('oth', 0) / total
    return ratios

# --- [ tempo features ] ---
# compute temp features for a song

# bpm_events: list of (time_sec, bpm_value) if you have tempo changes (else None) - uses single bpm_val if None
# beat_count: total number of beats in the chart (optional)
# song_duration: seconds
# returns: dict with t_l, t_s, t_f, t_mu (where t_mu = (#beats / length_sec) * 60)
def compute_tempo_features(bpm_val=None, bpm_events=None, beat_count=None, song_duration=None):
    
    # bpm_events: assume sorted by time
    # compute time-weighted coverage per tempo to get t_l (dominant), t_s, t_f
    if bpm_events and len(bpm_events) > 0:
        coverage = defaultdict(float)
        for i, (t0, bpmv) in enumerate(bpm_events):
            t1 = bpm_events[i+1][0] if i+1 < len(bpm_events) else song_duration
            dur = max(0.0, t1 - t0)
            coverage[bpmv] += dur
            
        # dominant/main tempo
        dominant = max(coverage.items(), key=lambda x: x[1])[0]
        t_l = int(round(dominant))
        t_s = int(round(min(coverage.keys())))
        t_f = int(round(max(coverage.keys())))
        
    # else use single bpm_val
    else:
        if bpm_val is None:
            raise ValueError("No bpm provided for tempo calculation")
        
        # single tempo
        t_l = int(round(bpm_val))
        t_s = t_l
        t_f = t_l

    # compute t_mu if beat_count and song_duration provided
    t_mu = None
    if beat_count is not None and song_duration:
        t_mu = (beat_count / song_duration) * 60.0

    return {"t_l": t_l, "t_s": t_s, "t_f": t_f, "t_mu": t_mu}

# compute all of our selected song and chart features with different options available!

# exclude_ticks: if True, only includes playable notes (slide, slide-critical) - for testing purposes
# include_tick_stats: if True, computes ticks-per-second separately - for testing purposes
# returns: feats dict, assignments list, optional tick_feats dict
def extract_basic_features(note_times, note_bars, bars, bpm_val, song_duration_val, 
                           note_count_csv=None, beats_per_bar=4, q_list=[4,8,12,16,24,32], 
                           tol_fraction=0.15, exclude_ticks=True, include_tick_stats=False, tick_clip_max=100):

    # fliter notes if excluding tick-lines
    if exclude_ticks:
        filtered_note_times = []
        filtered_note_bars = []
        
        # filter out tick-line notes
        for i, (note_attr, bar_idx) in enumerate(note_bars):
            if note_attr.get('class') not in ['tick-line']:
                filtered_note_times.append(note_times[i])
                filtered_note_bars.append((note_attr, bar_idx))
                
    # else use all notes
    else:
        filtered_note_times = note_times
        filtered_note_bars = note_bars

    # classify notes into layers
    assignments, counts, ratios = classify_notes_layers(
        note_times_list=filtered_note_times,
        note_bars_list=filtered_note_bars,
        bars_list=bars,
        beats_per_bar=beats_per_bar,
        bpm=bpm_val,
        song_duration=song_duration_val,
        q_list=q_list,
        tol_fraction=tol_fraction
    )
    
    # compute notes-per-second (note density) features
    feats_density = compute_notes_per_second(filtered_note_times, song_duration_val)
    if note_count_csv:
        scale_factor = note_count_csv / sum(feats_density['counts'])
        feats_density['counts'] = feats_density['counts'] * scale_factor
    
    # compute tempo features
    bpm_events = None
    beat_count = None
    tempo_feats = compute_tempo_features(
        bpm_val=bpm_val,
        bpm_events=bpm_events,
        beat_count=beat_count,
        song_duration=song_duration_val
    )
    
    # compute optional tick-line stats
    tick_feats = None
    if include_tick_stats:
        tick_times = [
            note_times[i]
            for i, (note_attr, _) in enumerate(note_bars)
            if note_attr.get('class') == 'tick-line'
        ]
        tick_feats = compute_notes_per_second(tick_times, song_duration_val)
        tick_feats['counts_clipped'] = np.clip(tick_feats['counts'], None, tick_clip_max)
        tick_feats['n_m_clipped'] = int(tick_feats['counts_clipped'].max())
    
    # compile all features for pca and kmeans later
    feats = {}
    feats.update({k: feats_density[k] for k in ("n_l","n_m","n_mu","n_var")})
    feats.update(ratios)
    feats.update(tempo_feats)
    feats["total_notes_svg"] = len(filtered_note_times)
    if note_count_csv is not None:
        feats["total_notes_csv"] = note_count_csv
    feats["notes_mapped_by_beatlines"] = globals().get('no_fallback_count', None)
    feats["notes_mapped_fallback"] = globals().get('fallback_count', None)
    
    # print information (debugging)
    print("*" * 40)
    print(f"Total notes (SVG): {feats['total_notes_svg']}")
    if "total_notes_csv" in feats:
        print(f"Total notes (CSV): {feats['total_notes_csv']}")
    print(f"Notes/sec min/max/mean/var: {feats['n_l']}, {feats['n_m']}, {feats['n_mu']:.3f}, {feats['n_var']:.3f}")
    print("Layer ratios:")
    for q in q_list:
        print(f"  l_{q}: {feats[f'l_{q}']:.4f}")
    print(f"  l_oth: {feats['l_oth']:.4f}")
    print("Tempo (t_l, t_s, t_f, t_mu):", (feats['t_l'], feats['t_s'], feats['t_f'], feats['t_mu']))
    
    if tick_feats:
        print("\n--- Tick Line Stats (per second, clipped) ---")
        print(f"Ticks/sec min/max/mean/var: {tick_feats['counts_clipped'].min()}, "
              f"{tick_feats['n_m_clipped']}, "
              f"{tick_feats['counts_clipped'].mean():.3f}, "
              f"{tick_feats['counts_clipped'].var():.3f}")
    
    return feats, assignments, tick_feats

# calls our feature extraction function
feats_filtered, assignments_filtered, tick_feats = extract_basic_features(
    note_times=note_times,
    note_bars=note_bars,
    bars=bars,
    bpm_val=bpm,
    song_duration_val=song_duration,
    note_count_csv=note_count_csv,          # optional
    beats_per_bar=4,
    q_list=[4,8,12,16,24,32],
    tol_fraction=0.15,
    exclude_ticks=False,                    # include all notes
    include_tick_stats=True,                # compute tick line stats
    tick_clip_max=100                       # clip extreme ticks/sec to prevent outliers
)

# --- [ debugging checks ] ---
# verify layer ratios sum to total notes
# layer_ratios_filtered = compute_layer_ratios(assignments_filtered)
# total_layer_notes_filtered = sum(layer_ratios_filtered[q] * len(assignments_filtered) for q in layer_ratios_filtered)
# print(f"Layer counts sum: {total_layer_notes_filtered}, CSV note count: {note_count_csv}")

# # counts per second from your current note_times
# feats_density = compute_notes_per_second(note_times, song_duration)
# counts = feats_density['counts']

# # sort descending to see max first
# sorted_counts = sorted(counts, reverse=True)
# print("Top 20 tick counts per second:")
# for i, c in enumerate(sorted_counts[:20]):
#     print(f"{i+1}: {c}")