import glob
import os
from svgpathtools import svg2paths
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import mplcursors


# ----------------------------- [ HELPER FUNCTIONS ] -----------------------------

# parameters:   attr - dictionary of SVG element attributes, using y1 for lines and y for shapes
# returns:      vertical position (Y) of the SVG element
def get_y(attr):
    return float(attr.get('y1', attr.get('y', 0)))


# purpose:      load song metadata CSV
def load_song_metadata(csv_path):
    return pd.read_csv(csv_path)


# parameters:   svg_filename - string filename of the SVG chart (songID_difficulty.svg)
# purpose:      extract song metadata from filename and main song metadata dataframe
# returns:      bpm, duration_seconds, note_count, difficulty_level, song_title_eng, song_title_jpn
def get_song_metadata_from_filename(svg_filename, song_metadata_df):
    base = os.path.basename(svg_filename)
    
    # extract song ID and difficulty from filename
    song_id_str, difficulty_ext = base.split("_", 1)
    difficulty = difficulty_ext.replace(".svg", "")
    song_id = int(song_id_str)

    # match song ID and difficulty in metadata dataframe
    row = song_metadata_df[
        (song_metadata_df['song_id'] == song_id) &
        (song_metadata_df['difficulty'].str.lower() == difficulty.lower())
    ]

    # error handling if no match found
    if row.empty:
        raise ValueError(f"No metadata found for song_id={song_id}, difficulty={difficulty}")

    # read in csv file values
    bpm = parse_bpm(row['bpm'].values[0], skip_ranges=False)
    if bpm is None:
        print(f"Skipping chart with complex BPM: {svg_filename}")
        return None
    
    if bpm is None:
        print("Skipping chart with complex BPM:", svg_filename)
    
    duration_seconds = float(row['playback_time_seconds'].values[0])
    note_count = int(row['note_count'].values[0])
    difficulty_level = int(row['difficulty_level'].values[0])  # numeric level like 32
    song_title_eng = row['song_title_eng'].values[0]
    song_title_jpn = row['song_title_jpn'].values[0]

    return bpm, duration_seconds, note_count, difficulty_level, song_title_eng, song_title_jpn


# parameters:   bpm_str - string representing BPM
# purpose:      parse BPM string into a float value to handle ranges and parentheses (ex: "150-170", "92 (184)")
#               takes the average for ranges, or the first number if parentheses
#               # option to skip special gimmick charts
#               TODO: handle different BPM changes more robustly!
# returns:      bpm as float                    
def parse_bpm(bpm_str, skip_ranges=True):    
    bpm_str = str(bpm_str).strip()

    # skip special charts if option enabled
    if skip_ranges and ('-' in bpm_str or '(' in bpm_str):
        return None

    # range case
    if '-' in bpm_str:
        parts = bpm_str.split('-')
        try:
            return float(sum(map(float, parts)) / len(parts))  # average of range
        except ValueError:
            return float(parts[0])

    # parentheses case
    elif '(' in bpm_str and ')' in bpm_str:
        main_part = bpm_str.split('(')[0].strip()
        try:
            return float(main_part)
        except ValueError:
            paren_part = bpm_str.split('(')[1].split(')')[0].strip()
            return float(paren_part)

    else:
        return float(bpm_str)


# parameters:   attributes - list of SVG element attribute dictionaries
# purpose:      filter SVG elements by class
# returns:      lane_lines, bar_lines, beat_lines, tick_lines, playable_notes
def filter_svg_elements(attributes):
    lane_lines  = [attr for attr in attributes if attr.get('class') == 'lane-line']
    bar_lines   = [attr for attr in attributes if attr.get('class') == 'bar-line']
    beat_lines  = [attr for attr in attributes if attr.get('class') == 'beat-line']
    tick_lines  = [attr for attr in attributes if attr.get('class') == 'tick-line']
    notes       = [attr for attr in attributes if attr.get('class') in ['slide', 'slide-critical']]
    
    return lane_lines, bar_lines, beat_lines, tick_lines, notes


# ----------------------------- [ NOTES & TIMING FUNCTIONS ] -----------------------------

# parameters:   notes, tick_lines, bar_lines, beat_lines
# purpose:      combine slide notes and tick-lines, then sort all relevant elements by descending Y (earlier in song first)
# returns:      all_notes, bar_lines, beat_lines
def combine_and_sort_notes(notes, tick_lines, bar_lines, beat_lines):
    all_notes = notes + tick_lines
    all_notes.sort(key=get_y, reverse=True)
    bar_lines.sort(key=get_y, reverse=True)
    beat_lines.sort(key=get_y, reverse=True)
    return all_notes, bar_lines, beat_lines


# parameters:   bar_lines
# purpose:      remove duplicate bar-lines at the same Y position
# returns:      unique_bar_lines
def remove_duplicate_bar_lines(bar_lines):
    unique_bar_lines = []
    seen_y = set()
    for b in bar_lines:
        y = get_y(b)
        if y not in seen_y:
            unique_bar_lines.append(b)
            seen_y.add(y)
    return unique_bar_lines


# parameters:   bar_lines, all_notes
# purpose:      create bars defined by top and bottom Y coordinates from bar-lines; extend first/last bars to include all notes
# returns:      bars (list of (top, bottom) tuples), min_y, max_y
def create_bars(bar_lines, all_notes):
    bars = []
    for i in range(len(bar_lines)-1):
        top = get_y(bar_lines[i])
        bottom = get_y(bar_lines[i+1])
        if top == bottom:
            bottom = top - 1e-6
        bars.append((top, bottom))

    min_y = min(get_y(n) for n in all_notes)
    max_y = max(get_y(n) for n in all_notes)
    if max_y > bars[0][0]:
        bars[0] = (max_y, bars[0][1])
    if min_y < bars[-1][1]:
        bars[-1] = (bars[-1][0], min_y)

    return bars, min_y, max_y


# parameters:   all_notes, bars
# purpose:      map each note to its corresponding bar index based on Y position
# returns:      note_bars (list of (note_attr, bar_index) tuples)
def map_notes_to_bars(all_notes, bars):
    note_bars = []
    for note in all_notes:
        y = get_y(note)
        bar_idx = next((i for i, (top, bottom) in enumerate(bars) if bottom <= y <= top), len(bars)-1)
        note_bars.append((note, bar_idx))
    return note_bars


# parameters:   beat_lines, bars, min_y, max_y
# purpose:      filter beat-lines within notes' Y range, then map beats to closest bar indices
# returns:      bar_beats (dict mapping bar_index to list of beat Y positions
def filter_and_map_beats(beat_lines, bars, min_y, max_y):
    filtered_beats = [b for b in beat_lines if min_y <= get_y(b) <= max_y]
    
    bar_beats = defaultdict(list)
    for beat in filtered_beats:
        y = get_y(beat)
        closest_bar_idx = min(
            range(len(bars)),
            key=lambda i: abs((bars[i][0] + bars[i][1])/2 - y)
        )
        if y not in bar_beats[closest_bar_idx]:
            bar_beats[closest_bar_idx].append(y)

    for bar_idx in bar_beats:
        bar_beats[bar_idx].sort(reverse=True)
        if len(bar_beats[bar_idx]) == 1:
            y = bar_beats[bar_idx][0]
            bar_beats[bar_idx] = [y + 1e-6, y - 1e-6]

    return bar_beats


# parameters:   note_bars, bars, bar_beats, bpm, song_duration, beats_per_bar
# purpose:      calculate timestamp in seconds for each note using beat-lines, BPM, and bar positions; 
#               uses linear mapping if beat-lines are missing; scales to song duration
# returns:      note_times (list of timestamps), no_fallback_count, fallback_count
def calculate_note_timestamps(note_bars, bars, bar_beats, bpm, song_duration, beats_per_bar=4):
    seconds_per_beat = 60 / bpm
    epsilon = 1e-6
    note_times = []
    no_fallback_count = 0
    fallback_count = 0

    for note, bar_idx in note_bars:
        note_y = get_y(note)
        bar_top, bar_bottom = bars[bar_idx]
        bar_height = bar_top - bar_bottom
        if bar_height == 0:
            bar_height = 1e-6

        beats_in_bar = bar_beats.get(bar_idx, [])
        assigned = False

        if beats_in_bar and len(beats_in_bar) > 1:
            for i in range(len(beats_in_bar)-1):
                y_top_beat = beats_in_bar[i]
                y_bottom_beat = beats_in_bar[i+1]
                beat_height = y_top_beat - y_bottom_beat
                if beat_height == 0:
                    beat_height = 1e-6

                if y_bottom_beat - epsilon <= note_y <= y_top_beat + epsilon:
                    fraction = (y_top_beat - note_y) / beat_height
                    timestamp = (
                        bar_idx * beats_per_bar * seconds_per_beat +
                        i * seconds_per_beat +
                        fraction * seconds_per_beat
                    )
                    note_times.append(timestamp)
                    assigned = True
                    no_fallback_count += 1
                    break

        if not assigned:
            fraction = (bar_top - note_y) / bar_height
            timestamp = bar_idx * beats_per_bar * seconds_per_beat + fraction * beats_per_bar * seconds_per_beat
            note_times.append(timestamp)
            fallback_count += 1

    max_time = max(note_times) if note_times else 1
    scale = song_duration / max_time if max_time > 0 else 1
    note_times = [t * scale for t in note_times]

    return note_times, no_fallback_count, fallback_count


# ----------------------------- [ SYNTHETIC BEAT GRID ] -----------------------------

# q-layers to detect
LAYER_QS = [4, 8, 12, 16, 24, 32]

# parameters:   note_times_local, song_duration_local
# purpose:      ensure scale factor and max time are computed
# returns:      scale_local, max_time
def _ensure_scale_and_times(note_times_local, song_duration_local):
    max_time = max(note_times_local) if len(note_times_local) > 0 else 1.0
    scale_local = song_duration_local / max_time if max_time > 0 else 1.0
    return scale_local, max_time


# parameters:   bar_idx, beats_per_bar, seconds_per_beat, scale, song_duration, q_list
# purpose:      build synthetic beat grid for a single bar at q-layers
# returns:      grids (dict mapping q to list of timestamps)
def build_synthetic_grid_for_bar(bar_idx, beats_per_bar, seconds_per_beat, scale, song_duration, q_list=LAYER_QS):
    scaled_spb = seconds_per_beat * scale
    bar_duration = beats_per_bar * scaled_spb
    bar_start_time = bar_idx * bar_duration

    grids = {}
    for q in q_list:
        interval = bar_duration / q
        grids[q] = [bar_start_time + k * interval for k in range(q + 1)]
    return grids


# parameters:   num_bars, beats_per_bar, seconds_per_beat, scale, song_duration, q_list
# purpose:      build synthetic beat grids for all bars at q-layers
# returns:      all_grids (dict mapping bar_idx to grids dict)
def build_synthetic_grid_all_bars(num_bars, beats_per_bar, seconds_per_beat, scale, song_duration, q_list=LAYER_QS):
    all_grids = {}
    for b in range(num_bars):
        all_grids[b] = build_synthetic_grid_for_bar(b, beats_per_bar, seconds_per_beat, scale, song_duration, q_list)
    return all_grids


# parameters:   note_time, bar_idx, beats_per_bar, seconds_per_beat, scale, q_list, tol_fraction
# purpose:      classify a single note into a q-layer based on proximity to synthetic grid points
# returns:      q-layer or 'oth' if no match
def classify_note_layer_for_single(note_time, bar_idx, beats_per_bar, seconds_per_beat, scale, q_list=LAYER_QS, tol_fraction=0.25):
    scaled_spb = seconds_per_beat * scale
    bar_duration = beats_per_bar * scaled_spb
    bar_start = bar_idx * bar_duration
    local = note_time - bar_start
    local_clamped = max(0.0, min(local, bar_duration))

    for q in sorted(q_list):
        interval = bar_duration / q
        idx = int(round(local_clamped / interval))
        nearest_time = idx * interval
        if abs(local_clamped - nearest_time) <= tol_fraction * interval:
            return q
    return 'oth'


# parameters:   note_times_list, note_bars_list, bars_list, beats_per_bar, bpm, song_duration, q_list, tol_fraction
# purpose:      classify all notes into q-layers based on synthetic grids
# returns:      assignments (list of q-layers), counts (Counter), ratios (dict
def classify_notes_layers(note_times_list, note_bars_list, bars_list, beats_per_bar, bpm, song_duration, q_list=LAYER_QS, tol_fraction=0.25):
    assert len(note_times_list) == len(note_bars_list)
    seconds_per_beat = 60.0 / bpm
    scale, _ = _ensure_scale_and_times(note_times_list, song_duration)

    assignments = []
    for i, nt in enumerate(note_times_list):
        note_attr, bar_idx = note_bars_list[i]
        bar_idx = min(max(bar_idx, 0), len(bars_list)-1)
        q = classify_note_layer_for_single(nt, bar_idx, beats_per_bar, seconds_per_beat, scale, q_list=q_list, tol_fraction=tol_fraction)
        assignments.append(q)

    counts = Counter(assignments)
    total_notes = len(assignments)
    ratios = {f"l_{q}": counts.get(q, 0)/total_notes for q in q_list}
    ratios["l_oth"] = counts.get('oth', 0)/total_notes
    return assignments, counts, ratios


# ----------------------------- [ NOTES PER SECOND ] -----------------------------

# parameters:   note_times, song_duration, bin_size
# purpose:      compute note density features
# returns:      dict with n_l, n_m, n_mu, n_var, counts

def compute_notes_per_second(note_times, song_duration, bin_size=1.0):
    n_bins = int(np.ceil(song_duration / bin_size))
    counts = np.zeros(n_bins, dtype=int)
    for t in note_times:
        idx = int(np.floor(min(max(t, 0.0), song_duration-1e-9)/bin_size))
        counts[idx] += 1
    n_l, n_m = int(counts.min()), int(counts.max())
    n_mu, n_var = float(counts.mean()), float(counts.var(ddof=0))
    return {"n_l": n_l, "n_m": n_m, "n_mu": n_mu, "n_var": n_var, "counts": counts}

def compute_layer_ratios(assignments, q_list=LAYER_QS):
    total = len(assignments) if len(assignments) > 0 else 1
    cnt = Counter(assignments)
    ratios = {f"l_{q}": cnt.get(q,0)/total for q in q_list}
    ratios["l_oth"] = cnt.get('oth',0)/total
    return ratios


# ----------------------------- [ TEMPO ] -----------------------------

# parameters:   bpm_val, bpm_events, beat_count, song_duration
# purpose:      compute tempo features
# returns:      dict with t_l, t_s, t_f, t_mu
def compute_tempo_features(bpm_val=None, bpm_events=None, beat_count=None, song_duration=None):
    if bpm_events and len(bpm_events) > 0:
        coverage = defaultdict(float)
        for i, (t0, bpmv) in enumerate(bpm_events):
            t1 = bpm_events[i+1][0] if i+1 < len(bpm_events) else song_duration
            coverage[bpmv] += max(0.0, t1 - t0)
        dominant = max(coverage.items(), key=lambda x:x[1])[0]
        t_l = int(round(dominant))
        t_s = int(round(min(coverage.keys())))
        t_f = int(round(max(coverage.keys())))
    else:
        if bpm_val is None: raise ValueError("No bpm provided for tempo calculation")
        t_l = t_s = t_f = int(round(bpm_val))
    t_mu = (beat_count / song_duration) * 60.0 if beat_count is not None and song_duration else None
    return {"t_l": t_l, "t_s": t_s, "t_f": t_f, "t_mu": t_mu}


# ----------------------------- [ FULL FEATURE EXTRACTION ] -----------------------------

def extract_basic_features(note_times, note_bars, bars, bpm_val, song_duration_val, 
                           note_count_csv=None, beats_per_bar=4, q_list=LAYER_QS, tol_fraction=0.25, 
                           exclude_ticks=True, include_tick_stats=False, tick_clip_max=100):
 
    # filter notes if needed
    if exclude_ticks:
        filtered_note_times, filtered_note_bars = zip(*[(t, nb) for t, nb in zip(note_times, note_bars) if nb[0].get('class') not in ['tick-line']])
    else:
        filtered_note_times, filtered_note_bars = note_times, note_bars

    # q-layer classification features
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

    # note density features
    feats_density = compute_notes_per_second(filtered_note_times, song_duration_val)
    if note_count_csv:
        scale_factor = note_count_csv / sum(feats_density['counts'])
        feats_density['counts'] = feats_density['counts'] * scale_factor

    # tempo features
    tempo_feats = compute_tempo_features(bpm_val=bpm_val, bpm_events=None, beat_count=None, song_duration=song_duration_val)

    # tick line stats
    tick_feats = None
    if include_tick_stats:
        tick_times = [note_times[i] for i, (note_attr, _) in enumerate(note_bars) if note_attr.get('class')=='tick-line']
        tick_feats = compute_notes_per_second(tick_times, song_duration_val)
        tick_feats['counts_clipped'] = np.clip(tick_feats['counts'], None, tick_clip_max)
        tick_feats['n_m_clipped'] = int(tick_feats['counts_clipped'].max())

    # combine and store features
    feats = {**{k: feats_density[k] for k in ("n_l","n_m","n_mu","n_var")},
             **ratios,
             **tempo_feats,
             "total_notes_svg": len(filtered_note_times)}
    if note_count_csv is not None:
        feats["total_notes_csv"] = note_count_csv
    feats["notes_mapped_by_beatlines"] = globals().get('no_fallback_count', None)
    feats["notes_mapped_fallback"] = globals().get('fallback_count', None)

    return feats, assignments, tick_feats


# ----------------------------- [ PROCESS CHARTS + PCA + CLUSTERING ] -----------------------------
if __name__ == "__main__":
    # load song metadata
    csv_path = os.path.join(os.path.dirname(__file__), '../data/song_metadata.csv')
    song_metadata = load_song_metadata(csv_path)

    charts_folder = os.path.join(os.path.dirname(__file__), '../data/charts/')
    svg_files = [f for f in os.listdir(charts_folder) if f.endswith(".svg")]

    charts = []
    feats_filtered_list = []
    chart_names = []

    # read and process each SVG chart
    for svg_file in svg_files:
        svg_path = os.path.join(charts_folder, svg_file)
        try:
            paths, attributes = svg2paths(svg_path)
            
            # get metadata
            metadata = get_song_metadata_from_filename(svg_path, song_metadata)
            
            # skip charts with complex BPM
            if metadata is None:
                continue        

            bpm, song_duration, note_count_csv, difficulty_level, song_title_eng, song_title_jpn = metadata
            difficulty_name = svg_file.split("_")[1].replace(".svg", "")
            
            # filter SVG elements
            lane_lines, bar_lines, beat_lines, tick_lines, notes = filter_svg_elements(attributes)
            all_notes, bar_lines, beat_lines = combine_and_sort_notes(notes, tick_lines, bar_lines, beat_lines)
            bar_lines = remove_duplicate_bar_lines(bar_lines)
            bars, min_y, max_y = create_bars(bar_lines, all_notes)
            note_bars = map_notes_to_bars(all_notes, bars)
            bar_beats = filter_and_map_beats(beat_lines, bars, min_y, max_y)
            note_times, no_fallback_count, fallback_count = calculate_note_timestamps(note_bars, bars, bar_beats, bpm, song_duration)

            # extract features
            feats_filtered, assignments_filtered, tick_feats = extract_basic_features(
                note_times=note_times,
                note_bars=note_bars,
                bars=bars,
                bpm_val=bpm,
                song_duration_val=song_duration,
                note_count_csv=note_count_csv,
                beats_per_bar=4,
                q_list=LAYER_QS,
                tol_fraction=0.25,
                exclude_ticks=False,
                include_tick_stats=True,
                tick_clip_max=100
            )
        
            layer_ratios_filtered = compute_layer_ratios(assignments_filtered)

            # debug summary printout
            # print("*" * 40)
            # print(f"Chart: {svg_file}")
            # print(f"BPM: {bpm}, Duration: {song_duration}s, Note Count (CSV): {note_count_csv}")
            # print(f"Notes mapped by beatlines: {no_fallback_count}, fallback mappings: {fallback_count}")
            # print("Layer ratios:", layer_ratios_filtered)

            # store chart data in one place           
            charts.append({
                "filename": svg_file,
                "difficulty": difficulty_name,
                "difficulty_level": difficulty_level,
                "note_times": note_times,
                "note_bars": note_bars,
                "bars": bars,
                "bpm": bpm,
                "song_duration": song_duration,
                "song_title_eng": song_title_eng,
                "song_title_jpn": song_title_jpn,
                "features": feats_filtered,
                "assignments": assignments_filtered,
                "tick_feats": tick_feats,
                "no_fallback_count": no_fallback_count,
                "fallback_count": fallback_count,
                "layer_ratios": layer_ratios_filtered
            })
            
            feats_filtered_list.append(feats_filtered)
            chart_names.append(svg_file)

        # error handling
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")


    # collect all features into DataFrame
    all_feats = [chart['features'] for chart in charts]
    df_feats = pd.DataFrame(all_feats).fillna(0.0)


    # split features into groups
    n_cols = ['n_l', 'n_m', 'n_mu', 'n_var']
    l_cols = [f"l_{q}" for q in [4,8,12,16,24,32]] + ['l_oth']
    t_cols = ['t_l', 't_s', 't_f']
    s_cols = ['total_notes_svg']
    
    if 'total_notes_csv' in df_feats.columns:
        s_cols.append('total_notes_csv')

    n_feats = df_feats[n_cols].values
    l_feats = df_feats[l_cols].values
    t_feats = df_feats[t_cols].values
    s_feats = df_feats[s_cols].values

    # standardize each group with optional weighting for layers
    scaler_n = StandardScaler(); n_scaled = scaler_n.fit_transform(n_feats)

    # weight layer features: downweight rare layers like l_8, l_12
    layer_weights = np.ones(len(l_cols))
    for rare_layer in ['l_8', 'l_12']:
        if rare_layer in l_cols:
            layer_weights[l_cols.index(rare_layer)] = 0.3               # reduce weight
    l_scaled = StandardScaler().fit_transform(l_feats) * layer_weights  # standardize then weight

    # l_scaled = StandardScaler().fit_transform(l_feats)                          # no weighting
    scaler_t = StandardScaler(); t_scaled = scaler_t.fit_transform(t_feats)
    scaler_s = StandardScaler(); s_scaled = scaler_s.fit_transform(s_feats)


    # ----------------------------- [ APPLY PCA PER GROUP ] -----------------------------
    pca_n = PCA(); pc_n = pca_n.fit_transform(n_scaled)
    pca_l = PCA(); pc_l = pca_l.fit_transform(l_scaled)
    pca_t = PCA(); pc_t = pca_t.fit_transform(t_scaled)

    # scale / size
    pca_s = PCA()
    pc_s_full = pca_s.fit_transform(s_scaled)
    cum_var = np.cumsum(pca_s.explained_variance_ratio_)
    n_pc_s = np.searchsorted(cum_var, 0.9) + 1
    pc_s = pc_s_full[:, :n_pc_s]

    # debug print explained variance
    def print_pca_variance(pca, group_name):
        print(f"--- PCA explained variance: {group_name} ---")
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.1f}%)")
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        print("Cumulative variance:", cum_var)
        print()

    print_pca_variance(pca_n, "Note Density (n)")
    print_pca_variance(pca_l, "Layer Ratios (l)")
    print_pca_variance(pca_t, "Tempo (t)")
    print_pca_variance(pca_s, "Scale / Size (s)")

    # final feature matrix by concatenating PCs from each group
    # X_final = np.hstack([pc_n, pc_l, pc_t])  # excluding pc_s
    X_final= np.hstack([pc_n, pc_l, pc_t, pc_s])  # including pc_s
    print("Final feature matrix shape:", X_final.shape)


    # ----------------------------- [ CLUSTERING ] -----------------------------
    # 34 clusters due to difficulty levels from 5 to 38
    # 2, 3, 4, 5, 6, 8, 10, 12, 16, 20 to test smaller vals
    # cluster_counts = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 34]  # k values to evaluate
    
    cluster_counts = range(1, 35)  # k values to evaluate

    # pca for visualization (2D)
    pca_final = PCA(n_components=2)
    X_vis = pca_final.fit_transform(X_final)

    # performing clustering for each k
    for k in cluster_counts:

        # kmeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_final)

        output_folder = os.path.join(os.path.dirname(__file__), '../data/outputs')
        os.makedirs(output_folder, exist_ok=True)

        # saving relevant results for later analysis
        results_df = pd.DataFrame({
            "chart": [chart["filename"] for chart in charts],
            "difficulty": [chart["difficulty"] for chart in charts],
            "difficulty_level": [chart["difficulty_level"] for chart in charts],
            "song_title_eng": [chart["song_title_eng"] for chart in charts],
            "song_title_jpn": [chart["song_title_jpn"] for chart in charts],
            "cluster": cluster_labels,
            "total_notes": df_feats.get("total_notes_csv", df_feats["total_notes_svg"]),
            "bpm": [chart["bpm"] for chart in charts],
            "duration_seconds": [chart["song_duration"] for chart in charts],
            "pca_x": X_vis[:,0],
            "pca_y": X_vis[:,1],
        })

        # sort by cluster, difficulty_level, total_notes
        results_df_sorted = results_df.sort_values(
            by=["cluster", "difficulty_level", "total_notes"],
            ascending=[True, True, True]
        )

        # saving file
        output_path = os.path.join(output_folder, f"cluster_results_k{k}_sorted.csv")
        results_df_sorted.to_csv(output_path, index=False)

        # plotting PCA with clusters
        # plt.figure(figsize=(10,7))
        # scatter = plt.scatter(X_vis[:,0], X_vis[:,1], c=cluster_labels, cmap='tab10', s=80)
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.title(f'PCA of Charts with KMeans Clusters (k={k})')
        # plt.grid(True)

        # # interactive cursor for data points
        # cursor = mplcursors.cursor(scatter, hover=False)

        # @cursor.connect("add")
        # def on_add(sel):
        #     idx = sel.index
        #     sel.annotation.set_text(chart_names[idx])

        # handles, _ = scatter.legend_elements()
        # plt.legend(handles, [f"Cluster {i}" for i in range(k)], title="Clusters")
        # plt.show()