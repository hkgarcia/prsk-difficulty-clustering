# BREAKDOWN OF .SVG DATA

## UNIQUE CLASSES AND DESCRIPTIONS
| Class | Description |
| --- | --- |
| background | Formatting component |
| bar-count-flag | Background rectangles (for bar counts in chart). Contained in lanes, broken down by #'s |
| bar-line | Same as bar-count-flag, with the only difference being x1 (0 vs. 40 for bar-line). About one per musical measure |
| beat-line | Lines for timings within each bar/bar-count-flag. Quarter notes in a musical measure|
| decoration | Background elements or decorative paths (unclickable) |
| event-flag | Flags for events (skill activation, super fever, etc.) |
| lane | Large lanes in SVG, containing all chart components |
| lane-line | Breaks down each lane into sections (vertical lines), with each lane having 7 lines (or 6 section) |
| meta | Formatting component |
| meta-line | Formatting component |
| slide | Green slide notes |
| slide-critical | Yellow slide notes |
| speed-line | Markers for speed or tempo changes |
| tick-line | Tick lines for note timing. Variable x1 is either 16 or 32 |

## CLASS HIERARCHY AND RELATIONSHIPS
**LANES** consist of several **BARS** (bar-count-flag) that are separated by a **#.** Each **BAR** contains **BEAT LINES** (about 3 per bar) for separation. Breaking down the structure further, each **BAR** has **TICK-LINES** corresponding to the note and tick timings of a chart. 

Note that total note count can be lower than the total number of tick lines, as tick lines account for major "checks." For example, extended slide notes may correspond to several ticks depending on their arc or movement.

## GIVEN INFORMATION
- **Playable notes (from classes):** Slide and slide-critical. Tick-line, BPM, and playback can be used to find notes (mapped to timestamps).


## INFORMATION TO CALCUATE
Information that must be derived and calulated from .SVG file components include: 

- **Number of notes per second**: Calculate from note timestamps. Divide charts into parts for each one second. The following features must be obtained when counting the number of notes on each part:
    - $n_l$: # of notes on the part which has the least notes
    - $n_m$: # of notes on the part which has the most notes
    - $n_\mu$: Average of # of notes through all parts
    - $n_\sigma^2$: Variance of # of notes through all parts

- **Note ratio for each beat layer:** Based on timing divisions. A set of timing obtained by dividing a bar into $q$ equal parts ($q>=4$) is defined as the "$q$th beat layer." Let the lowest layer which the timing where a given note belongs to the $q$th layer, the step should belong to $q$th layer and defined as the "$q$th step;" here, "the lowest" means least $q$. If some notes belong to the higher layer, the sequence of notes will be more difficult and complex (Tsujino et al).
    - $l_4$ = # of notes in 4th layer / # of all notes
    - $l_8$ = # of notes in 8th layer / # of all notes
    - $l_{12}$ = # of notes in 12th layer / # of all notes
    - $l_{16}$ = # of notes in 16th layer / # of all notes
    - $l_{24}$ = # of notes in 24th layer / # of all notes
    - $l{32}$ = # of notes in 32th layer / # of all notes
    - $l_{oth}$ = 1 - $\sum\limits_{q\in4,8,12,16,24,32} l_q$

- **Frequency of specific note types**: Including slide, slide-critical, normal note, etc. Due to the format of .SVG files, there is no difference between a normal note and a flick note. More data must be derived.


## NOTES AND CONTEXT COLLECTED DURING DATA CLEANING

- **bar-line**
    - Represent measures when mapping notes
    - Unique [x1, x2] value are ONLY [40, 232], meaning we must use vertical checks

    - Five unique bar-lines (corresponds to the four bars per lane) that repeat in a cycle
    - We see our bar-line as a **measure**
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '2670', 'y2': '2670'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '2011', 'y2': '2011'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '1351', 'y2': '1351'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '692', 'y2': '692'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '32', 'y2': '32'}
    

- **beat-line**
    - Represent quarter notes when mapping notes
    - Unique [x1, x2] value are ONLY [40, 232], meaning we must use vertical checks
    - Should match expected beats per bar in song

    - 15 unique beats-lines exist that repeat in a cycle
    - We see these as **subdivisions of the bars**
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '2505', 'y2': '2505'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '2340', 'y2': '2340'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '2176', 'y2': '2176'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '1846', 'y2': '1846'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '1681', 'y2': '1681'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '1516', 'y2': '1516'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '1186', 'y2': '1186'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '1021', 'y2': '1021'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '856', 'y2': '856'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '527', 'y2': '527'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '362', 'y2': '362'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '197', 'y2': '197'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '-133', 'y2': '-133'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '-298', 'y2': '-298'}
        - {'class': 'beat-line', 'x1': '40', 'x2': '232', 'y1': '-463', 'y2': '-463'}


- **tick-lines** 
    - Unique [x1] values are [16, 32] and [x2] is [40]
    - Higher total of ticklines than note count due to tempo/BPM
    - Duplicate tickline classes exist (same path/attributes) due to overlapping slides, extended notes, etc.
        - Keep duplicates for timing accurancy and clustering - filter later

    - 157 unique Y-values (granular subdivisions), with each value representing the smallest unit in our chart
    - Cover the full Y-axis of the chart


## FULL CODE EXPLANATION / WALKTHROUGH (WIP)
1. Extract bar-line Y positions
- Parse SVG and find all elements that represent a bar line. Remove duplicates, sort them, and pair them as (top, bottom) bar boundaries. Each pair becomes a bar 

2. Extend first and last bars
- Since the SVG may be missing a bar line at exact edges of our chart (ensures every note falls inside a bar), we:
    - Extend the first bar downwards to cover early notes
    - Extend the last bar upwards to cover late notes

3. Assign each note to a bar (bar index)
- For each note, we:
    - Look at its Y position
    - Find the bar whose Y-range contains the note

4. Filter beat lines
- Beat lines in the SVG can extend above or below visible notes, so we ensure to get the min and may y, keeping beat lines within this range

5. Map beat lines to bars
- We assign beat lines to a bar by:
    - Compute the center Y of each bar (midpoint), and for each beat line, pick the closest bar center
    - Then, we sort the beats ascending (earliest to latest) by Y and handle bars with only one beat

6. Calculate note time stamps
- Since we ahve only assigned notes bars, we now have to give them specific timestamps - bar numbers were only structural positions
- We use BPM + beat lines
    - Method A: Beat-line interpolation
        - Find two consecutive beat lines a note falls between
        - Compute its relative position between those beats (how far between the beats the note lies)
        - Convert that into time using BPM
            - More aaccurate since beats are the core of our charts!

    - Method B: Linear interpolation inside a bar
        - If a bar has zero beat lines or notes outside a beat-region, we approximate
            - Time = linear mapping from bar top to bar bottom
