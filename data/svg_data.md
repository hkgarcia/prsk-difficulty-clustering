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
Information that must be derived and calulated from .SVG file componentsinclude: 

- **Number of notes per second**: Calculate from note timestamps. Divide charts into parts for each one second. The following features must be obtained when counting the number of notes on each part:
    - $n_l$: # of notes on the part which has the least notes
    - $n_m$: # of notes on the part which has the most notes
    - $n_\mu$: Average of # of notes through all parts
    - $n_\sigma^2$: Variance of # of notes through all parts

- **Note ratio for each beat layer:** Based on timing divisions. A set of timing obtained by dividing a bar into $q$ equal parts ($q>=4$) is defined as the "$q$th beat layer." Let the lowest layer which the timing where a given note belongs to the $q$th layer, the step should belong to $q$th layer and defined as the "$q$th step;" here, "the lowest" means least $q$. If some notes belong to the higher layer, the sequence of notes will be more difficult and complex (Tsujino et al).
    - $l_4$ = # of notes in 4th layer / # of all notes
    - $l_8$ = # of notes in 8th layer / # of all notes
    - $l_12$ = # of notes in 12th layer / # of all notes
    - $l_16$ = # of notes in 16th layer / # of all notes
    - $l_24$ = # of notes in 24th layer / # of all notes
    - $l_32$ = # of notes in 32th layer / # of all notes
    - $l_oth$ = 1 - $\sum\limits_{q\in4,8,12,16,24,32} l_q$

- **Frequency of specific note types**: Including slide, slide-critical, normal note, etc. Due to the format of .SVG files, there is no difference between a normal note and a flick note. More data must be derived.


## NOTES AND CONTEXT COLLECTED DURING DATA CLEANING

- **bar-line**
    - Five unique bar-lines (corresponds to the four bar-lines per lane)
    - Will keep **one bar line per y1** to represent measures when mapping notes
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '2670', 'y2': '2670'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '2011', 'y2': '2011'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '1351', 'y2': '1351'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '692', 'y2': '692'}
        - {'class': 'bar-line', 'x1': '40', 'x2': '232', 'y1': '32', 'y2': '32'}


- **beat-line**
    - 15 unique beats-lines exist
    - Will keep **one beat line per y1** to represent quarter notes when mapping notes
    - Should match expected beats per bar in song


- **tick-lines** 
    - Higher total of ticklines than note count due to tempo/BPM
    - Duplicate tickline classes exist (same path/attributes) due to overlapping slides, extended notes, etc.
    - Can possibly group tick lines by bar (y1) to track ticks per bar
