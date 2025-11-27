import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

### CLUSTER ANALYSIS OF A CHOSEN K VALUE: CHOOSE FILE ACCORDINGLY ###
# find k from filename
file_name = 'cluster_results_k5_sorted.csv'     # CHANGE THIS FILE NAME AS NEEDED
match = re.search(r'k(\d+)', file_name)
k_val = int(match.group(1)) if match else None

file_path = os.path.join(os.path.dirname(__file__), f'../../data/outputs/{file_name}')
df = pd.read_csv(file_path)

# setting up output directory
output_dir = os.path.join(os.path.dirname(__file__), '../../data/outputs')
os.makedirs(output_dir, exist_ok=True)

# ensuring correct order of difficulty categories for later visualization
difficulty_order = ["easy", "normal", "hard", "expert", "master", "append"]
df["difficulty"] = pd.Categorical(df["difficulty"], categories=difficulty_order, ordered=True)

# fixing title
df['title'] = df['song_title_eng'].fillna(df['song_title_jpn'])

# --- [ qualititative summary by cluster ] ---
# number of songs, unique difficulties, min/max/mean/variance difficulty_level, min/max/mean/variance total_notes
qual_summary = df.groupby('cluster').agg(
    num_songs=('title', 'nunique'),
    difficulties=('difficulty', lambda x: ', '.join(sorted(x.unique()))),
    
    min_difficulty=('difficulty_level', 'min'),
    max_difficulty=('difficulty_level', 'max'),
    mean_difficulty=('difficulty_level', 'mean'),
    variance_difficulty=('difficulty_level', 'var'),
    
    min_notes=('total_notes', 'min'),
    max_notes=('total_notes', 'max'),
    mean_notes=('total_notes', 'mean'),
    variance_notes=('total_notes', 'var'),
).reset_index()

qual_summary.to_csv(os.path.join(output_dir, f"qualitative_analysis_k{k_val}.csv"), index=False)
print("Saved qualitative analysis in qualitative_analysis_k{0}.csv".format(k_val))

# --- [ condensed quantitative summary by cluster ] ---
# using a few chosen variables

# count of difficulty types per cluster
quant_diff_cluster = df.groupby('cluster')['difficulty'].value_counts().unstack(fill_value=0)
quant_diff_cluster.to_csv(os.path.join(output_dir, f"difficulty_analysis_k{k_val}.csv"))
print("Saved difficulty type analysis in difficulty_analysis_k{0}.csv".format(k_val))

# count of difficulty levels per cluster
quant_difflevel_cluster = df.groupby('cluster')['difficulty_level'].value_counts().unstack(fill_value=0)
quant_difflevel_cluster.to_csv(os.path.join(output_dir, f"difficulty_level_analysis_k{k_val}.csv"))
print("Saved difficulty level analysis in difficulty_level_analysis_k{0}.csv".format(k_val))

# count of total notes ranges per cluster
bins = [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, df['total_notes'].max() + 1]
labels = ['<200', '<400', '<600', '<800', '<1000', '<1200', '<1500', '<2000', '2000+']
df['note_range'] = pd.cut(df['total_notes'], bins=bins, labels=labels, right=False)

quant_notes_cluster = df.groupby('cluster')['note_range'].value_counts().unstack(fill_value=0)
quant_notes_cluster.to_csv(os.path.join(output_dir, f"note_analysis_k{k_val}.csv"))
print("Saved total notes range analysis.")
 
# --- [ plotting each qualititative summary! ] ---

# color mapping according to actual game colors for difficulties!
color_map = {
    "easy":   "#66de14",
    "normal": "#34bcef",
    "hard":   "#ffa903",
    "expert": "#ed4467",
    "master": "#bc33ee",
    "append": "#ff82c5"
}

# --- [ difficulty types ] ---
plt.figure(figsize=(12,6))
clusters = quant_diff_cluster.index
n_clusters = len(clusters)
categories = quant_diff_cluster.columns
bar_width = 0.8 / len(categories)

# plotting bars
for i, cat in enumerate(categories):
    values = quant_diff_cluster[cat]
    bars = plt.bar(clusters + i*bar_width, values, width=bar_width, label=str(cat), color=color_map.get(cat.lower()))

    # add labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text( bar.get_x() + bar.get_width()/2, height, f"{int(height)}", ha="center", va="bottom", fontsize=8 )

# details
plt.xlabel("Cluster (k)")
plt.ylabel("Count")
plt.title(f"Count of Difficulty Types per Cluster (k = {k_val})")
plt.xticks(clusters + bar_width*(len(categories)/2 - 0.5), clusters)
plt.legend(title="Difficulty", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- [ difficulty levels ] ---
plt.figure(figsize=(16,6))
clusters = quant_difflevel_cluster.index
categories = quant_difflevel_cluster.columns
bar_width = 0.8 / len(categories)

# plotting bars
for i, cat in enumerate(categories):
    values = quant_difflevel_cluster[cat]
    bars = plt.bar(clusters + i*bar_width, values, width=bar_width, label=str(cat))

    # add labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text( bar.get_x() + bar.get_width()/2, height, f"{int(height)}", ha="center", va="bottom", fontsize=8 )

# details
plt.xlabel("Cluster (k)")
plt.ylabel("Count")
plt.title(f"Count of Difficulty Levels per Cluster (k = {k_val})")
plt.xticks(clusters + bar_width*(len(categories)/2 - 0.5), clusters)
plt.legend(title="Difficulty Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# analyzing specific clusters
target_clusters = range(0, k_val)  # all clusters for k=k_val

for c in target_clusters:
    if c not in quant_difflevel_cluster.index:
        print(f"Cluster {c} not found, skipping.")
        continue

    plt.figure(figsize=(8,4))
    values = quant_difflevel_cluster.loc[c]

    # bar plot for a single cluster
    bars = plt.bar(values.index, values.values)

    # add numbers above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Difficulty Level")
    plt.ylabel("Count")
    plt.title(f"Difficulty Levels in Cluster {c}")
    plt.tight_layout()
    plt.show()


# --- [ total note ranges ] ---
plt.figure(figsize=(14,6))
clusters = quant_notes_cluster.index
categories = quant_notes_cluster.columns
bar_width = 0.8 / len(categories)

# create gradient of reds for color
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(categories)))

# plotting bars
for i, (cat, color) in enumerate(zip(categories, colors)):
    values = quant_notes_cluster[cat]
    bars = plt.bar(clusters + i*bar_width, values,
                   width=bar_width, label=str(cat), color=color)

    # add labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{int(height)}", ha="center", va="bottom", fontsize=8)
        
# details
plt.xlabel("Cluster (k)")
plt.ylabel("Count")
plt.title(f"Count of Total Notes Ranges per Cluster (k = {k_val})")
plt.xticks(clusters + bar_width*(len(categories)/2 - 0.5), clusters)
plt.legend(title="Note Range", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- [ difficulty types summary with percentages ] ---
# total charts
total_charts = len(df)
print(f"Total Chart Count: {total_charts}")

# overall counts
overall_counts = df['difficulty'].value_counts()
print("Overall Difficulty Breakdown:")
for diff, cnt in overall_counts.items():
    print(f"{diff}: {cnt} ({cnt/total_charts*100:.1f}%)")
print("*" * 40)

# per cluster breakdown: visualization and data for presentation and report
print("\nCluster Breakdown:")
for cluster_id, cluster_df in df.groupby('cluster'):
    cluster_total = len(cluster_df)
    print(f"Cluster {cluster_id} ({cluster_total} charts):")
    
    # difficulty type breakdown
    diff_counts = cluster_df['difficulty'].value_counts()
    for diff, cnt in diff_counts.items():
        print(f"{diff}: {cnt} charts ({cnt/cluster_total*100:.1f}% of cluster)")
    
    # difficulty level breakdown
    values = quant_difflevel_cluster.loc[cluster_id]
    values = values[values > 0].sort_values(ascending=False).head(5)
    top5_total = values.sum()
    
    # print top 5 difficulty levels with counts and percentages (for data analysis and presentation)
    print(f"\nTop 5 Difficulty Levels in Cluster {cluster_id}:")
    for level, cnt in values.items():
        pct_cluster = cnt / top5_total * 100
        pct_total = cnt / len(df) * 100
        print(f"Level {level}: {cnt} charts ({pct_total:.1f}%)")
    print("*" * 40)
    
    # pie chart for difficulty types in this cluster
    colors = [color_map.get(d.lower(), "#cccccc") for d in diff_counts.index]
    plt.figure(figsize=(6,6))
    plt.pie(diff_counts.values, labels=diff_counts.index,
            autopct=lambda p: f"{p:.1f}% ({int(round(p*cluster_total/100))})",
            colors=colors)
    plt.title(f"Cluster {cluster_id} Difficulty Breakdown ({cluster_total} charts)")
    plt.tight_layout()
    plt.show()