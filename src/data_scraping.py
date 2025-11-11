import datetime as DT
import json
import pandas as pd
import requests

# source URLs from sekai-world (sekai best) database via github
MUSICS_ENG = "https://raw.githubusercontent.com/Sekai-World/sekai-master-db-en-diff/refs/heads/main/musics.json"
MUSICS_JPN = "https://raw.githubusercontent.com/Sekai-World/sekai-master-db-diff/main/musics.json"
MUSIC_DIFFICULTIES = "https://raw.githubusercontent.com/Sekai-World/sekai-master-db-diff/main/musicDifficulties.json"
OTHER_DATA = "../data/id_bpm_playback.csv"

# data we need to collect:
# bpm (from sekaipedia)
# playback time (from sekaipedia)
# song title (eng + jpn) (from sekai-world)
# note count (from sekai-world)
# hasGimmick - not included for now

def fetch_json(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def main():
    # musics.json: song id, title (eng)
    musics_eng = fetch_json(MUSICS_ENG)
    df_music_eng = pd.DataFrame(musics_eng)[[
        "id",
        "title",
    ]].rename(columns={
        "id": "song_id",
        "title": "song_title_eng",
    })
    df_music_eng["song_id"] = df_music_eng["song_id"].astype(str)
    
    # musics.json: song id, title (jpn)
    musics_jpn = fetch_json(MUSICS_JPN)
    df_music_jpn = pd.DataFrame(musics_jpn)[[
        "id",
        "title",
    ]].rename(columns={
        "id": "song_id",
        "title": "song_title_jpn",
    })
    df_music_jpn["song_id"] = df_music_jpn["song_id"].astype(str)
    
    # musicDifficulties.json: song id, difficulty, note count
    diffs = fetch_json(MUSIC_DIFFICULTIES)
    df_diffs = pd.DataFrame(diffs)[[
        "musicId",
        "musicDifficulty",
        "totalNoteCount"
    ]].rename(columns={
        "musicId": "song_id",
        "musicDifficulty": "difficulty",
        "totalNoteCount": "note_count"
    })
    df_diffs["song_id"] = df_diffs["song_id"].astype(str)
    
    # other data (pulled from sekaipedia): song id, bpm, playback time (minutes)
    other_data = pd.read_csv(OTHER_DATA)
    df_other = other_data[[
        "songID",
        "bpm",
        "playback"
    ]].rename(columns={
        "songID": "song_id",
        "bpm": "bpm",
        "playback": "playback_time_minutes"
    })
    df_other["song_id"] = df_other["song_id"].astype(str)
    
    # converting minutes to seconds
    # adding 00: to each entry for timedelta format
    df_other["playback_time_minutes"] = "00:0" + df_other["playback_time_minutes"]
    df_other["playback_time_seconds"] = pd.to_timedelta(df_other["playback_time_minutes"]).dt.total_seconds()
    
    # merge on song_id (str)
    df = df_diffs.merge(df_music_jpn, on="song_id", how="left")
    df = df.merge(df_music_eng, on="song_id", how="left")
    df = df.merge(df_other, on="song_id", how="left")
    
    # sort by song_id, note count (can update accordingly in this script)
    #df = df.sort_values(by=["song_id", "note_count"])

    # save our output
    output_path = "../data/song_metadata.csv"
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
