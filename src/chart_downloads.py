import os
import requests

# chart svg URL pattern
BASE_URL = "https://storage.sekai.best/sekai-music-charts/jp/{song_id:04d}/{difficulty}.svg"

# difficulties to download 
# users can edit list as needed
DIFFICULTIES = [
    "easy",
    "normal",
    "hard",
    "expert",
    "master",
    "append"
]

# range of song IDs to download
# users can edit range as needed
START_ID = 1
END_ID = 700

# save charts in ../data/charts relative to src/
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "charts")
os.makedirs(SAVE_DIR, exist_ok=True)

# download charts
for song_id in range(START_ID, END_ID + 1):
    for diff in DIFFICULTIES:
        url = BASE_URL.format(song_id=song_id, difficulty=diff)
        filename = os.path.join(SAVE_DIR, f"{song_id}_{diff}.svg")

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Not found: {url}")

        except Exception as e:
            print(f"Error downloading {url}: {e}")