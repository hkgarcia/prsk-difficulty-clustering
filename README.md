<a id="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project
This project will focus on the characteristics of beatmaps in the mobile rhythm game Hatsune Miku: Colorful Stage. Each song in the game has multiple beatmaps (charts) ranging from Easy to Master, with assigned difficulty levels ranging from 5 to 37 (38 as of November 2025). Charts are composed of note types such as taps, flicks, and holds. Together, these notes combine to create patterns on a player’s screen. With these mechanics in mind, this project aims to analyze how note types, BPM, and other related song features contribute to the difficulty of a chart. I plan to conduct rhythm game chart clustering using the k-means method with a set of designed features. These features were carefully chosen based on what determines the “difficulty” of a rhythm game. 

But what exactly defines the difficulty of a rhythm game in the first place? Is it the complexity of note combinations, the number of notes (note density), the speed of notes, timing strictness, or the number of misses permitted? (Liang et al.) The definition of difficulty varies from game to game. An official Q&A session from the developers of Hatsune Miku: Colorful Stage (fron “Wondershow Channel #32,” unofficially translated by X user @pjsekai_eng), answered the question, “What criteria do you use to determine a song’s difficulty level?” The developers stated, “The difficulty levels are decided based on how difficult it is to clear the song. For example, songs with a lot of flick notes and difficult-looking note placements may be set at a higher difficulty. For some songs where there’s a big gap between how difficult it is to clear the song and how difficult it is to FC [full combo] the song, the decision may be based on how difficult it is to FC the song instead.

With this context in mind, I followed a similar method to Tsujino et al. when selecting features for clustering that build onto the difficulty of the charts. I plan to use two types of features: Score features, differing according to each chart’s difficulty level, and song features, which are consistent for charts of the same song.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure and File Descriptions

### Data Folder (`data/`)

- `data/charts/`: Contains **.svg** files of charts downloaded from [sekai.best](https://sekai.best). Downloaded using **src/chart_downloads.py**.

- `data/example_charts/`: Test charts used for smaller **svg_parsing.py**. Used to test functions, calculations, and feature logic before applying to all charts.

- `data/id_bpm_playback.csv`: CSV file with **songID**, **bpm**, and **playback** (minute:seconds format) for songs from [sekaipedia](https://www.sekaipedia.org/wiki/Song_data_list). Provides song features.

- `data/song_metadata.csv`: Main CSV file used for data analysis. Combines **id_bpm_playback.csv** with other databases ([sekai.best](https://sekai.best), [sekai-world](https://github.com/Sekai-World)). Contains: **songID**, **difficulty** (Easy, Normal, Hard, Expert, Master, Append), **difficulty_level** (numeric, ranging from 5-38, used for checking results), **note_count**, **song_title_jpn**, **song_title_eng**, **bpm**, **playback_time_minutes** (minute:second), and **playback_time_seconds** (seconds, used for data analysis).

- `data/svg_data.md`: Notes and breakdown of **.svg** data. Includes a list of unique SVG classes, class hierarchy and relationships, notes from data cleaning, and code explanations and walkthroughs for parsing and feature calculations.

---

### Source Folder (`src/`)

- `src/archive/svg_parsing.py`: Initial single-chart parsing script, using charts in **data/example_charts** for testing, coding, and function exploration. Contains detailed comments and explanations for each function.
- `src/archive/test_svg_parsing.py`: Original, simple parsing script. Used to identify classes and relationships in SVG files before proceeding with coding and functions.

- `src/chart_downloads.py`: Script to download **.svg** chart files from [sekai.best](https://sekai.best) to **data/charts/**. Users can adjust which songs (by songID range) and difficulties (Easy to Append) to download.

- `src/data_scraping.py`: Reads **id_bpm_playback.csv** and other databases ([sekai.best](https://sekai.best), [sekai-world](https://github.com/Sekai-World)) to create **song_metadata.csv**. Collects BPM and playback time from id_bpm_playback.csv. Collects song title (Japanese and English), difficulty (Easy to Append), difficulty level (5 to 38), and total note count.

- `src/svg_parsing_all.py`: Main script for feature calculations and extraction, PCA, and clustering. Uses the same methods and functions described in **svg_parsing.py** for feature calculation, and applies data analysis through PCA per feature group, Kmeans clustering, and plots of clustering results. Currently calculates k = 2, 3, 4, 5, 6, 8, 10, 12, 16, 30, and 45. Can update the number of clusters used. Lastly, saves data from clustering analysis in **data/outputs**: chart, difficulty, difficulty_level, cluster, total_notes, bpm, duration_seconds, pca_x, pca_y.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

To get a local copy up and running, follow these steps to run the SVG chart analysis.

### Prerequisites

- Python 3.10+  
- pip
- Required packages (via requirements.txt)

### Installation

1. **Clone the repo**
   ```sh
   git clone https://github.com/hkgarcia/prsk-difficulty-clustering.git
   ```

2. **Install dependencies**

   ```sh
   pip install r-requirements.txt
   ```

3. **Download SVG chart files**
   ```sh
   python src/chart_downloads/py
   ```
4. **Collect song metadata**
   ```sh
   python src/data_scraping.py
   ```

5. **Run full analysis**
   ```sh
   python src/svg_parsing_all.py
   ```

- *Approximate time for full analysis (600+ charts):* ~7 minutes

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REFERENCES -->

## References

### Academic Papers
Cao, D., Wang, Z., Echevarria, J., & Liu, Y. (2023). SVGformer: Representation learning for continuous vector graphics using transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023).* [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_SVGformer_Representation_Learning_for_Continuous_Vector_Graphics_Using_Transformers_CVPR_2023_paper.pdf)

Liang, Y., Li, W., & Ikeda, K. (2019). Procedural content generation of rhythm games using deep learning methods. In *Proceedings of the 1st Joint International Conference on Entertainment Computing and Serious Games (ICEC-JCSG)* (pp. 134–145). Springer. [PDF](https://inria.hal.science/hal-03652042v1/document)

Sakaamini, A., Van Slyke, A., Partouche, J., Wu, T., & Wiersma, R. D. (2023). An AI-based universal phantom analysis method based on XML-SVG wireframes with novel functional object identifiers. *Physics in Medicine and Biology, 68*(14), 10.1088/1361-6560/acdb44. [DOI](https://doi.org/10.1088/1361-6560/acdb44)

Tsujino, Y., Yamanishi, R., & Yamashita, Y. (2019). Characteristics study of dance-charts on rhythm-based video games. In *Proceedings of the 2019 IEEE Conference on Computational Intelligence and Games (CIG)* (pp. 1–8). IEEE. [PDF](https://ieee-cog.org/2019/papers/paper_157.pdf)

### GitHub and Code Library Resources
- [Scraping Dynamic JavaScript/Ajax Websites With BeautifulSoup](https://github.com/oxylabs/Scraping-Dynamic-JavaScript-Ajax-Websites-With-BeautifulSoup?tab=readme-ov-file)  
- [svgpathtools Library](https://pypi.org/project/svgpathtools/)
- [Web Scraping With Python](https://github.com/KOrfanakis/Web_Scraping_With_Python/tree/main)  

### Community Game Data & Resources
- [pjsekai.moe](https://pjsekai.moe/#/)  
- [sekai.best](https://sekai.best/)  
- [Sekaipedia Song Data List](https://www.sekaipedia.org/wiki/Song_data_list)
- [Sekai World (English) Master DB Difference](https://github.com/Sekai-World/sekai-master-db-en-diff)  
- [Sekai World GitHub Database](https://github.com/Sekai-World)
- [Unofficial English X/Twitter (@pjsekai_eng)](https://x.com/pjsekai_eng?lang=en)  

### Official Game Resources
- [Official Project SEKAI X/Twitter (@pj_sekai)](https://x.com/pj_sekai)  
- [Official Project SEKAI “Colorful Stage” Website](https://colorfulstage.com/)  
- [Official Project SEKAI SEGA site (jp)](https://pjsekai.sega.jp/)  

<!-- CONTACT -->
## Contact

Hannah Garcia - hkgarcia04@gmail.com

Project Link: [https://github.com/hkgarcia/prsk-difficulty-clustering](https://github.com/hkgarcia/prsk-difficulty-clustering)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
