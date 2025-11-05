<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- ABOUT THE PROJECT -->
## About The Project
This project will focus on the characteristics of beatmaps in the mobile rhythm game Hatsune Miku: Colorful Stage. Each song in the game has multiple beatmaps (charts) ranging from Easy to Master, with assigned difficulty levels ranging from 5 to 37. Charts are composed of note types such as taps, flicks, and holds. Together, these notes combine to create patterns on a player’s screen. With these mechanics in mind, this project aims to analyze how note types, BPM, and other related song features contribute to the difficulty of a chart. I plan to conduct rhythm game chart clustering using the k-means method with a set of designed features. These features were carefully chosen based on what determines the “difficulty” of a rhythm game. 

But what exactly defines the difficulty of a rhythm game in the first place? Is it the complexity of note combinations, the number of notes (note density), the speed of notes, timing strictness, or the number of misses permitted? (Liang et al.) The definition of difficulty varies from game to game. An official Q&A session from the developers of Hatsune Miku: Colorful Stage (fron “Wondershow Channel #32,” unofficially translated by X user @pjsekai_eng), answered the question, “What criteria do you use to determine a song’s difficulty level?” The developers stated, “The difficulty levels are decided based on how difficult it is to clear the song. For example, songs with a lot of flick notes and difficult-looking note placements may be set at a higher difficulty. For some songs where there’s a big gap between how difficult it is to clear the song and how difficult it is to FC [full combo] the song, the decision may be based on how difficult it is to FC the song instead.

With this context in mind, I followed a similar method to Tsujino et al. when selecting features for clustering that build onto the difficulty of the charts. I plan to use two types of features: Score features, differing according to each chart’s difficulty level, and song features, which are consistent for charts of the same song.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
TODO: Update section

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage
TODO: Update section

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REFERENCES -->

## References
Cao, D., Wang, Z., Echevarria, J., & Liu, Y. (2023). SVGformer: Representation learning for continuous vector graphics using transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023). https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_SVGformer_Representation_Learning_for_Continuous_Vector_Graphics_Using_Transformers_CVPR_2023_paper.pdf
Liang, Y., Li, W., & Ikeda, K. (2019). Procedural content generation of rhythm games using deep learning methods. In Proceedings of the 1st Joint International Conference on Entertainment Computing and Serious Games (ICEC-JCSG) (pp. 134–145). Springer. https://inria.hal.science/hal-03652042v1/document
Sakaamini, A., Van Slyke, A., Partouche, J., Wu, T., & Wiersma, R. D. (2023). An AI-based universal phantom analysis method based on XML-SVG wireframes with novel functional object identifiers. Physics in medicine and biology, 68(14), 10.1088/1361-6560/acdb44. https://doi.org/10.1088/1361-6560/acdb44
Tsujino, Y., Yamanishi, R., & Yamashita, Y. (2019). Characteristics study of dance-charts on rhythm-based video games. In Proceedings of the 2019 IEEE Conference on Computational Intelligence and Games (CIG) (pp. 1–8). IEEE. https://ieee-cog.org/2019/papers/paper_157.pdf

<!-- CONTACT -->
## Contact

Hannah Garcia - hkgarcia04@gmail.com

Project Link: [https://github.com/hkgarcia/prsk-difficulty-clustering](https://github.com/hkgarcia/prsk-difficulty-clustering)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
