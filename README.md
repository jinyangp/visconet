<a name="readme-top"></a>

## *ViscoNet*: Bridging and Harmonizing Visual and Textual Conditioning for ControlNet

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#results">Proposed Architecture</a></li>
        <li><a href="#built-with">Results</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The aim of this project was to implement a novel model architecture that incorporated the 
latest image prompt adapters, mainly ControlNet and IP-Adapter, with pre-trained text-to-image models to perform Human Pose Transfer using multi-modality prompts. Building upon [Visconet](https://github.com/soon-yau/visconet), the proposed model architecture is able to incorporate both text and image conditioning to perform Human Pose Transfer over a diverse range of samples.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PROPOSED ARCHITECTURE -->
## Proposed Architecture

![Architecture](./assets/architecture.png)
<p align="center"><em>Proposed Model Architecture</em></p>

Our proposed architecture contains of two modules, the ControlNet Module and the Global Styles Fusion Module. The ControlNet Module contains the improved Style Encoder Sub-Module *H<sub>E</sub>* and the Control Feautre Mask Module *H<sub>M</sub>* while the Global Styles Fusion Module contains the Style Extractor Sub-Module *H<sub>S</sub>* and the Attention Fusion Module *H<sub>A</sub>*. 

Meanwhile, the original Visconet contains only the ControlNet module with the original Style Encoder Sub-Module *H<sub>E,0</sub>*.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results

![Architecture](./assets/ablation_results.png)
<p align="center"><em>Results obtained from the chosen baselines and each ablation</em></p>

<div align="center">

| Model | SSIM (↑)  | FID (↓)  | LPIPS (↓) |
| ----------- | ----------- | ----------- | ----------- | 
| PIDM **(B)** | 0.656 | 0.390 | 0.184 | 
| CFLD **(B)** | 0.660 | 0.195 | 0.183 |
| Naive HF implementation **(B)** | 0.566 | 7.250 | 0.234 |
| Baseline Visconet **(B)** | 0.401 | 9.790 | 0.306 |
| Ablation 1 | 0.407 | 8.004 | 0.308 |
| Ablation 2 | 0.467 | 6.566 | 0.266 |
| Ablation 3 | 0.538 | 1.557 | 0.242 |
| Ablation 4 | 0.541 | 1.604 | 0.243 |

</div>
<p align="center"><em>Metrics obtained from the chosen baselines, denoted with **(B)**, and each ablation</em></p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTINUE HERE LATER
### Requirements
A suitable [conda](https://conda.io/) environment named `control` can be created
and activated with:
```
conda env create -f environment.yaml
conda activate control
```
### Files
All model and data files are in [here](https://huggingface.co/soonyau/visconet/tree/main).
Including eval.zip containing all images used in human evaluation.

### Gradio App
[![App](./assets/app.png)](https://youtu.be/3_6Zq3hk86Q)
run ```python gradio_visconet.py```
-->

### Acknowledgement
This project is based on the work done in [Visconet](https://github.com/soon-yau/visconet).  
Special thanks to the original authors for their contributions.
