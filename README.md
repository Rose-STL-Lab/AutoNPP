<p align="center" >
  <a href="https://github.com/Rose-STL-Lab/AI-STPP"><img src="https://fremont.zzhou.info/images/2022/10/06/image-20221006102054441.png" width="256" height="256" alt="AI-STPP"></a>
</p>
<h1 align="center">Auto-NPP</h1>
<h4 align="center">✨Automatic Integration for Neural Point Process✨</h4>

<p align="center">
    <a href="https://zzhou.info/LICENSE"><img src="https://camo.githubusercontent.com/87d0b0ec1c0a97dbf68ce4d3098de6912bca75aa006304dd0a55976e6673cbe1/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f64656c67616e2f6c6f677572752e737667" alt="license"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-yellow" alt="python">
    <img src="https://img.shields.io/badge/Version-1.0.0-green" alt="version">
</p>

## | Paper

[Automatic Integration for Fast and Interpretable Neural Point Processes](https://proceedings.mlr.press/v211/zhou23a/zhou23a.pdf)

## | Installation

Dependencies: `make`, `conda-lock`

```bash
make create_environment
conda activate autonpp
```

## | Dataset Download

```bash
make download prefix=data
```

## | Get Trained Models

```
make download prefix=models
```

## | Training and Testing

Specify the parameters in `configs/test_autoint_1d_dataset.yaml` and then run

```bash
make run
```

The loss curves and example intensity predictions are saved to `figs/`. 
With real-world datasets, the ground truth intensity is a placeholder and can be safely ignored.
The logs are saved to `logs/`.
The models are saved to `models/`.

To use the trained models, set `retrain: false`.

## | Cite

```
@article{zhou2023automatic,
  title={Automatic Integration for Fast and Interpretable Neural Point Processes},
  author={Zhou, Zihao and Yu, Rose},
  journal={Learning for Dynamics and Control (L4DC)},
  year={2023}
}

```
