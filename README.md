# FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation

[[Paper]](https://arxiv.org/pdf/2411.15753) [[Project Page]](https://tonyfang.net/FoAR/)

**Authors**: [Zihao He*](https://github.com/Alan-Heoooh), [Hongjie Fang*](https://tonyfang.net/), [Jingjing Chen](mailto:jjchen20@sjtu.edu), [Hao-Shu Fang](https://fang-haoshu.github.io/), [Cewu Lu](https://www.mvig.org/)

![teaser](assets/images/teaser.png)

## 🛫 Getting Started

### 💻 Installation

We use [RISE](https://rise-policy.github.io/) as our real robot baseline, Please following the [installation guide](assets/docs/INSTALL.md) to install the `foar` conda environments and the dependencies, as well as the real robot environments. Also, remember to adjust the constant parameters in `dataset/constants.py` and `utils/constants.py` according to your own environment.

### 📷 Calibration

Please calibrate the camera(s) with the robot before data collection and evaluation to ensure correct spatial transformations between camera(s) and the robot. Please refer to [calibration guide](assets/docs/CALIB.md) for more details.

### 🛢️ Data Collection
Data will be released soon.

### 🧑🏻‍💻 Training
The training scripts are saved in [script](script).

```bash
conda activate foar
bash script/command_train.sh # Train Foar policy
```

### 🤖 Evaluation

 Please follow the [deployment guide](assets/docs/DEPLOY.md) to modify the evaluation script.

Modify the arguments in `script/command_eval.sh`, then

```bash
conda activate foar
bash script/command_eval.sh
```

## ✍️ Citation

```bibtex
@article{
  he2024force,
  title = {FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation},
  author = {He, Zihao and Fang, Hongjie and Chen, Jingjing and Fang, Hao-Shu and Lu, Cewu},
  journal = {arXiv preprint arXiv:2411.15753},
  year = {2024}
}
```

