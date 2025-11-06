# FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation

[[Paper]](https://arxiv.org/pdf/2411.15753) [[Project Page]](https://tonyfang.net/FoAR/)

**Authors**: [Zihao He*](https://alan-heoooh.github.io/), [Hongjie Fang*](https://tonyfang.net/), [Jingjing Chen](mailto:jjchen20@sjtu.edu), [Hao-Shu Fang](https://fang-haoshu.github.io/), [Cewu Lu](https://www.mvig.org/)

![teaser](assets/images/teaser.png)

## 🛫 Getting Started

### 💻 Installation

We use [RISE](https://rise-policy.github.io/) as our real robot baseline, Please following the [installation guide](assets/docs/INSTALL.md) to install the `foar` conda environments and the dependencies, as well as the real robot environments. Also, remember to adjust the constant parameters in `dataset/constants.py` and `utils/constants.py` according to your own environment.

### 📷 Calibration

Please calibrate the camera(s) with the robot before data collection and evaluation to ensure correct spatial transformations between camera(s) and the robot. Please refer to [calibration guide](assets/docs/CALIB.md) for more details.

### 🛢️ Data Collection

We apply the data collection process in the <a href="https://rh20t.github.io/">RH20T</a> paper. You may need to adjust `dataset/realworld.py` to accommodate different data formats. The sample data of peeling and wiping tasks can be found <a href="https://drive.google.com/drive/folders/1Tq7wR9MdLLFr5AZTqXvypF2_2CJ-ghd5?usp=sharing">here</a>, which have the format of

```
peel
|-- calib/
|   |-- [calib timestamp 1]/
|   |   |-- extrinsics.npy             # extrinsics (camera to marker)
|   |   |-- intrinsics.npy             # intrinsics
|   |   `-- tcp.npy                    # tcp pose of calibration
|   `-- [calib timestamp 2]/           # similar calib structure
`-- train/
    |-- [episode identifier 1]
    |   |-- metadata.json              # metadata
    |   |-- timestamp.txt              # calib timestamp
    |   |-- high_freq_data/            # high frequency data
    |   |   `-- force_torque_tcp_joint_timestamp.npy
    |   |                              # force/torque, tcp, joint, timestamp data
    |   |-- cam_[serial_number 1]/    
    |   |   |-- color                  # RGB
    |   |   |   |-- [timestamp 1].png
    |   |   |   |-- [timestamp 2].png
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].png
    |   |   |-- depth                  # depth
    |   |   |   |-- [timestamp 1].png
    |   |   |   |-- [timestamp 2].png
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].png
    |   |   |-- tcp                    # tcp
    |   |   |   |-- [timestamp 1].npy
    |   |   |   |-- [timestamp 2].npy
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].npy
    |   |   `-- gripper_command        # gripper command
    |   |       |-- [timestamp 1].npy
    |   |       |-- [timestamp 2].npy
    |   |       |-- ...
    |   |       `-- [timestamp T].npy
    |   `-- cam_[serial_number 2]/     # similar camera structure
    `-- [episode identifier 2]         # similar episode structure
```

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
@ARTICLE{10964857,
  author={He, Zihao and Fang, Hongjie and Chen, Jingjing and Fang, Hao-Shu and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation}, 
  year={2025},
  volume={10},
  number={6},
  pages={5625-5632},
  keywords={Robots;Robot sensing systems;Force;Point cloud compression;Dynamics;Sensors;Visualization;Transformers;Noise measurement;Training;Force and tactile sensing;imitation learning;perception for grasping and manipulation},
  doi={10.1109/LRA.2025.3560871}}
```

