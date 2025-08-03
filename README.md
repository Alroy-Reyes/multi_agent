# Multi-Agent Project

## Installation

### Clone the repository
```bash
git clone https://github.com/Alroy-Reyes/multi_agent.git
cd multi_agent
```

### Activate virtual environment

#### Activate on Windows
```bash
marl_env\Scripts\activate
```

#### Activate on macOS/Linux
```bash
source marl_env/bin/activate
```

### Install dependencies
```bash
pip install pettingzoo gymnasium ray[rllib]
pip install scikit-image opencv-python
pip install ray[default]==2.9.3
pip install torch
pip install numpy==1.24.4
pip install tensorboard
```

## Quickstart
```bash
python training\train_ppo.py
```
