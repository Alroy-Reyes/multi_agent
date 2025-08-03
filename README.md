Installation

Clone the repository
git clone https://github.com/Alroy-Reyes/multi_agent.git
cd multi_agent

Activate virtual environment
Activate on Windows
marl_env\Scripts\activate

Activate on macOS/Linux
source marl_env/bin/activate

Install dependencies
pip install pettingzoo gymnasium ray[rllib] 
pip install scikit-image opencv-python
pip install ray[default]==2.9.3
pip install torch 
pip install numpy==1.24.4
pip install tensorboard

Quickstart
python training\train_ppo.py
