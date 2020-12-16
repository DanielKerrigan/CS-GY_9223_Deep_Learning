# CS-GY 9223: Deep Learning Project

This project uses a modified version of RLCard for the RL environments. You can find [the RLCard fork here](https://github.com/DanielKerrigan/rlcard/tree/dan)

## Contents

### Implementations

- `DQNAgent.py`: DQN agent
- `fully_connected.py`: fully-connected feed forward network with ReLU activations
- `memory.py`: Memory buffers used during RL

### Kuhn Poker

- `kuhn_poker_cfr_train.ipynb`: Train CFR agent
- `kuhn_poker_dqn_train.ipynb`: Train DQN agent
- `kuhn_poker_dqn_analyze.ipynb`: Examine the moves the DQN agent makes
- `kuhn_poker_eval.ipynb`: Evaluate the agents against each other

### Leduc hold'em
- `leduc_holdem_cfr_train.ipynb`: Train CFR agent
- `leduc_holdem_dqn_train.ipynb`: Train the DQN agent
- `leduc_holdem_dqn_analyze.ipynb` Examine the moves the DQN agent makes
- `leduc_holdem_eval.ipynb`: Evaluate the agents against each other

### Limit hold'em
- `limit_holdem_eval.ipynb`: Evaluate the agents against each other
- `limit_holdem_dqn_train.ipynb`: Train the DQN agent
- `limit_holdem_human_dqn.py`: Play against the DQN agent

### Other
- `vis.ipynb`: Visualizations for the report
- `images/`: Folder containing images of visualizations for the report
- `spec-file.txt`: List of Conda dependencies for this project's environment

## Results

The above notebooks and scripts reference files in the `models` and `experiments` folders. You can find these folders on [Google Drive](https://drive.google.com/drive/folders/1Q1NhFZDi6oDIR_MzvlgGEKXAaR_FmzeZ?usp=sharing).
