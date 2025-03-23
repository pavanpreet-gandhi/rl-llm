# rl-llm

## About
This repositpry explores training LLMs with Reinforcement Learning (RL) using the BabyAI-Text environment. The goal is to create a framework that allows for the training of LLMs in a simulated environment, enabling them to learn from their interactions and improve their performance over time.

## Setup
1. Clone this repository
    ```bash
    git clone <repo_url>
    cd rl-llm
    git checkout <branch_name>
    ```
2. Create `.conda` environment
    ```bash
    conda create --prefix ./.conda python=3.9 -y
    conda activate ./.conda
    ```
3. Make sure `pip` is pointing to the correct conda environment (optional)
    ```bash
    which pip
    ```
4. Install this repo in editable mode
    ```bash
    pip install -e .
    ```
5. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
6. Install BabyAI-Text
    ```bash
    git clone https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git
    cd Grounding_LLMs_with_online_RL
    pip install blosc; cd babyai-text/babyai; pip install -e .; cd ..
    cd gym-minigrid; pip install -e.; cd ..
    pip install -e .
    cd ../..
    ```
7. Login to wandb (create an account first if you don't have one)
    ```bash
    wandb login
    ```
8. Login to huggingface (create an account first if you don't have one)
    ```bash
    huggingface-cli login
    ```