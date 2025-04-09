# rl-llm

## About
This repositpry explores training LLMs with Reinforcement Learning (RL) using the BabyAI-Text environment. The goal is to create a framework that allows for the training of LLMs in a simulated environment, enabling them to learn from their interactions and improve their performance over time.

## Setup
0. Connect to Cloud GPU

    Connect from vscode to vast.ai
    ```
    ssh -i ~/.ssh/id_rsa -p 30077 root@185.150.27.254 -L 8080:localhost:8080
    ```

    Generate SSH key to connect vast.ai instance with github
    ```
    eval "$(ssh-agent -s)" # start ssh agent, not automatic on vast
    ssh-keygen -t ed25519
    ssh-add; ssh-add -l
    echo "public key:"
    cat ~/.ssh/id_ed25519.pub
    ```
    
1. Clone this repository
    ```bash
    git clone https://github.com/pavanpreet-gandhi/rl-llm/
    cd rl-llm
    git checkout clean-mvp-dev
    ```
2. Create and activate `.venv` environment
    ```bash
    python3 -m venv --system-site-packages .venv
    source .venv/bin/activate
    ```
3. Install this repo in editable mode
    ```bash
    pip install -e .
    ```
4. Install the requirements
    
    For lambda labs ARM GPU:
    ```bash
    pip install -r requirements.txt
    pip install numpy==1.23.1 tf-keras # fix dependancy issues
    ```
    Otherwise:
    ```bash
    pip install -r requirements.txt
    ```
5. Install BabyAI-Text
    ```bash
    cd Grounding_LLMs_with_online_RL
    pip install blosc; cd babyai-text/babyai; pip install -e .; cd ..
    cd gym-minigrid; pip install -e.; cd ..
    pip install -e .
    cd ../..
    ```
6. Configure git
    ```bash
    git config --global credential.helper store
    git config --global user.name "Pavan"
    git config --global user.email "pavanpreet.gandhi@gmail.com"
    ```
7. Login to wandb (create an account first if you don't have one)
    ```bash
    wandb login
    ```
8. Login to huggingface (create an account first if you don't have one)
    ```bash
    huggingface-cli login
    ```
9. Logging Settings:

    If you are just running a trial and don't want to log in hugging face:
    
    ```
    In train.py / parse_args() change "push_to_hub" to False
    ```
    If you are doing a trial run and don't want to log to wandb group account:
    ```
    In Train.py / parse_args(), change "entity": "OE_2025" to some other entities in your own account
    ```

10. Loading pre-trained model:

    To load pretrained model, in train.py / parse_args():

    ```
    "pretrained_dir": "your-hf-username/your-model-repo"
    "load_checkpoint": True,
    ```
    
## Run everything at once (Lamnbda)
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install numpy==1.23.1 tf-keras # fix dependancy issues
cd Grounding_LLMs_with_online_RL
pip install blosc; cd babyai-text/babyai; pip install -e .; cd ..
cd gym-minigrid; pip install -e.; cd ..
pip install -e .
cd ../..
git config --global credential.helper store
git config --global user.name "Pavan"
git config --global user.email "pavanpreet.gandhi@gmail.com"
wandb login
huggingface-cli login
```