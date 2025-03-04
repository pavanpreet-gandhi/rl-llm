import gym
import babyai_text


def create_environment(env_name="BabyAI-MixedTrainLocal-v0"):
    """
    Creates and returns the BabyAI environment.
    """
    print(f"Creating environment: {env_name}...")
    env = gym.make(env_name)
    print("Environment created successfully.\n")
    return env
