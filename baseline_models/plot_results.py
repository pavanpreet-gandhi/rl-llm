import matplotlib.pyplot as plt


def plot_results(episode_rewards, episode_steps):
    """
    Plots the reward and number of steps taken in each episode.

    Args:
        episode_rewards (list): Rewards earned in each episode.
        episode_steps (list): Number of steps taken in each episode.
    """

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot total reward per episode
    axes[0].plot(
        range(1, len(episode_rewards) + 1), episode_rewards, marker="o", linestyle="-"
    )
    axes[0].set_title("Total Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)

    # Plot number of steps per episode
    axes[1].plot(
        range(1, len(episode_steps) + 1),
        episode_steps,
        marker="s",
        linestyle="-",
        color="orange",
    )
    axes[1].set_title("Number of Steps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps Taken")
    axes[1].grid(True)

    # Display the plots
    plt.tight_layout()
    plt.show()
