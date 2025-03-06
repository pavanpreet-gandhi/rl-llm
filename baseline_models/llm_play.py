import sys
import gym
import warnings
import babyai_text
from transformers import pipeline

warnings.filterwarnings("ignore")

# Maps BabyAI actions to text:
action_to_text = {
    0: "turn left",
    1: "turn right",
    2: "go forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}
# Reverse mapping: text -> integer
text_to_action = {v: k for k, v in action_to_text.items()}


def parse_observation(info):
    """
    Prints the textual descriptions stored in info['descriptions'],
    exactly like in human_play.py.
    """
    print("Observation:")
    if "descriptions" in info and info["descriptions"]:
        for description in info["descriptions"]:
            print(description)
    else:
        print("(No textual descriptions found in info['descriptions'])")
    print()


def llm_choose_action(generator, mission, descriptions):
    """
    Uses a small LLM to decide which action to take next,
    given the mission and current textual descriptions.

    Returns an integer action ID corresponding to the chosen action.
    """

    # Build a detailed prompt that includes chain-of-thought reasoning
    possible_actions = "\n".join(
        f"{idx}: {action}" for idx, action in action_to_text.items()
    )
    obs_text = "\n".join(descriptions) if descriptions else "No descriptions available."

    prompt = (
        f"Mission: {mission}\n\n"
        f"Observations:\n{obs_text}\n\n"
        f"Available actions:\n{possible_actions}\n\n"
        "Think step by step about what might be the best action to take given the mission and observations.\n"
        "After your reasoning, on a new line, write 'Final action:' followed by one of the available actions exactly as written above.\n"
    )

    # Generate a response from the model
    response = generator(
        prompt,
        max_new_tokens=50,  # limit the generation length
        do_sample=True,
        top_p=0.9,
        truncation=True,
    )[0]["generated_text"]

    print("LLM prompt:")
    print(prompt)
    print("LLM response:")
    print(response)

    # 1) Look for a line containing 'Final action:'
    final_action_line = None
    for line in response.splitlines():
        lower_line = line.lower()
        if "final action:" in lower_line:
            final_action_line = lower_line.split("final action:")[-1].strip()
            break

    # 2) If not found, consider the entire response as the final action
    #    (in case the model just outputs "3" or "turn left" by itself).
    if not final_action_line:
        final_action_line = response.strip().lower()

    # A) Try numeric ID first
    tokens = final_action_line.split(":")
    first_token = tokens[0].strip()
    try:
        action_id = int(first_token)
        if action_id in action_to_text:
            print(
                f"Chose action by numeric ID: {action_id} -> {action_to_text[action_id]}"
            )
            return action_id
    except ValueError:
        pass

    # B) If there's a colon, check text after the colon
    if len(tokens) > 1:
        after_colon = tokens[1].strip()
        for action_str, action_id in text_to_action.items():
            if after_colon == action_str:
                print(f"Chose action by text after colon: {after_colon}")
                return action_id
            # Or partial match if needed:
            # if action_str in after_colon:
            #     return action_id

    # C) Finally, see if the entire final_action_line matches an action string
    for action_str, action_id in text_to_action.items():
        if final_action_line == action_str:
            print(f"Chose action by text: {action_str}")
            return action_id
        # Or partial match if needed:
        # if action_str in final_action_line:
        #     return action_id

    # If we get here, no valid action was found
    print(f"No valid action found in '{final_action_line}'. Defaulting to 'done'.")
    return text_to_action["done"]


def run_episode(env, generator):
    """
    Runs one BabyAI episode using the LLM to select actions.
    Prints the mission, textual observations, chosen actions, etc.
    """

    # Reset environment
    obs, info = env.reset()
    done = False
    step = 0

    # Print the mission (goal)
    mission = obs.get("mission", "No mission provided")
    print("Mission:")
    print(mission)
    print()

    while not done:
        step += 1

        # Print textual observation
        parse_observation(info)

        # Decide on an action using the LLM
        descriptions = info.get("descriptions", [])
        action_id = llm_choose_action(generator, mission, descriptions)
        action_str = action_to_text.get(action_id, f"Unknown({action_id})")

        # Take the step in the environment
        obs, reward, done, info = env.step(action_id)

        # Log
        print(
            f"Step {step}: Action = {action_str} (ID={action_id}), Reward={reward}, Done={done}"
        )
        print()

    print(f"Episode finished in {step} steps.\n")


def main():
    print("Loading small LLM from Hugging Face (FLAN-T5-Small)...")
    # Use the text2text-generation pipeline with FLAN-T5-Small
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    print("LLM loaded.\n")

    # Create your BabyAI environment
    env = gym.make("BabyAI-MixedTrainLocal-v0")

    # Run a couple of episodes with LLM-driven actions
    num_episodes = 1
    for i in range(num_episodes):
        print(f"=== Starting Episode {i+1} ===")
        run_episode(env, generator)
        print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
