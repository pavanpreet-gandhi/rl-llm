import sys
import gym
import warnings
import babyai_text
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

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
text_to_action = {v: k for k, v in action_to_text.items()}


def parse_observation(info):
    """
    Prints the textual descriptions stored in info['descriptions'].
    """
    print("Observation:")
    if "descriptions" in info and info["descriptions"]:
        for description in info["descriptions"]:
            print(description)
    else:
        print("(No textual descriptions found in info['descriptions'])")
    print()


def llm_choose_action(
    generator, mission, descriptions, prev_action=None, unchanged=False
):
    """
    Uses Qwen1.5-0.5B-Instruct to decide which action to take next,
    given the mission and current textual descriptions.
    If the observation is unchanged (unchanged=True) and prev_action is provided,
    it adds an extra instruction not to repeat the previous action.

    Returns an integer action ID corresponding to the chosen action.
    """
    possible_actions = "\n".join(
        f"{idx}: {action}" for idx, action in action_to_text.items()
    )
    obs_text = "\n".join(descriptions) if descriptions else "No descriptions available."

    extra_instruction = ""
    if unchanged and prev_action is not None:
        extra_instruction = (
            f"\nNote: Your previous action was '{action_to_text[prev_action]}' "
            "and it did not change the situation. Do not repeat this action."
        )

    # Construct prompt for Qwen
    prompt = (
        f"Mission: {mission}\n\n"
        f"Observations:\n{obs_text}\n\n"
        f"Available actions:\n{possible_actions}\n\n"
        """**Instructions:**
        - **Choose the best action** based on the mission and observations.
        - If the **target object is visible**, take the **next logical step** to complete the task.
        - If the **target is not visible**, explore by turning or moving forward.
        **Example 1:**
        If the correct action is *go forward*, your answer should be exactly:
        **go forward**
        **Example 2:**
        If the target object is not visible, explore by turning:
        **turn left** (or **turn right**)
        **Now, based on the mission and observations above, output ONLY one phrase (exactly one of the following):**
        turn left, turn right, go forward, pick up, drop, toggle, done.
        ### Answer: """
        # # f"{extra_instruction}\n\n"
        # "Your answer: "
    )

    response = generator(
        prompt,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        truncation=True,
    )[0]["generated_text"]

    # Debug prints:
    # print("LLM prompt:")
    # print(prompt)
    print("\nLLM response:")
    print(response)

    # Clean up the response.
    if response.startswith(prompt):
        response = response[len(prompt) :]
    answer = response.strip().lower()

    if answer in text_to_action:
        print(f"Chose action: {answer}")
        return text_to_action[answer]

    for key in text_to_action.keys():
        if key in answer:
            print(f"Chose action: {key}")
            return text_to_action[key]

    else:
        try:
            num = int(answer)
            if num in action_to_text:
                print(f"Chose action by numeric value: {num} -> {action_to_text[num]}")
                return num
        except ValueError:
            pass

    print(f"No valid action found in '{answer}'. Defaulting to 'done'.")
    return text_to_action["done"]


def run_episode(env, generator):
    """
    Runs one BabyAI episode using Qwen to select actions.
    """
    obs, info = env.reset()
    done = False
    step = 0
    last_obs = None
    last_action = None

    mission = obs.get("mission", "No mission provided")
    print("Mission:")
    print(mission)
    print()

    while not done:
        step += 1
        parse_observation(info)
        descriptions = info.get("descriptions", [])

        # Determine if the observation is unchanged.
        unchanged = (descriptions == last_obs) if last_obs is not None else False

        action_id = llm_choose_action(
            generator,
            mission,
            descriptions,
            prev_action=last_action,
            unchanged=unchanged,
        )
        action_str = action_to_text.get(action_id, f"Unknown({action_id})")

        obs, reward, done, info = env.step(action_id)
        print(
            f"Step {step}: Action = {action_str} (ID={action_id}), Reward={reward}, Done={done}\n"
        )

        # Update last observation and action.
        last_obs = descriptions
        last_action = action_id

    print(f"Episode finished in {step} steps.\n")


def main():
    print("Loading Qwen1.5-0.5B-Instruct from Hugging Face...")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("LLM loaded.\n")

    # env = gym.make("BabyAI-MixedTrainLocal-v0")
    env = gym.make("BabyAI-GoToObj-v0")
    num_episodes = 1
    for i in range(num_episodes):
        print(f"=== Starting Episode {i+1} ===")
        run_episode(env, generator)
        print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
