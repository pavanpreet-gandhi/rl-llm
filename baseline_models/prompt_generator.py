# Maps BabyAI actions to text
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


def generate_prompt(mission, descriptions):
    """
    Constructs a prompt for the LLM based on the mission and observations.
    """
    possible_actions = "\n".join(
        f"{idx}: {action}" for idx, action in action_to_text.items()
    )
    obs_text = "\n".join(descriptions) if descriptions else "No descriptions available."

    return (
        f"Mission: {mission}\n\n"
        f"Observations:\n{obs_text}\n\n"
        f"Available actions:\n{possible_actions}\n\n"
        "Think step by step about what might be the best action to take given the mission and observations.\n"
        "After your reasoning, on a new line, write 'Final action:' followed by one of the available actions exactly as written above.\n"
    )


def parse_llm_response(response):
    """
    Extracts the chosen action from the LLM response.
    Returns an integer action ID corresponding to the chosen action.
    """

    # 1) Look for a line containing 'Final action:'
    final_action_line = None
    for line in response.splitlines():
        lower_line = line.lower()
        if "final action:" in lower_line:
            final_action_line = lower_line.split("final action:")[-1].strip()
            break

    # 2) If not found, consider the entire response
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

    # B) Check text after colon
    if len(tokens) > 1:
        after_colon = tokens[1].strip()
        if after_colon in text_to_action:
            print(f"Chose action by text after colon: {after_colon}")
            return text_to_action[after_colon]

    # C) Match entire response
    if final_action_line in text_to_action:
        print(f"Chose action by text: {final_action_line}")
        return text_to_action[final_action_line]

    # Default to "done"
    print(f"No valid action found in '{final_action_line}'. Defaulting to 'done'.")
    return text_to_action["done"]


def llm_choose_action(generator, mission, descriptions):
    """
    Uses an LLM to decide the best action given the mission and observations.
    """

    # Generate prompt
    prompt = generate_prompt(mission, descriptions)

    # Get LLM response
    response = generator(
        prompt, max_new_tokens=50, do_sample=True, top_p=0.9, truncation=True
    )[0]["generated_text"]

    print("LLM Prompt:\n", prompt)
    print("LLM Response:\n", response)

    # Parse the response
    return parse_llm_response(response)
