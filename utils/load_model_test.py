import torch
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from peft import PeftModel

# === Load model and tokenizer ===
model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Or your fine-tuned checkpoint
lora_path = "Heisenger/delete-me-2025-04-10_22-59-18"             # Path to LoRA adapter directory

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with value head + PEFT adapter
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_path).to(device)

# Attach generation config manually
model.generation_config = GenerationConfig.from_pretrained(model_id)

# === Define chat formatting ===
def make_chat_input(observation: str, goal: str):
    messages = [
        {
            "role": "system",
            "content": f"You are an intelligent agent. Your goal is to **{goal}**. Before outputting your final answer, think step-by-step for at most 10 words about which action best advances your goal. You are navigating in a grid and can only see what is ahead.\n\nYour response should end with 'final answer: [answer]', where final answer is a chosen valid action (up to two words and nothing else) from below: \n\n- turn left\n- turn right\n- go forward\n- pick up\n- drop\n- toggle\n\nPLAY!"
        },
        {"role": "user", "content": observation},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(next(model.parameters()).device)
    return input_ids

# === Inference loop ===
def interact(goal, observation):
    input_ids = make_chat_input(observation, goal)
    generation_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.7,
        "pad_token_id": tokenizer.pad_token_id,
    }
    outputs = model.generate(input_ids, **generation_kwargs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== RESPONSE ===")
    print(decoded.split("\n"))

# Example usage
if __name__ == "__main__":
    goal = "put the purple ball next to the yellow key"
    observation = "You see a wall 5 steps forward\nYou see a wall 1 step right"
    interact(goal, observation)
