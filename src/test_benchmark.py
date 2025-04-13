import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig
from huggingface_hub.utils import EntryNotFoundError
from trl import AutoModelForCausalLMWithValueHead

# === Load model + tokenizer (friend-style) ===
def load_model(model_id, lora_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        peft_config = PeftConfig.from_pretrained(model_id)
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_id)
    except (EntryNotFoundError, ValueError) as e:
        print(f"Warning: {e}. Falling back to base model without LoRA.")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # model.generation_config = GenerationConfig.from_pretrained(model_id)
    return model, tokenizer

# === Format ARC-C test examples ===
def format_arc(dataset):
    prompts, labels = [], []
    for ex in dataset:
        q = ex["question"]
        choices = ex["choices"]
        label = ex["answerKey"]
        choice_lines = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(choices['label'], choices['text'])])
        # prompt = f"Q: {q}\n{choice_lines}\nA:"
        prompt = f"Question: {q}\n{choice_lines}\nAnswer:"
        prompts.append(prompt)
        labels.append(label)
    return prompts, labels

def get_device(model):
    return next(model.parameters()).device

# === Run predictions and extract first letter answer ===
def predict(model, tokenizer, prompts, max_new_tokens=5):
    preds = []
    device = get_device(model)
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = extract_choice_letter(decoded)
        preds.append(pred)
    return preds

def extract_choice_letter(text):
    for ch in text:
        if ch in "ABCDE":
            return ch
    return "?"  # fallback for invalid output

# === Main entry ===
if __name__ == "__main__":
    # Replace with your LoRA model
    # model_id = "Heisenger/final_runs-No_Reasoning_trial_1_dist_3"
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    model, tokenizer = load_model(model_id)

    # Load ARC-Challenge test set
    arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    print("First example in dataset:", arc[0])  # Optional debug print

    prompts, labels = format_arc(arc)
    preds = predict(model, tokenizer, prompts)

    # Compute accuracy
    correct = sum([p == gt for p, gt in zip(preds, labels)])
    acc = 100 * correct / len(labels)

    # === Final output (matches your image) ===
    print("\nARC-C | 0-shot | Metric: acc")
    print(f"Model Accuracy: {acc:.1f}")
