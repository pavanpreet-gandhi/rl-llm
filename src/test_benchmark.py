import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM  # GenerationConfig not used here
from peft import PeftModel, PeftConfig
from huggingface_hub.utils import EntryNotFoundError
import csv

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
    return model, tokenizer

# === Format ARC-C test examples ===
def format_arc(dataset):
    prompts, labels = [], []
    for ex in dataset:
        q = ex["question"]
        choices = ex["choices"]
        label = ex["answerKey"]
        choice_lines = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(choices['label'], choices['text'])])
        prompt = (
            "You are an expert at answering multiple-choice questions. "
            "Please provide only the single letter (A, B, C, D, or E) corresponding to the correct answer.\n\n"
            f"Question: {q}\n{choice_lines}\nAnswer:"
        )
        prompts.append(prompt)
        labels.append(label)
    return prompts, labels

# === Format HellaSwag test examples ===
def format_hellaswag(dataset):
    prompts, labels = [], []
    for ex in dataset:
        context = ex["ctx_a"]
        endings = ex["endings"]
        label = ex["label"]

        # Convert the label to an integer; if conversion fails, skip the example.
        try:
            label_int = int(label)
        except Exception as e:
            print(f"Warning: Skipping example with invalid label: {ex}")
            continue

        # Skip examples where label is negative
        if label_int < 0:
            print(f"Warning: Skipping example with invalid label: {ex}")
            continue

        choice_lines = "\n".join([f"{i}. {ending}" for i, ending in enumerate(endings)])
        prompt = (
            "You are an expert at completing sentences. "
            "Please provide only the single number (0, 1, 2, or 3) corresponding to the correct ending.\n\n"
            f"Context: {context}\n{choice_lines}\nAnswer:"
        )
        prompts.append(prompt)
        labels.append(str(label_int))
    return prompts, labels

def get_device(model):
    return next(model.parameters()).device

# === Helper to obtain input_ids using chat formatting if available ===
def get_input_ids(prompt, tokenizer, device):
    try:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(device)
    except Exception as e:
        print(f"Chat template error: {e}, falling back to standard tokenization.")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.size(0) == 0:
        print(f"Warning: input_ids is empty for prompt: {prompt}")
        return None
    return input_ids

# === Run predictions and extract the answer letter/digit ===
def predict(model, tokenizer, prompts, max_new_tokens=5):
    preds = []
    decodeds = []
    device = get_device(model)
    for prompt in prompts:
        input_ids = get_input_ids(prompt, tokenizer, device)

        if input_ids is None or input_ids.size(1) == 0:
            print(f"Error: input_ids is empty for prompt: {prompt}")
            preds.append("?")
            decodeds.append("")
            continue

        # Create attention mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for deterministic outputs
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = extract_choice_letter(decoded)
        preds.append(pred)
        decodeds.append(decoded)

    with open("predictions.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Prediction", "Decoded"])
        for prompt, pred, decoded in zip(prompts, preds, decodeds):
            writer.writerow([prompt, pred, decoded])
    return preds

def extract_choice_letter(text):
    if "Answer:" in text:
        completion = text.split("Answer:")[-1].strip()
        for ch in completion:
            if ch in "ABCDE0123":
                return str(ch)
    return "?"

if __name__ == "__main__":
    # Use the instruct variant of the mo del
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model_id = "Heisenger/final_runs-Reasoning_trial_1_dist_5"  # Uncomment for larger model
    model_id = "Heisenger/final_runs-No_Reasoning_trial_1_dist_5"  # Uncomment for larger model
    hellaswag_flag = False  # Set to False if you don't want to run HellaSwag

    model_list = ["Heisenger/final_runs-Reasoning_trial_1_dist_5", "Heisenger/final_runs-No_Reasoning_trial_1_dist_5"]

    for model_id in model_list: 

        model, tokenizer = load_model(model_id)

        # Load ARC-C test set (using a small subset for quick testing)
        arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        # print("First ARC-C example:", arc[0])
        arc_prompts, arc_labels = format_arc(arc)
        arc_preds = predict(model, tokenizer, arc_prompts)

        correct = sum([p == gt for p, gt in zip(arc_preds, arc_labels)])
        acc = 100 * correct / len(arc_labels)
        print(f'Moel ID: {model_id} ---------------------')
        print("\nARC-C | 0-shot | Metric: acc")
        print(f"Model Accuracy: {acc:.1f}")

        if hellaswag_flag:

            # Load HellaSwag validation set (since test split may lack labels)
            hellaswag = load_dataset("hellaswag", split="validation[:100]")
            # print("First HellaSwag example:", hellaswag[0])
            hs_prompts, hs_labels = format_hellaswag(hellaswag)
            hs_preds = predict(model, tokenizer, hs_prompts)

            if len(hs_labels) == 0:
                print("\nHellaSwag | 0-shot | Metric: acc")
                print("Warning: No valid examples in HellaSwag dataset. Accuracy cannot be computed.")
            else:
                hs_correct = sum([p == gt for p, gt in zip(hs_preds, hs_labels)])
                hs_acc = 100 * hs_correct / len(hs_labels)
                print("\nHellaSwag | 0-shot | Metric: acc")
                print(f"Model Accuracy: {hs_acc:.1f}")

            
