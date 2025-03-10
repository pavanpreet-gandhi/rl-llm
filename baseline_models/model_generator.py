from transformers import pipeline


def load_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Loads the text-to-text generation model from Hugging Face.
    Default: Qwen2.5.
    """
    print(f"Loading model: {model_name}...")
    generator = pipeline("text2text-generation", model=model_name)
    print("Model loaded successfully.\n")
    return generator
