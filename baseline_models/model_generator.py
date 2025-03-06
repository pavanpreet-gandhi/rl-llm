from transformers import pipeline


def load_model(model_name="google/flan-t5-small"):
    """
    Loads the text-to-text generation model from Hugging Face.
    Default: FLAN-T5-Small.
    """
    print(f"Loading model: {model_name}...")
    generator = pipeline("text2text-generation", model=model_name)
    print("Model loaded successfully.\n")
    return generator
