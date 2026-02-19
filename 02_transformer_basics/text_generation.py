"""
text_generation.py
------------------
Text generation using GPT-2 via the Hugging Face pipeline API.

Given a prompt, GPT-2 autoregressively samples tokens to complete the text.
Multiple prompts are demonstrated, and results are printed with formatting.

Usage:
    python text_generation.py
"""

from transformers import pipeline


def run_text_generation():
    """
    Load GPT-2 and generate continuations for a list of prompts.

    Args (configurable via constants below):
        MAX_LENGTH          : Maximum total token length of generated text.
        NUM_RETURN_SEQUENCES: Number of independent completions per prompt.
    """
    MAX_LENGTH = 60
    NUM_RETURN_SEQUENCES = 1

    print("Loading GPT-2 text-generation model (this may take a moment)...")
    text_generator = pipeline("text-generation", model="gpt2")
    print("Model loaded successfully.\n")

    prompts = [
        "Artificial intelligence will",
        "The future of deep learning is",
        "PyTorch is a powerful framework because",
    ]

    print("=" * 60)
    print("  GPT-2 Text Generation")
    print("=" * 60)

    for prompt in prompts:
        results = text_generator(
            prompt,
            max_length=MAX_LENGTH,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            truncation=True,
        )
        print(f'\n✍️  Prompt : "{prompt}"')
        for i, result in enumerate(results, start=1):
            generated = result["generated_text"]
            print(f"📝  Output {i}: \"{generated}\"")

    print("\nDone.")


if __name__ == "__main__":
    run_text_generation()
