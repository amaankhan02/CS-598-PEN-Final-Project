import os
from dotenv import load_dotenv

load_dotenv()   # load the environment variables from the .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite" # has the highest RPM

DEFAULT_ENV_CONFIG = {
    "max_steps": 20,
    "topic": "Astronomy",
}

DEFAULT_TRAINING_CONFIG = {
    "num_iterations": 25,
    "lr": 5e-5,
    "train_batch_size": 512,
}

DEFAULT_LLM_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 150,
}


def validate_config():
    """Basic validation for essential configurations."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please ensure it's set in your "
            ".env file or environment variables."
        )
    print("Configuration loaded successfully.")
    print(f"  Using Gemini Model: {GEMINI_MODEL_NAME}")

# Automatically validate when this module is imported
validate_config()