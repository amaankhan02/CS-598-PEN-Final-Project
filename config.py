import os

from dotenv import load_dotenv

load_dotenv()  # load the environment variables from the .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"  # has the highest RPM

DEFAULT_ENV_CONFIG = {
    "max_steps": 20,
    "topic": "Astronomy",
    "num_students": 3,
    "student_types": ["beginner", "intermediate", "advanced"],
}

DEFAULT_TRAINING_CONFIG = {
    "num_iterations": 10,
    "lr": 5e-5,
    "train_batch_size": 128, # 512 typically
    # episodes_per_iteration is not a direct parameter in RLlib 2.37.0
    # We'll set the appropriate parameters in the PPOConfig.rollouts() section instead
}

DEFAULT_LLM_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 150,
    "requests_per_minute": 25,  # Gemini-2.0-flash-lite takes 30 requests per minute, so do a bit less
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
