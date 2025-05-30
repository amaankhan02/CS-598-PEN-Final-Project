import os

from dotenv import load_dotenv

load_dotenv()  # load the environment variables from the .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"  # has the highest RPM
METRICS_DIR = "metrics/testing"

DEFAULT_ENV_CONFIG = {
    "max_steps": 10,
    "topic": "Theoretical Physics",
    "num_students": 3,
    "student_types": ["beginner", "intermediate", "advanced"],
    "teacher_reward_weights": [7.5, 0.01, 0.25, 1.0, 0.001] # [progress, equity, engagement, explanation‑quality, time‑penalty]
}
LOG_DIR = "logs"
LOG_FILE_NAME = f"{LOG_DIR}/log_experiment_3.txt"  # this will be changed by train.py
# episodes_per_iteration = ceil(train_batch_size / max_steps)
DEFAULT_TRAINING_CONFIG = {
    "num_iterations": 5,
    "lr": 8e-5,
    "train_batch_size": 32,
    "sgd_minibatch_size": 16,
    "num_sgd_iter": 10,
}

DEFAULT_LLM_CONFIG = {
    "temperature": 0.9,
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
