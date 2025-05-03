import time

import google.generativeai as genai

import config
import re

class LLMClient:
    """
    A client class to handle interactions with the configured Gemini LLM.
    """

    def __init__(
        self, api_key, model_name, temperature=0.7, top_p=0.9, max_output_tokens=150
    ):
        """
        Initializes the LLM client, configures the API, and creates the model instance.

        Args:
            api_key (str): The Google API key.
            model_name (str): The name of the Gemini model to use.
            temperature (float): Controls randomness in generation.
            top_p (float): Controls diversity via nucleus sampling.
            max_output_tokens (int): Maximum number of tokens in the response.
        """
        if not api_key:
            raise ValueError("API key is required for LLMClient.")

        self.model_name = model_name
        self.model = None

        # Rate limiting properties
        self.requests_per_minute = config.DEFAULT_LLM_CONFIG.get(
            "requests_per_minute", 25
        )
        self.request_interval = 60.0 / self.requests_per_minute
        self.last_request_time = 0

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            print(
                f"Gemini model '{self.model_name}' initialized successfully in LLMClient."
            )
        except Exception as e:
            print(f"Error configuring Gemini or initializing model in LLMClient: {e}")
            # Allow initialization to continue, but get_response will fail gracefully

        self.generation_config = genai.GenerationConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens
        )

    def _wait_for_rate_limit(self):
        """
        Implements a simple rate limiter using the token bucket algorithm.
        Waits if necessary to respect the configured requests per minute.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.request_interval:
            wait_time = self.request_interval - time_since_last_request
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def get_response(self, prompt, max_retries=3, delay=2, specific_gen_config=None):
        """
        Sends a prompt to the configured Gemini model and returns the response.

        Args:
            prompt (str): The text prompt to send to the LLM.
            max_retries (int): Maximum number of retries in case of API errors.
            delay (int): Delay in seconds between retries.

        Returns:
            str: The generated text response from the LLM, or an error message string
                 if the request fails after retries or if the model wasn't initialized.
        """
        if self.model is None:
            return "[Error: LLM Model not initialized]"
        
        current_gen_config = specific_gen_config if specific_gen_config else self.generation_config

        retries = 0
        while retries < max_retries:
            try:
                self._wait_for_rate_limit()

                response = self.model.generate_content(
                    prompt,
                    generation_config=current_gen_config,
                )
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text.strip()
                else:
                    feedback = (
                        response.prompt_feedback
                        if hasattr(response, "prompt_feedback")
                        else "No feedback available"
                    )
                    finish_reason = (
                        response.candidates[0].finish_reason
                        if response.candidates
                        else "No candidates"
                    )
                    return f"[LLM Error: Empty response or blocked content. Feedback: {feedback}, Finish Reason: {finish_reason}]"

            except Exception as e:
                retries += 1
                error_message = str(e).lower()

                # Check if this is a rate limit error
                if (
                    "rate limit" in error_message
                    or "quota" in error_message
                    or "429" in error_message
                ):
                    retry_delay = delay * 2  # Wait longer for rate limit errors
                    print(
                        f"Rate limit exceeded. Waiting {retry_delay} seconds before retry..."
                    )
                    time.sleep(retry_delay)
                else:
                    print(
                        f"Error calling Gemini API (Attempt {retries}/{max_retries}): {e}"
                    )
                    if retries >= max_retries:
                        return f"[Error: LLM API call failed after {max_retries} retries - {e}]"
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        return "[Error: LLM call failed unexpectedly]"


# --- Global Instance ---
# Create a single instance of the client for the application to use.
# This assumes one central LLM configuration. If different parts need
# different configs, you might instantiate the client where needed instead.
try:
    llm_client = LLMClient(
        api_key=config.GOOGLE_API_KEY,
        model_name=config.GEMINI_MODEL_NAME,
        temperature=config.DEFAULT_LLM_CONFIG["temperature"],
        top_p=config.DEFAULT_LLM_CONFIG["top_p"],
        max_output_tokens=config.DEFAULT_LLM_CONFIG["max_output_tokens"],
    )
except ValueError as e:
    print(f"Failed to create global LLMClient instance: {e}")
    llm_client = None  # Ensure it's None if creation fails

def analyze_explanation_quality(text, topic="the topic"):
    """
    Classifies a teacher explanation on Bloom's 1‑6 scale, returns int.
    """
    prompt = f"""
    Evaluate the following teacher explanation of '{topic}'.
    Only output the Bloom's Taxonomy level (1‑6) that best describes
    the cognitive complexity of the explanation.

    Explanation: \"\"\"{text}\"\"\"

    Integer Level:"""
    resp = get_gemini_response(prompt, max_retries=3, delay=2)
    m = re.search(r"\b([1-6])\b", resp)
    return int(m.group(1)) if m else 1


def get_gemini_response(prompt, max_retries=3, delay=2):
    """
    Helper function that uses the global LLMClient instance.
    Provided here so that other modules don't necessarily need to know about the client object
    and can just use this helper function.
    """
    if llm_client:
        return llm_client.get_response(prompt, max_retries, delay)
    else:
        return "[Error: Global LLM Client not available]"

def analyze_text_for_bloom(text_to_analyze, student_level_desc="unknown", topic="the topic"):
    """
    Uses the LLM to estimate the Bloom's Taxonomy level of the provided text.

    Args:
        text_to_analyze (str): The student's question or statement.
        student_level_desc (str): Description of the student's current Bloom level.
        topic (str): The topic being discussed.

    Returns:
        int: The estimated Bloom's level (1-6), or 1 as a fallback on error.
    """
    if not llm_client:
        print("[Warning] LLM Client not available for Bloom analysis.")
        return 1 # Fallback level

    prompt = f"""
    Analyze the following student's text, generated while learning about '{topic}'.
    The student is currently thinking at the '{student_level_desc}' stage (Bloom's Taxonomy).

    Student Text: "{text_to_analyze}"

    Based *only* on the text provided, estimate which level of Bloom's Taxonomy (1-6) it best represents:
    1: Remembering (Recall facts, basic concepts)
    2: Understanding (Explain ideas, concepts)
    3: Applying (Use information in new situations)
    4: Analyzing (Draw connections among ideas)
    5: Evaluating (Justify a stand or decision)
    6: Creating (Produce new or original work)

    Respond with ONLY the integer number (1, 2, 3, 4, 5, or 6). Do not add any other text or explanation.
    Integer Level: """ # Prompt designed for easy parsing

    # Use a slightly different generation config for classification - lower temperature
    analysis_gen_config = genai.GenerationConfig(temperature=0.2, max_output_tokens=5)
    response = llm_client.get_response(prompt, specific_gen_config=analysis_gen_config)

    # Attempt to parse the integer response
    try:
        # Find the first integer in the response string
        match = re.search(r'\d+', response)
        if match:
            level = int(match.group(0))
            # Clamp the value between 1 and 6
            level = max(1, min(level, 6))
            print(f"[LLM Analysis] Text: '{text_to_analyze[:50]}...' -> Estimated Bloom Level: {level}")
            return level
        else:
            print(f"[Warning] Could not parse Bloom level integer from LLM response: '{response}'. Defaulting to 1.")
            return 1 # Fallback level
    except Exception as e:
        print(f"[Error] Failed to parse Bloom level from LLM response '{response}': {e}. Defaulting to 1.")
        return 1 # Fallback level



def test_llm():
    print("\nTesting LLMClient instance...")
    if llm_client:
        test_prompt = (
            "Explain the concept of multi-agent reinforcement learning in one sentence."
        )
        print(f"Sending prompt: '{test_prompt}'")
        response = llm_client.get_response(test_prompt)
        print("\nReceived response:")
        print(response)
    else:
        print("Skipping test, LLMClient instance is not available.")

    print("\nTesting helper function...")
    response_helper = get_gemini_response("What is the capital of France?")
    print(f"Response via helper: {response_helper}")


# Testing
if __name__ == "__main__":
    test_llm()
