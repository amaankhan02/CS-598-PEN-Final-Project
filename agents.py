from collections import OrderedDict

import numpy as np

from utils import get_gemini_response

BLOOM_LEVEL_MAP = {
    1: "Remembering basic facts",
    2: "Understanding concepts",
    3: "Applying knowledge",
    4: "Analyzing information",
    5: "Evaluating ideas",
    6: "Creating new work",
}   
class TeacherAgent:

    # constants for the possible teacher actions. For now its just 2 actions
    # but we can have more complex actions later on
    ACTION_SIMPLE = 0
    ACTION_COMPLEX = 1

    # TODO: make this dynamic based on the length of the action list or whatever
    NUM_ACTIONS = 2

    def __init__(self, agent_id):
        """Initializes the teacher agent

        Args:
            agent_id (str): A unique identifier for this agent.
        """
        self.agent_id = agent_id

    def reset(self):
        """Resets the teacher's internal state (if any) at the start
        of an episode
        """
        # No internal state for now so skip
        pass

    def generate_explanation(self, student_bloom_level, topic):
        """Generates an explanation tailored to the student's Bloom level using the LLM."""
        # Map Bloom level integer to descriptive string for the prompt
        level_desc = BLOOM_LEVEL_MAP.get(student_bloom_level, "at an unknown cognitive level")

        prompt = (
            f"You are an expert teacher. Explain the core concept of '{topic}' "
            f"in a concise and brief manner (less than 50 words) for a student who is currently at the '{level_desc}' stage (Bloom's Taxonomy)."
        )
        # TODO: add few shot examples for bloom levels or explanation on what they mean so that the explanatino is better?
        # TODO: update prommpt to include ZPD alignment
        
        explanation = get_gemini_response(prompt)
        return explanation


# TODO: add feature to store the student's past questions
# TODO: add feature to store like student's "notes" and that should be fed into each question. 
# TODO: add feature where the teacher quizzes the student at the end of an episode? maybe? maybe based on the teachers quiz and rubric?
class StudentAgent:
    """Represents the student agent in the environment"""

    # constants for possible student actions - again very simple for now
    ACTION_STUDY = 0
    ACTION_ASK = 1

    NUM_ACTIONS = (
        2  # TODO: make this dynamic based on the length of the action list or whatever
    )

    # constants for knowledge levels
    BLOOM_REMEMBER = 1
    BLOOM_UNDERSTAND = 2
    BLOOM_APPLY = 3
    BLOOM_ANALYZE = 4
    BLOOM_EVALUATE = 5
    BLOOM_CREATE = 6
    NUM_BLOOM_LEVELS = 6 # Max level

    # _NUM_KNOWLEDGE_LEVELS = 3  # used for observation space def

    # Define student types with their learning coefficients and initial knowledge
    # TODO: change the knowledge and learning coefficients to be more realistic
    STUDENT_TYPES = {  # Type: (learning_coefficient, initial_bloom_level)
        "beginner":     (0.8, BLOOM_REMEMBER),
        "intermediate": (1.0, BLOOM_UNDERSTAND),
        "advanced":     (1.2, BLOOM_APPLY),
        "gifted":       (1.5, BLOOM_UNDERSTAND), # Gifted might start understanding quickly
        "struggling":   (0.6, BLOOM_REMEMBER),
    }

    def __init__(self, agent_id, student_type="beginner"):
        """Initializes the student agent

        Args:
            agent_id (str): a unique identifier for this agent
            student_type (str, optional): The type of student (beginner, intermediate, advanced, etc.).
                Defaults to "beginner".
        """

        self.agent_id = agent_id

        # Set student type and get its properties
        if student_type not in self.STUDENT_TYPES:
            print(
                f"Warning: Unknown student type '{student_type}'. Defaulting to 'beginner'"
            )
            student_type = "beginner"

        # self.student_type = student_type
        # self.learning_coef = self.STUDENT_TYPES[student_type]["learning_coef"]
        # self.initial_knowledge = self.STUDENT_TYPES[student_type]["initial_knowledge"]
        # self.knowledge = self.initial_knowledge
        self.student_type = student_type
        type_props = self.STUDENT_TYPES[student_type]
        self.learning_coef = type_props[0]
        self.initial_bloom_level = type_props[1]
        self.current_bloom_level = self.initial_bloom_level
        self.last_question_bloom_level = self.BLOOM_REMEMBER # Default to lowest
        

    def reset(self):
        """Resets the student's knowledge to the initial level"""
        self.current_bloom_level = self.initial_bloom_level
        self.last_question_bloom_level = self.BLOOM_REMEMBER
        
    def _get_bloom_advancement_probability(self, teacher_action):
        """
        Determines the probability of advancing one Bloom level based on type,
        current level, and teacher action. More complex actions are needed for higher levels.
        """
        # Cannot advance if already at max level
        if self.current_bloom_level >= self.NUM_BLOOM_LEVELS:
            return 0.0
        # TODO: these probabilities are arbitrary - change this to be directly dependent on their bloom level and other things 
        # * this can leave some students behind or cause slow learning. so change these values to be dependent on something to improve the results and speed of learning
        base_prob = 0.0
        # Define effectiveness of teacher actions based on current student Bloom level
        if teacher_action == TeacherAgent.ACTION_SIMPLE:
            # Simple actions are most effective for moving from Remember -> Understand or Understand -> Apply
            if self.current_bloom_level <= self.BLOOM_UNDERSTAND:
                base_prob = 0.7
            elif self.current_bloom_level == self.BLOOM_APPLY:
                base_prob = 0.3 # Less effective for higher levels
            else:
                base_prob = 0.1 # Least effective

        elif teacher_action == TeacherAgent.ACTION_COMPLEX:
            # Complex actions are needed for higher levels
            if self.current_bloom_level == self.BLOOM_REMEMBER:
                base_prob = 0.1 # Not very effective for beginners
            elif self.current_bloom_level == self.BLOOM_UNDERSTAND:
                base_prob = 0.4
            elif self.current_bloom_level == self.BLOOM_APPLY:
                base_prob = 0.8 # Effective for Apply -> Analyze
            elif self.current_bloom_level >= self.BLOOM_ANALYZE:
                base_prob = 0.9 # Very effective for Analyze -> Evaluate -> Create

        # Apply student's learning coefficient
        final_prob = base_prob * self.learning_coef
        return np.clip(final_prob, 0.0, 1.0) # Ensure probability is between 0 and 1

    def update_state(self, teacher_action, student_action) -> bool:
        """Updates the student's Bloom level probabilistically based on actions.

        Args:
            teacher_action (int): the action taken by the teacher
            student_action (int): the action taken by this student

        Returns:
            bool: True if knowledge increased, False otherwise
        """
        level_advanced = False
        if student_action == self.ACTION_STUDY:
            advance_prob = self._get_bloom_advancement_probability(teacher_action)
            # Probabilistic advancement
            if np.random.rand() < advance_prob:
                 if self.current_bloom_level < self.NUM_BLOOM_LEVELS:
                    self.current_bloom_level += 1
                    level_advanced = True

        # Asking questions currently doesn't change Bloom level directly
        # (but will be rewarded based on its own Bloom level)
        elif student_action == self.ACTION_ASK:
            # TODO
            pass

        return level_advanced

    def get_observation(self):
        """Returns the student's observable state (their knowledge level)"""
        return self.current_bloom_level

    def generate_question(self, topic="reinforcement learning"):
        """Generates a question based on the student's type and Bloom level using the LLM."""
        # Map Bloom level integer to descriptive string
        level_desc = BLOOM_LEVEL_MAP.get(self.current_bloom_level, "at an unknown cognitive level")

        type_desc_map = {
            "beginner": "As a beginner student,",
            "intermediate": "As an intermediate student,",
            "advanced": "As an advanced student,",
            "gifted": "As a gifted student,",
            "struggling": "As a struggling student,"
        }
        type_desc = type_desc_map.get(self.student_type, "As a student,")

        prompt = (
            f"You are a student learning about '{topic}'. You are currently thinking at the '{level_desc}' stage (Bloom's Taxonomy). "
            f"{type_desc} Ask one specific, relevant question to your teacher that demonstrates this level of thinking "
            f"and helps you move towards the next level. The question should be less than 15 words maximum."
            # Optional: Add constraint to encourage higher-level questions if appropriate
            # f" Try to ask a question that involves [Applying/Analyzing/Evaluating/Creating] if possible."
        )

        question = get_gemini_response(prompt)
        return question
