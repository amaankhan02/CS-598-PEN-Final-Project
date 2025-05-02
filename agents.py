from collections import OrderedDict

import numpy as np

from utils import get_gemini_response


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

    def generate_explanation(self, student_knowledge_level, topic):
        """Generates an explanation for the given topic based on the student's knowledge level"""

        level_map = {
            StudentAgent.KNOWLEDGE_LOW: "complete beginner",
            StudentAgent.KNOWLEDGE_MEDIUM: "basic",
            StudentAgent.KNOWLEDGE_HIGH: "very high",
        }
        level_desc = level_map.get(student_knowledge_level, "unknown")

        # Simple prompt example
        prompt = (
            f"You are an expert teacher. Explain the core concept of '{topic}' "
            f"in one or two simple sentences for a student who has a {level_desc} level of knowledge/understanding."
        )

        print(f"\n[{self.agent_id}] Generating explanation with prompt: '{prompt}'")
        explanation = get_gemini_response(prompt)
        print(f"[{self.agent_id}] Received explanation: '{explanation}'")
        return explanation


class StudentAgent:
    """Represents the student agent in the environment"""

    # constants for possible student actions - again very simple for now
    ACTION_STUDY = 0
    ACTION_ASK = 1

    NUM_ACTIONS = (
        2  # TODO: make this dynamic based on the length of the action list or whatever
    )

    # constants for knowledge levels
    KNOWLEDGE_LOW = 0
    KNOWLEDGE_MEDIUM = 1
    KNOWLEDGE_HIGH = 2

    _NUM_KNOWLEDGE_LEVELS = 3  # used for observation space def

    # Define student types with their learning coefficients and initial knowledge
    # TODO: change the knowledge and learning coefficients to be more realistic
    STUDENT_TYPES = {
        "beginner": {"learning_coef": 0.8, "initial_knowledge": KNOWLEDGE_LOW},
        "intermediate": {"learning_coef": 1.0, "initial_knowledge": KNOWLEDGE_MEDIUM},
        "advanced": {"learning_coef": 1.2, "initial_knowledge": KNOWLEDGE_MEDIUM},
        "gifted": {"learning_coef": 1.5, "initial_knowledge": KNOWLEDGE_LOW},
        "struggling": {"learning_coef": 0.5, "initial_knowledge": KNOWLEDGE_LOW},
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

        self.student_type = student_type
        self.learning_coef = self.STUDENT_TYPES[student_type]["learning_coef"]
        self.initial_knowledge = self.STUDENT_TYPES[student_type]["initial_knowledge"]
        self.knowledge = self.initial_knowledge

    def reset(self):
        """Resets the student's knowledge to the initial level"""
        self.knowledge = self.initial_knowledge

    def update_state(self, teacher_action, student_action) -> bool:
        """Updates the student's knowledge state based on teacher and self actions.

        Args:
            teacher_action (int): the action taken by the teacher
            student_action (int): the action taken by this student

        Returns:
            bool: True if knowledge increased, False otherwise
        """
        # TODO: all of this logic needs to changed to be more realistic. like a proper gameplan needs to be made
        knowledge_increased = False

        # Base learning chance - will be modified by learning coefficient
        learning_chance = 0.0

        if student_action == self.ACTION_STUDY:
            if teacher_action == TeacherAgent.ACTION_SIMPLE:
                # Simple teaching helps if knowledge is not maxed out
                if self.knowledge < self.KNOWLEDGE_HIGH:
                    learning_chance = 1.0  # Base chance for simple teaching
            elif teacher_action == TeacherAgent.ACTION_COMPLEX:
                # Complex teaching helps only if student has medium/high knowledge
                if (
                    self.knowledge >= self.KNOWLEDGE_MEDIUM
                    and self.knowledge < self.KNOWLEDGE_HIGH
                ):
                    learning_chance = 1.0  # Base chance for complex teaching
                elif self.knowledge < self.KNOWLEDGE_MEDIUM:
                    learning_chance = (
                        0.1  # Small chance to learn complex material for beginners
                    )

        # Apply student learning coefficient
        final_chance = learning_chance * self.learning_coef

        # Determine if knowledge increases
        if final_chance >= 1.0 or (
            final_chance > 0 and np.random.random() < final_chance
        ):
            if self.knowledge < self.KNOWLEDGE_HIGH:
                self.knowledge += 1
                knowledge_increased = True

        # Asking questions currently doesn't directly change knowledge state
        # but could be expanded in the future to provide small knowledge boosts
        elif student_action == self.ACTION_ASK:
            # Advanced students might learn just from asking good questions
            if self.student_type in ["advanced", "gifted"] and np.random.random() < 0.2:
                if self.knowledge < self.KNOWLEDGE_HIGH:
                    self.knowledge += 1
                    knowledge_increased = True

        return knowledge_increased

    def get_observation(self):
        """Returns the student's observable state (their knowledge level)"""
        return self.knowledge

    def generate_question(self, topic):
        """Generates a question for the given topic the student is learning about"""

        # Knowledge level descriptions
        level_map = {
            self.KNOWLEDGE_LOW: "I'm a complete beginner and don't understand much yet",
            self.KNOWLEDGE_MEDIUM: "I have some basic understanding but need clarification",
            self.KNOWLEDGE_HIGH: "I understand the basics but want to explore deeper",
        }
        level_desc = level_map.get(
            self.knowledge, "I have an unknown level of understanding"
        )

        # Add student type specific behavior to prompt
        type_desc = ""
        if self.student_type == "advanced":
            type_desc = "I tend to ask detailed, technical questions."
        elif self.student_type == "beginner":
            type_desc = "I need clear, simple explanations."
        elif self.student_type == "intermediate":
            type_desc = "I understand basics but need help connecting concepts."
        elif self.student_type == "gifted":
            type_desc = "I'm quick to understand but like to explore connections between concepts."
        elif self.student_type == "struggling":
            type_desc = "I sometimes have trouble following complex explanations."

        # Simple prompt example
        prompt = (
            f"You are a student learning about '{topic}'. {level_desc}. {type_desc} "
            f"Ask one specific, relevant question to your teacher to improve your understanding."
        )

        print(f"\n[{self.agent_id}] Generating question with prompt: '{prompt}'")
        question = get_gemini_response(prompt)
        print(f"[{self.agent_id}] Received question: '{question}'")
        return question
