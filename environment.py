from collections import OrderedDict

import gymnasium as gym
import numpy as np
from ray.rllib.env import MultiAgentEnv

from agents import StudentAgent, TeacherAgent


class ClassroomEnv(MultiAgentEnv):
    """A simple classroom environment for teaching and learning"""

    def __init__(self, config):
        """Initializes the classroom environment with agent
        instances.

        Args:
            config (dict): Configuration for the environment
        """
        super().__init__()

        self.config = config
        self.max_steps = config.get("max_steps", 20)

        # Required configs
        if "topic" not in config:
            raise ValueError("'topic' must be provided in the config")
        self.topic = config["topic"]
        self.teacher = TeacherAgent("teacher_0")
        self.num_students = config.get("num_students", 1)
        student_types = config.get("student_types", ["beginner"] * self.num_students)

        # If not enough student types are provided, default remaining students to "beginner"
        if len(student_types) < self.num_students:
            student_types.extend(
                ["beginner"] * (self.num_students - len(student_types))
            )

        # Initialize students with their respective types
        self.students = {}
        for i in range(self.num_students):
            student_id = f"student_{i}"
            student_type = student_types[i]
            self.students[student_id] = StudentAgent(student_id, student_type)

        # Collect all agent IDs
        self._agent_ids = {self.teacher.agent_id}
        self._agent_ids.update(self.students.keys())

        # state variables
        self.current_step = 0

        # Define observation spaces and action spaces for all agents
        self.observation_spaces = {}

        # Teacher observes all students' knowledge levels
        if self.num_students == 1:
            # If there's only one student, keep the simple observation space
            self.observation_spaces[self.teacher.agent_id] = gym.spaces.Discrete(
                StudentAgent._NUM_KNOWLEDGE_LEVELS
            )
        else:
            # For multiple students, teacher observes an array of knowledge levels
            self.observation_spaces[self.teacher.agent_id] = gym.spaces.Box(
                low=0,
                high=StudentAgent._NUM_KNOWLEDGE_LEVELS - 1,
                shape=(self.num_students,),
                dtype=np.int32,
            )

        # Each student observes their own knowledge level
        for student_id in self.students:
            self.observation_spaces[student_id] = gym.spaces.Discrete(
                StudentAgent._NUM_KNOWLEDGE_LEVELS
            )

        self.action_spaces = {}

        self.action_spaces[self.teacher.agent_id] = gym.spaces.Discrete(
            TeacherAgent.NUM_ACTIONS
        )

        for student_id in self.students:
            self.action_spaces[student_id] = gym.spaces.Discrete(
                StudentAgent.NUM_ACTIONS
            )

    def reset(self, *, seed=None, options=None):
        """Resets the environment and agents to initial states.

        Returns:
            tuple: (observation dicitionary, info dictionary)
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.teacher.reset()
        for student in self.students.values():
            student.reset()

        observations = self._get_obs()
        infos = self._get_infos()  # Get base infos

        # Clear any lingering LLM outputs from previous episode infos
        # infos[self.teacher.agent_id].pop("last_student_question", None)
        infos[self.teacher.agent_id].pop("last_teacher_explanation", None)

        for student_id in self.students:
            infos[student_id].pop("last_student_question", None)
            # infos[student_id].pop("last_teacher_explanation", None)

        return observations, infos

    def step(self, action_dict):
        """Advances the environment by one step.

        Args:
            action_dict (dict): Actions from each agent, e.g., {"teacher_0": 0, "student_0": 1}

        Returns:
            tuple: (observations, rewards, terminated flag, truncated flag, infos as dict)
        """

        self.current_step += 1

        teacher_action = action_dict[self.teacher.agent_id]

        print(f"\n--- Step {self.current_step} ---")
        print(f"Teacher Action: {teacher_action}")

        # Store old knowledge levels for reward calculation
        old_knowledge = {
            student_id: student.knowledge
            for student_id, student in self.students.items()
        }

        # Track if any knowledge increased and capture student actions
        knowledge_increased = {student_id: False for student_id in self.students}
        student_actions = {}

        # Process student actions and update their states
        for student_id, student in self.students.items():
            if student_id in action_dict:
                student_action = action_dict[student_id]
                student_actions[student_id] = student_action
                print(f"{student_id} Action: {student_action}")

                # Update student state based on teacher's action and their own action
                knowledge_increased[student_id] = student.update_state(
                    teacher_action, student_action
                )
            else:
                print(f"Warning: No action provided for {student_id}")

        # Capture new knowledge levels
        new_knowledge = {
            student_id: student.knowledge
            for student_id, student in self.students.items()
        }

        # Trigger LLM calls based on specific actions
        explanation = None
        questions = {}

        if teacher_action == TeacherAgent.ACTION_SIMPLE:
            # TODO: area of improvement - generate personalized explanations
            # For simplicity, teacher generates one explanation for all students
            # Could be extended to generate personalized explanations
            avg_knowledge = sum(
                student.knowledge for student in self.students.values()
            ) / len(self.students)
            explanation = self.teacher.generate_explanation(
                round(avg_knowledge), self.topic
            )
            # TODO: see if generate_explanation() with avg_knowledge would work like this
            print(f"[LLM Output] Teacher Explanation: {explanation}")
        elif teacher_action == TeacherAgent.ACTION_COMPLEX:
            # TODO: add a different explanation type here
            pass

        for student_id, student in self.students.items():
            if (
                student_id in student_actions
                and student_actions[student_id] == StudentAgent.ACTION_ASK
            ):
                questions[student_id] = student.generate_question(self.topic)
                print(f"[LLM Output] {student_id} Question: {questions[student_id]}")

        # Calculate rewards based on the state transitions
        rewards = self._calculate_rewards(
            teacher_action, student_actions, old_knowledge, new_knowledge
        )

        # Check for termination (all students reached high knowledge) and truncation (max steps reached)
        # TODO: area of improvement - do we need all students to reach high knowledge?
        all_students_high_knowledge = all(
            knowledge == StudentAgent.KNOWLEDGE_HIGH
            for knowledge in new_knowledge.values()
        )
        terminate_episode = all_students_high_knowledge
        truncate_episode = self.current_step >= self.max_steps

        terminated = {agent_id: terminate_episode for agent_id in self._agent_ids}
        truncated = {agent_id: truncate_episode for agent_id in self._agent_ids}
        # ! ^^ set individual flags for now. TODO: see if we need to change this

        terminated["__all__"] = terminate_episode
        truncated["__all__"] = truncate_episode

        observations = self._get_obs()
        infos = self._get_infos()

        # add the last_teacher_explanation and last_student_question to infos
        if explanation:
            # we don't need to store the teacher explanation for each student anymore since its the same explanation for all
            # infos[self.student.agent_id]["last_teacher_explanation"] = explanation
            infos[self.teacher.agent_id]["last_teacher_explanation"] = explanation

        for student_id, question in questions.items():
            # every student has a different question so we don't need to store it in the teacher's info
            # infos[self.teacher.agent_id][
            #     "last_student_question"
            # ] = question  # Teacher gets the last question asked
            infos[student_id]["last_student_question"] = question

        return observations, rewards, terminated, truncated, infos

    def _get_obs(self):
        """Gets the current observations for each agent"""

        observations = {}

        # Teacher's observation depends on the number of students
        if self.num_students == 1:
            # For a single student, the teacher observes that student's knowledge level
            student_id = next(iter(self.students))
            observations[self.teacher.agent_id] = self.students[
                student_id
            ].get_observation()
        else:
            # For multiple students, the teacher observes an array of knowledge levels
            observations[self.teacher.agent_id] = np.array(
                [student.get_observation() for student in self.students.values()]
            )

        # Each student observes their own knowledge level
        for student_id, student in self.students.items():
            observations[student_id] = student.get_observation()

        return observations

    def _get_infos(self):
        """Gets the current infos for each agent"""

        infos = {}

        # Teacher's info includes knowledge levels of all students
        teacher_info = {
            f"{student_id}_knowledge": student.knowledge
            for student_id, student in self.students.items()
        }
        teacher_info["class_avg_knowledge"] = sum(
            student.knowledge for student in self.students.values()
        ) / len(self.students)
        infos[self.teacher.agent_id] = teacher_info

        # Each student's info includes their own knowledge and type
        for student_id, student in self.students.items():
            infos[student_id] = {
                "self_knowledge": student.knowledge,
                "student_type": student.student_type,
            }
        # TODO: do we also want to add the last_student_question and last_teacher_explanation to the infos?
        return infos

    def _calculate_rewards(
        self, teacher_action, student_actions, old_knowledge, new_knowledge
    ):
        """Calculates rewards based on the state transition and actions.

        Args:
            teacher_action (int): The action taken by the teacher
            student_actions (dict): Dictionary mapping student IDs to their actions
            old_knowledge (dict): Dictionary mapping student IDs to their previous knowledge levels
            new_knowledge (dict): Dictionary mapping student IDs to their current knowledge levels

        Returns:
            dict: Dictionary mapping agent IDs to their rewards
        """
        rewards = {agent_id: 0.0 for agent_id in self._agent_ids}

        # Calculate average knowledge improvement for teacher reward
        total_improvement = 0
        num_students_improved = 0

        for student_id in self.students:
            # --- Core Reward: Knowledge Improvement ---
            knowledge_improved = new_knowledge[student_id] > old_knowledge[student_id]
            if knowledge_improved:
                # Student gets reward for improving
                rewards[student_id] += 1.0
                total_improvement += 1
                num_students_improved += 1

            # --- Contextual Student Rewards/Penalties ---
            if student_id in student_actions:
                student_action = student_actions[student_id]
                if student_action == StudentAgent.ACTION_ASK:
                    if old_knowledge[student_id] == StudentAgent.KNOWLEDGE_LOW:
                        rewards[student_id] += 0.2  # Encourage beginners to ask
                    else:
                        rewards[student_id] += 0.05  # Small reward for asking otherwise

            # --- Goal Achievement Bonus ---
            if (
                new_knowledge[student_id] == StudentAgent.KNOWLEDGE_HIGH
                and old_knowledge[student_id] < StudentAgent.KNOWLEDGE_HIGH
            ):
                rewards[student_id] += 5.0  # Big bonus for reaching high knowledge

        # --- Teacher Rewards ---
        # Base reward based on how many students improved
        if num_students_improved > 0:
            # Scale reward by percentage of students who improved
            improvement_percentage = num_students_improved / len(self.students)
            rewards[self.teacher.agent_id] += improvement_percentage * 1.0

        # Contextual teacher rewards/penalties based on class composition
        low_knowledge_count = sum(
            1 for k in old_knowledge.values() if k == StudentAgent.KNOWLEDGE_LOW
        )
        high_knowledge_count = sum(
            1 for k in old_knowledge.values() if k == StudentAgent.KNOWLEDGE_HIGH
        )

        # Penalize using complex teaching when most students are beginners
        if (
            teacher_action == TeacherAgent.ACTION_COMPLEX
            and low_knowledge_count > len(self.students) / 2
        ):
            rewards[self.teacher.agent_id] -= 0.5

        # Reward using simple teaching when most students are beginners
        elif (
            teacher_action == TeacherAgent.ACTION_SIMPLE
            and low_knowledge_count > len(self.students) / 2
        ):
            rewards[self.teacher.agent_id] += 0.2

        # Goal achievement bonus for teacher when all students reach high knowledge
        if all(
            k == StudentAgent.KNOWLEDGE_HIGH for k in new_knowledge.values()
        ) and not all(k == StudentAgent.KNOWLEDGE_HIGH for k in old_knowledge.values()):
            rewards[self.teacher.agent_id] += 5.0

        return rewards


# for testing. main training script is in train.py
if __name__ == "__main__":
    env = ClassroomEnv({})
    obs, infos = env.reset()
    print("Initial Observations: ", obs)
    print("Initial Infos: ", infos)

    terminated = {"__all__": False}
    truncated = {"__all__": False}
    step_count = 0

    # running a sample episode
    try:
        while not terminated["__all__"] and not truncated["__all__"]:
            step_count += 1
            action_dict = env.action_space.sample()

            print(f"\n--- Step {step_count} ---")
            print(f"Actions: {action_dict}")

            obs, rewards, terminated, truncated, infos = env.step(action_dict)

            print(f"Observations: {obs}")
            print(f"Rewards: {rewards}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Infos: {infos}")  # Now includes knowledge in infos

            if terminated["__all__"]:
                print("\nEpisode terminated (goal reached).")
            if truncated["__all__"]:
                print("\nEpisode truncated (max steps reached).")

        print(f"\nEpisode finished after {step_count} steps.")
    except Exception as e:
        print(f"Error occured: {e}")
