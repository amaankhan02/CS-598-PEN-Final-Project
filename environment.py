import gymnasium as gym
import numpy as np
from ray.rllib.env import MultiAgentEnv

from agents import StudentAgent, TeacherAgent
from collections import OrderedDict


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
        self.teacher = TeacherAgent("teacher_0")
        self.student = StudentAgent("student_0")  # just defining one student for now
        self._agent_ids = {self.teacher.agent_id, self.student.agent_id}
        
        if "topic" not in config:
            raise ValueError("'topic' must be provided in the config")
        
        self.topic = config["topic"]

        # state variables
        self.current_step = 0

        # Define observation and actionspaces
        # Teacher and student observe the student's knowledge level for now and only that
        # student_obs_space = gym.spaces.Dict({"knowledge": gym.spaces.Discrete(StudentAgent._NUM_KNOWLEDGE_LEVELS)})

        self.observation_spaces = {
            self.teacher.agent_id: gym.spaces.Discrete(StudentAgent._NUM_KNOWLEDGE_LEVELS),
            self.student.agent_id: gym.spaces.Discrete(StudentAgent._NUM_KNOWLEDGE_LEVELS),
        }

        teacher_action_space = gym.spaces.Discrete(TeacherAgent.NUM_ACTIONS)
        student_action_space = gym.spaces.Discrete(StudentAgent.NUM_ACTIONS)

        self.action_spaces = gym.spaces.Dict(
            {
                self.teacher.agent_id: teacher_action_space,
                self.student.agent_id: student_action_space,
            }
        )

    def reset(self, *, seed=None, options=None):
        """Resets the environment and agents to initial states.

        Returns:
            tuple: (observation dicitionary, info dictionary)
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.teacher.reset()
        self.student.reset()
        
        observations = self._get_obs()
        infos = self._get_infos() # Get base infos
        
        # Clear any lingering LLM outputs from previous episode infos
        infos[self.teacher.agent_id].pop("last_student_question", None)
        infos[self.teacher.agent_id].pop("last_teacher_explanation", None)
        
        infos[self.student.agent_id].pop("last_student_question", None)
        infos[self.student.agent_id].pop("last_teacher_explanation", None)
        
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
        student_action = action_dict[self.student.agent_id]
        
        print(f"\n--- Step {self.current_step} ---")
        print(f"Actions: Teacher={teacher_action}, Student={student_action}")
        
        # Trigger LLM calls based on specific actions and print the output.
        # TODO: Note that the LLM output does NOT affect the state, obs, or rewards yet - must add this later
        explanation = None
        question = None
        if teacher_action == TeacherAgent.ACTION_SIMPLE:
            explanation = self.teacher.generate_explanation(self.student.knowledge, self.topic)
            print(f"[LLM Output] Teacher Explanation: {explanation}")
        elif teacher_action == TeacherAgent.ACTION_COMPLEX:
            # TODO: add a different explanation type here
            pass

        if student_action == StudentAgent.ACTION_ASK:
            question = self.student.generate_question(self.topic)
            print(f"[LLM Output] Student Question: {question}")

        old_knowledge = self.student.knowledge

        did_knowledge_increase = self.student.update_state(
            teacher_action, student_action
        )
        new_knowledge = self.student.knowledge

        # calculate rewards based on the state transition (where state = knowledge for now)
        rewards = self._calculate_rewards(
            teacher_action, student_action, old_knowledge, new_knowledge
        )

        # check for termination (goal reached) and truncation (max steps reached)
        terminate_episode = new_knowledge == StudentAgent.KNOWLEDGE_HIGH
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
            infos[self.student.agent_id]["last_teacher_explanation"] = explanation
            infos[self.teacher.agent_id]["last_teacher_explanation"] = explanation
        if question:
            infos[self.teacher.agent_id]["last_student_question"] = question
            infos[self.student.agent_id]["last_student_question"] = question
        
        # print("[A] Observations: ", observations)
        # print("[A] Infos: ", infos)

        return observations, rewards, terminated, truncated, infos

    def _get_obs(self):
        """Gets the current observations for each agent"""
        
        # for now, each teacher and student agents have the same observation
        student_knowledge = self.student.get_observation()
        # * teacher's observation is the student's knowledge for now, so just return that
        
        return {
            self.teacher.agent_id: student_knowledge,
            self.student.agent_id: student_knowledge,
        }
        

    def _get_infos(self):
        """Gets the current infos for each agent"""

        # TODO: we can expand this later
        return {
            self.teacher.agent_id: {"student_knowledge": self.student.knowledge},
            self.student.agent_id: {"self_knowledge": self.student.knowledge},
        }

    def _calculate_rewards(
        self, teacher_action, student_action, old_knowledge, new_knowledge
    ):
        """Calculates rewards based on the state transition and actions.

        TODO: move this function to its own like file/class so that we can scale this
        to different kinds of reward functions

        TODO: improve this reward function calculation to be more complex
        """
        rewards = {self.teacher.agent_id: 0.0, self.student.agent_id: 0.0}

        # --- Core Reward: Knowledge Improvement ---
        if new_knowledge > old_knowledge:
            rewards[self.student.agent_id] += 1.0
            rewards[self.teacher.agent_id] += 1.0

        # --- Contextual Teacher Rewards/Penalties ---
        if teacher_action == TeacherAgent.ACTION_COMPLEX and old_knowledge == StudentAgent.KNOWLEDGE_LOW:
            rewards[self.teacher.agent_id] -= 0.5    # Penalize teaching complex too early
        elif teacher_action == TeacherAgent.ACTION_SIMPLE and old_knowledge == StudentAgent.KNOWLEDGE_LOW:
            rewards[self.teacher.agent_id] += 0.1 # Small bonus for teaching simply to a beginner

        # --- Contextual Student Rewards/Penalties ---
        if student_action == StudentAgent.ACTION_ASK:
            if old_knowledge == StudentAgent.KNOWLEDGE_LOW:
                rewards[self.student.agent_id] += 0.2 # Encourage beginners to ask
            else:
                rewards[self.student.agent_id] += 0.05 # Small reward for asking otherwise
        # Optionally add small reward for ACTION_STUDY? - nah cuz like what does it even mean to study... 
        # elif student_action == StudentAgent.ACTION_STUDY:
        #     rewards[self.student.agent_id] += 0.05

        # --- Goal Achievement Bonus ---
        if (new_knowledge == StudentAgent.KNOWLEDGE_HIGH and
            old_knowledge < StudentAgent.KNOWLEDGE_HIGH):
            rewards[self.teacher.agent_id] += 5.0
            rewards[self.student.agent_id] += 5.0

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
