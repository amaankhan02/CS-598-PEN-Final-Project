from collections import OrderedDict

import gymnasium as gym
import numpy as np
from ray.rllib.env import MultiAgentEnv
import math
from agents import BLOOM_LEVEL_MAP, StudentAgent, TeacherAgent
from utils import analyze_text_for_bloom
from config import LOG_FILE_NAME
from utils import analyze_text_for_bloom, analyze_explanation_quality


def log_data(msg):
    with open(LOG_FILE_NAME, "a") as f:
        f.write(msg + "\n")
        print(msg)

def teacher_action_to_str(action):
    if action == TeacherAgent.ACTION_SIMPLE:
        return "Simple"
    elif action == TeacherAgent.ACTION_COMPLEX:
        return "Complex"
    else:
        return "Unknown - " + str(action)

def student_action_to_str(action):
    if action == StudentAgent.ACTION_ASK:
        return "Ask"
    elif action == StudentAgent.ACTION_STUDY:
        return "Study"
    else:
        return "Unknown - " + str(action)
    
def student_bloom_level_to_str(bloom_level):
    if bloom_level == StudentAgent.BLOOM_REMEMBER:
        return "Remember"
    elif bloom_level == StudentAgent.BLOOM_UNDERSTAND:
        return "Understand"
    elif bloom_level == StudentAgent.BLOOM_APPLY:
        return "Apply"
    elif bloom_level == StudentAgent.BLOOM_ANALYZE:
        return "Analyze"
    elif bloom_level == StudentAgent.BLOOM_EVALUATE:
        return "Evaluate"
    elif bloom_level == StudentAgent.BLOOM_CREATE:
        return "Create"
    else:
        return "Unknown - " + str(bloom_level)

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
        self._episode_count = 0     # used by my logger

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

        self.students = {}
        for i in range(self.num_students):
            student_id = f"student_{i}"
            student_type = student_types[i]
            self.students[student_id] = StudentAgent(student_id, student_type)

        self._agent_ids = {self.teacher.agent_id}
        self._agent_ids.update(self.students.keys())
        self.current_step = 0

        # Define observation spaces and action spaces for all agents
        self.observation_spaces = {}

        # TODO: right now the teacher's observation space is all the student's bloom levels
        # * We can try with this, and run another experiment where it observes other things as well or something different
        # Teacher observes all students' boom levels
        if self.num_students == 1:
            # Use Discrete if only 1 student (easier for simple networks)
            # Note: Bloom levels are 1-6, Discrete space is 0-5. So its shifted by -1 in _get_obs.
            self.observation_spaces[self.teacher.agent_id] = gym.spaces.Discrete(
                StudentAgent.NUM_BLOOM_LEVELS
            )
        else:
            # Use Box for multiple students
            # Note: Box low=1, high=6 to represent actual Bloom levels
            self.observation_spaces[self.teacher.agent_id] = gym.spaces.Box(
                low=1,
                high=StudentAgent.NUM_BLOOM_LEVELS,
                shape=(self.num_students,),
                dtype=np.int32,
            )

        # Each student observes their own knowledge level
        for student_id in self.students:
            self.observation_spaces[student_id] = gym.spaces.Discrete(
                StudentAgent.NUM_BLOOM_LEVELS
            )

        self.action_spaces = {}

        self.action_spaces[self.teacher.agent_id] = gym.spaces.Discrete(
            TeacherAgent.NUM_ACTIONS
        )

        for student_id in self.students:
            self.action_spaces[student_id] = gym.spaces.Discrete(
                StudentAgent.NUM_ACTIONS
            )

        print(f"  Observation Spaces: {self.observation_spaces}")
        print(f"  Action Spaces: {self.action_spaces}")

    def reset(self, *, seed=None, options=None):
        """Resets the environment and agents to initial states.

        Returns:
            tuple: (observation dicitionary, info dictionary)
        """
        # ! NEW EPISODE STARTS AFTER RESET IS CALLED ! 
        super().reset(seed=seed)
        self.current_step = 0
        self.teacher.reset()
        for student in self.students.values():
            student.reset()

        observations = self._get_obs()
        infos = self._get_infos()  # Get base infos

        infos[self.teacher.agent_id].pop("last_teacher_explanation", None)
        for student_id in self.students:
            infos[student_id].pop("last_student_question", None)
            infos[student_id].pop("question_bloom_level", None)
        return observations, infos

    def step(self, action_dict):
        """
        Advances the environment by one step, triggers LLM calls, analyzes
        question Bloom level, and calculates ZPD/Bloom-based rewards.

        TODO: area of improvement - have teacher generate personalized explanations
        TODO: area of improvement - change the termination or success criteria for the teacher?
        """
        self.current_step += 1
        
        if self.current_step == 1: # NEW EPISODE HAS STARTED
            log_data("\n\n" + "*"*10 + f"Episode {self._episode_count} has started" + "*"*10)
            self._episode_count += 1
        log_data(f"\n-----Step {self.current_step} has started-----")

        if self.teacher.agent_id not in action_dict:
            print(f"Warning: No teacher action provided")
            # ! not really sure what to do here. but i'm confused why this would happen
            return self._get_obs(), {}, {}, {}, self._get_infos()
        teacher_action = action_dict.get(self.teacher.agent_id, TeacherAgent.ACTION_SIMPLE)

        # print(f"\n--- Step {self.current_step} ---")
        # print(f"Teacher Action: {teacher_action}")
        log_data(f"Teacher Action: {teacher_action_to_str(teacher_action)}")

        # Store old Bloom levels for reward calculation
        old_bloom_levels = {
            student_id: student.current_bloom_level
            for student_id, student in self.students.items()
        }

        # Process student actions, update states, and trigger LLM calls
        level_advanced = {student_id: False for student_id in self.students}
        student_actions = {}
        questions = {}
        estimated_bloom_levels = {}  # Store analysis results

        for student_id, student in self.students.items():
            if student_id in action_dict:
                student_action = action_dict[student_id]
                student_actions[student_id] = student_action
                
                print(
                    f"{student_id} Action: {student_action} (Current Bloom: {student.current_bloom_level})"
                )
                log_data(f"{student_id} Action: {student_action_to_str(student_action)} (Current Bloom: {student_bloom_level_to_str(student.current_bloom_level)})")

                # Update student state (Bloom level)
                level_advanced[student_id] = student.update_state(
                    teacher_action, student_action
                )
                log_data(f"{student_id} Level Advanced: {'TRUE' if level_advanced[student_id] else 'FALSE'}")

                # If student asks a question, generate it and analyze its Bloom level
                if student_action == StudentAgent.ACTION_ASK:
                    question_text = student.generate_question(self.topic)
                    questions[student_id] = question_text
                    print(f"[LLM Output] {student_id} Question: {question_text}")
                    # log_data(f"{student_id} Question: {question_text}")
                    student_level_desc = BLOOM_LEVEL_MAP.get(
                        student.current_bloom_level, "Unknown"
                    )
                    log_data(f"{student_id} Student Level Desc: {student_level_desc}")
                    estimated_level = analyze_text_for_bloom(
                        question_text, student_level_desc, self.topic
                    )
                    estimated_bloom_levels[student_id] = estimated_level
                    log_data(f"{student_id} Estimated Bloom Level: {estimated_level}")
                    student.last_question_bloom_level = (
                        estimated_level  # Store in agent state
                    )
            else:
                # print(f"Warning: No action provided for {student_id}")
                log_data(f"Warning: No action provided for {student_id}")

        # Teacher generates explanation if applicable (after student actions)
        explanation = None
        if (
            teacher_action == TeacherAgent.ACTION_SIMPLE
            or teacher_action == TeacherAgent.ACTION_COMPLEX
        ):
            # Generate explanation based on average Bloom level for simplicity
            avg_bloom_level = round(
                sum(s.current_bloom_level for s in self.students.values())
                / self.num_students
            )
            # TODO: pass the student's zpd alignment to the teacher and have the teacher's response based on that as well
            explanation = self.teacher.generate_explanation(avg_bloom_level, self.topic)
            print(
                f"[LLM Output] Teacher Explanation (for avg level {avg_bloom_level}): {explanation}"
            )
            exp_quality = (
                 analyze_explanation_quality(explanation, self.topic) if explanation else 1
             )
            num_students_asking = sum(
                 1 for a in student_actions.values()
                 if a == StudentAgent.ACTION_ASK
            )

        # Capture new Bloom levels
        new_bloom_levels = {
            student_id: student.current_bloom_level
            for student_id, student in self.students.items()
        }
        print(f"Bloom levels update: {old_bloom_levels} -> {new_bloom_levels}")
        log_data(f"Bloom levels update: {old_bloom_levels} -> {new_bloom_levels}")

        # calculate rewards
        rewards = self._calculate_rewards(
            teacher_action,
            student_actions,
            old_bloom_levels,
            new_bloom_levels,
            level_advanced,
            estimated_bloom_levels,  # Pass estimated levels from analysis
            exp_quality,
            num_students_asking,
        )
        print(f"Rewards calculated: {rewards}")
        log_data(f"Rewards calculated: {rewards}")

        # Terminate if all students reach the highest Bloom level
        all_students_max_bloom = any(
            level == StudentAgent.NUM_BLOOM_LEVELS
            for level in new_bloom_levels.values()
        )
        terminate_episode = all_students_max_bloom
        truncate_episode = self.current_step >= self.max_steps
        # truncate_episode = False

        terminated = {agent_id: all_students_max_bloom for agent_id in self._agent_ids}
        truncated = {agent_id: truncate_episode for agent_id in self._agent_ids}
        terminated["__all__"] = terminate_episode
        truncated["__all__"] = truncate_episode

        # get observations and infos
        observations = self._get_obs()
        infos = self._get_infos()

        # add data to infos
        self._add_zpd_alignment_info(infos, old_bloom_levels, teacher_action)    
        self._add_question_explanation_analysis_info(infos, questions, estimated_bloom_levels, explanation)

        print(f"Infos Keys: {infos.keys()}")
        
        return observations, rewards, terminated, truncated, infos
    
    # TODO: make sure this is correct
    def _add_zpd_alignment_info(self, infos, old_bloom_levels, teacher_action):
        for student_id, student in self.students.items():
            old_level = old_bloom_levels[student_id]
            in_zpd_step = 0 # 0 = neutral, 1 = in ZPD, -1 = complex too early
            if (teacher_action == TeacherAgent.ACTION_SIMPLE and old_level <= StudentAgent.BLOOM_UNDERSTAND) or \
            (teacher_action == TeacherAgent.ACTION_COMPLEX and old_level >= StudentAgent.BLOOM_APPLY):
                in_zpd_step = 1
            elif (teacher_action == TeacherAgent.ACTION_COMPLEX and old_level <= StudentAgent.BLOOM_UNDERSTAND):
                in_zpd_step = -1
        
            # Add to the specific student's info dict
            if student_id not in infos: 
                infos[student_id] = {}      # Ensure dict exists
            infos[student_id]["zpd_alignment"] = in_zpd_step
            
    def _add_question_explanation_analysis_info(self, infos, questions, estimated_bloom_levels, explanation):
        if explanation:
            infos[self.teacher.agent_id]["last_teacher_explanation"] = explanation
        for student_id, question in questions.items():
            infos[student_id]["last_student_question"] = question
            # Also add the estimated bloom level from analysis
            if student_id in estimated_bloom_levels:
                infos[student_id]["question_bloom_level"] = estimated_bloom_levels[student_id]

    def _get_obs(self):
        """Gets the current observations for each agent based on Bloom levels."""
        observations = {}
        student_bloom_levels = [s.current_bloom_level for s in self.students.values()]

        # Teacher observation - observes all students' Bloom levels
        if self.num_students == 1:
            # Shift by -1 for Discrete space (0-5)
            observations[self.teacher.agent_id] = student_bloom_levels[0] - 1
        else:
            # Use actual Bloom levels (1-6) for Box space
            observations[self.teacher.agent_id] = np.array(
                student_bloom_levels, dtype=np.int32
            )

        # Student observations - each student observes their own Bloom level
        for i, (student_id, _) in enumerate(self.students.items()):
            observations[student_id] = student_bloom_levels[i] - 1

        return observations

    def _get_infos(self):
        """Gets auxiliary information, including Bloom levels."""
        infos = {}
        teacher_info = {
            f"{sid}_bloom_level": s.current_bloom_level
            for sid, s in self.students.items()
        }
        teacher_info["class_avg_bloom"] = (
            sum(s.current_bloom_level for s in self.students.values())
            / self.num_students
        )
        infos[self.teacher.agent_id] = teacher_info

        # Student info
        for student_id, student in self.students.items():
            infos[student_id] = {
                "self_bloom_level": student.current_bloom_level,
                "student_type": student.student_type,
            }
        return infos

    def _calculate_student_reward(
        self,
        student_id,
        student_action,
        old_level,
        new_level,
        level_advanced,
        teacher_action,
        rewards,
        estimated_question_levels,
    ):
        # TODO: all the numbers here being added/subtracted are arbitrary - need to find good values or tune them or wtv
        
        # --- ZPD Check ---
        # Define if teacher action was appropriate for the student's ZPD (old_level)
        in_zpd = False
        teacher_zpd_bonus_delta = 0.0
        teacher_zpd_penalty_delta = 0.0
        student_bloom_bonus_total_delta = 0.0
        if (teacher_action == TeacherAgent.ACTION_SIMPLE and old_level <= StudentAgent.BLOOM_UNDERSTAND) or \
            (teacher_action == TeacherAgent.ACTION_COMPLEX and old_level >= StudentAgent.BLOOM_APPLY):
            in_zpd = True
            teacher_zpd_bonus_delta += 0.1  # Small bonus per student helped within ZPD
        elif (teacher_action == TeacherAgent.ACTION_COMPLEX and old_level <= StudentAgent.BLOOM_UNDERSTAND):
            teacher_zpd_penalty_delta -= 0.15

        # --- Reward for Bloom Level Advancement ---
        if level_advanced[student_id]:
            advancement_reward = 1.5  # Base reward for advancing a level
            # TODO: definitely change this - likely remove (if statement and +0.5 reward)
            if in_zpd:
                advancement_reward += 0.5  # Bonus if teacher action was in ZPD
            rewards[student_id] += advancement_reward

        # --- Reward for Asking High-Level Questions ---
        if student_action == StudentAgent.ACTION_ASK:
            # Use the estimated Bloom level from LLM analysis
            # Default to 1 if no analysis
            question_level = estimated_question_levels.get(student_id, 1)
            
            # Reward proportional to the question level (scaled)
            bloom_reward = (question_level / StudentAgent.NUM_BLOOM_LEVELS) * 0.5  # Max 0.5 reward
            rewards[student_id] += bloom_reward
            student_bloom_bonus_total_delta += bloom_reward  # Accumulate for potential teacher reward

        # --- Goal Achievement Bonus (Reaching Max Bloom Level) ---
        if new_level == StudentAgent.NUM_BLOOM_LEVELS and old_level < StudentAgent.NUM_BLOOM_LEVELS:
            rewards[student_id] += 5.0  # Bonus for reaching highest level
            # TODO: 5 might be kinda crazy high - LOW PRIORITY to change
            
        return teacher_zpd_bonus_delta, teacher_zpd_penalty_delta, student_bloom_bonus_total_delta
    
    def _calculate_teacher_reward(self, rewards, old_bloom_levels, new_bloom_levels, level_advanced, 
                                  num_students_asking, exp_quality, student_bloom_bonus_total, teacher_zpd_bonus, teacher_zpd_penalty):
        # TODO: all the numbers here being added/subtracted are arbitrary - need to find good values or tune them or wtv
        # Base reward for average advancement (simple version)
        # avg_advancement = sum(level_advanced.values()) / self.num_students
        # rewards[self.teacher.agent_id] += avg_advancement * 1.0

        # # Apply accumulated ZPD bonuses/penalties
        # rewards[self.teacher.agent_id] += teacher_zpd_bonus
        # rewards[self.teacher.agent_id] += teacher_zpd_penalty

        # # Reward teacher for eliciting high-level questions (average Bloom bonus given to students)
        # if self.num_students > 0:
        #     rewards[self.teacher.agent_id] += (
        #         student_bloom_bonus_total / self.num_students
        #     )

        # # Goal achievement bonus for teacher (all students reach max level)
        # if all(level == StudentAgent.NUM_BLOOM_LEVELS for level in new_bloom_levels.values()) \
        #     and not all(level == StudentAgent.NUM_BLOOM_LEVELS for level in old_bloom_levels.values()):
        #     rewards[self.teacher.agent_id] += 5.0

        #  """
        # Multi‑term teacher reward:

        #     β1 · progress           (gap‑scaled)
        #   – β2 · std‑dev            (equity)
        #   + β3 · engagement         (# ASK actions / N)
        #   + β4 · explanation score  (Bloom‑normed)
        #   – β5 · time penalty       (t / T_max)
        #   + ZPD & question bonuses
        #   + terminal bonus
        # """
        N = self.num_students

        # 0.  β‑weights (fallback to defaults if user forgot the key)
        β1, β2, β3, β4, β5 = self.config.get(
            "teacher_reward_weights", [1.0, 0.5, 0.3, 0.5, 0.05]
        )
        progress = sum(
            (new_bloom_levels[s] - old_bloom_levels[s])
            / max(1, StudentAgent.NUM_BLOOM_LEVELS - old_bloom_levels[s])
            for s in self.students
        ) / N

        # 2. equity term (negative std‑dev of current Bloom levels)
        mean_lvl = sum(new_bloom_levels.values()) / N
        equity_pen = math.sqrt(
            sum((new_bloom_levels[s] - mean_lvl) ** 2 for s in self.students) / N
        )

        # 3. engagement term
        engagement = num_students_asking / N

        # 4. explanation‑quality term (normalised 0‑1)
        exp_norm = exp_quality / StudentAgent.NUM_BLOOM_LEVELS

        # 5. time penalty (encourage finishing early)
        time_pen = self.current_step / self.max_steps

        # ----- aggregate everything -----
        teacher_reward = (
            β1 * progress
            - β2 * equity_pen
            + β3 * engagement
            + β4 * exp_norm
            - β5 * time_pen
            + teacher_zpd_bonus
            + teacher_zpd_penalty
            + (student_bloom_bonus_total / N if N else 0.0)  # carry‑over bonus
        )

        # terminal bonus if all students reach Bloom level 6
        if all(
            lvl == StudentAgent.NUM_BLOOM_LEVELS for lvl in new_bloom_levels.values()
        ) and not all(
            lvl == StudentAgent.NUM_BLOOM_LEVELS for lvl in old_bloom_levels.values()
        ):
            teacher_reward += 5.0

        rewards[self.teacher.agent_id] += teacher_reward

    def _calculate_rewards(
        self,
        teacher_action,
        student_actions,
        old_bloom_levels,
        new_bloom_levels,
        level_advanced,
        estimated_question_levels,
        exp_quality,
        num_students_asking,
    ):
        """
        Calculates rewards incorporating Bloom's Taxonomy and Zone of Proximal Development (ZPD).
        """
        rewards = {agent_id: 0.0 for agent_id in self._agent_ids}
        teacher_zpd_bonus = 0.0
        teacher_zpd_penalty = 0.0
        student_bloom_bonus_total = 0.0

        # calculate student rewards
        for student_id, _ in self.students.items():
            old_level = old_bloom_levels[student_id]
            new_level = new_bloom_levels[student_id]
            student_action = student_actions.get(student_id, None)

            teacher_zpd_bonus_delta, teacher_zpd_penalty_delta, student_bloom_bonus_total_delta = self._calculate_student_reward(
                student_id,
                student_action,
                old_level,
                new_level,
                level_advanced,
                teacher_action,
                rewards,
                estimated_question_levels,
            )
            
            teacher_zpd_bonus += teacher_zpd_bonus_delta
            teacher_zpd_penalty += teacher_zpd_penalty_delta
            student_bloom_bonus_total += student_bloom_bonus_total_delta

        # calculate teacher rewards
        self._calculate_teacher_reward(rewards, old_bloom_levels, new_bloom_levels, level_advanced, 
                                       num_students_asking, exp_quality, student_bloom_bonus_total, teacher_zpd_bonus, teacher_zpd_penalty)

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
