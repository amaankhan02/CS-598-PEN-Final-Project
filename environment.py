from collections import OrderedDict

import gymnasium as gym
import numpy as np
from ray.rllib.env import MultiAgentEnv

from agents import BLOOM_LEVEL_MAP, StudentAgent, TeacherAgent
from utils import analyze_text_for_bloom


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

        self.students = {}
        for i in range(self.num_students):
            student_id = f"student_{i}"
            student_type = student_types[i]
            self.students[student_id] = StudentAgent(student_id, student_type)

        self._agent_ids = {self.teacher.agent_id}
        self._agent_ids.update(self.students.keys())
        self.current_step = 0
        
        # Initialize infos dictionary for tracking metrics
        self.infos = {agent_id: {} for agent_id in self._agent_ids}

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
        super().reset(seed=seed)
        self.current_step = 0
        self.teacher.reset()
        for student in self.students.values():
            student.reset()
            
        # Reset tracking infos
        self.infos = {agent_id: {} for agent_id in self._agent_ids}

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

        teacher_action = action_dict[self.teacher.agent_id]

        print(f"\n--- Step {self.current_step} ---")
        print(f"Teacher Action: {teacher_action}")

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

                # Update student state (Bloom level)
                level_advanced[student_id] = student.update_state(
                    teacher_action, student_action
                )

                # If student asks a question, generate it and analyze its Bloom level
                if student_action == StudentAgent.ACTION_ASK:
                    question_text = student.generate_question(self.topic)
                    questions[student_id] = question_text
                    print(f"[LLM Output] {student_id} Question: {question_text}")

                    student_level_desc = BLOOM_LEVEL_MAP.get(
                        student.current_bloom_level, "Unknown"
                    )
                    estimated_level = analyze_text_for_bloom(
                        question_text, student_level_desc, self.topic
                    )
                    estimated_bloom_levels[student_id] = estimated_level
                    student.last_question_bloom_level = (
                        estimated_level  # Store in agent state
                    )
            else:
                print(f"Warning: No action provided for {student_id}")

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

        # Capture new Bloom levels
        new_bloom_levels = {
            student_id: student.current_bloom_level
            for student_id, student in self.students.items()
        }
        print(f"Bloom levels update: {old_bloom_levels} -> {new_bloom_levels}")

        # calculate rewards
        rewards = self._calculate_rewards(
            teacher_action,
            student_actions,
            old_bloom_levels,
            new_bloom_levels,
            level_advanced,
            estimated_bloom_levels,  # Pass estimated levels from analysis
        )
        print(f"Rewards calculated: {rewards}")

        # Terminate if all students reach the highest Bloom level
        # all_students_max_bloom = all(
        #     level == StudentAgent.NUM_BLOOM_LEVELS
        #     for level in new_bloom_levels.values()
        # )
        # TODO: chnaged to it ending when any student reaches max bloom level
        all_students_max_bloom = all(
            level == StudentAgent.NUM_BLOOM_LEVELS
            for level in new_bloom_levels.values()
        )
        
        terminate_episode = all_students_max_bloom
        truncate_episode = self.current_step >= self.max_steps

        terminated = {agent_id: terminate_episode for agent_id in self._agent_ids}
        truncated = {agent_id: truncate_episode for agent_id in self._agent_ids}
        terminated["__all__"] = terminate_episode
        truncated["__all__"] = truncate_episode

        # get observations and infos
        observations = self._get_obs()
        infos = self._get_infos()
        for agent_id in self._agent_ids:
            infos[agent_id]["action"] = action_dict[agent_id]

        # Calculate metrics for this step
        metrics = self._calculate_step_metrics(
            teacher_action, student_actions, old_bloom_levels, new_bloom_levels, level_advanced
        )
        
        # Add metrics to infos
        for agent_id in self._agent_ids:
            infos[agent_id]["metrics"] = metrics
        
        # add LLM outputs and analysis to infos
        self._add_question_explanation_analysis_info(infos, questions, estimated_bloom_levels, explanation)
        
        # Add ZPD alignment info
        self._add_zpd_alignment_info(infos, old_bloom_levels, teacher_action)

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

        # --- Reward for Bloom Level Advancement Based on Magnitude of Change ---
        if level_advanced[student_id]:
            level_delta = new_level - old_level
            if level_delta > 0:
                w_adv = 1.0
                advancement_reward = w_adv * level_delta
                rewards[student_id] += advancement_reward

        # --- Reward for Asking High-Level Questions (Based on Quality of Question compared to Student's Current Level) ---
        if student_action == StudentAgent.ACTION_ASK:
            # Use the estimated Bloom level from LLM analysis
            # Default to student's current level if no analysis
            question_level = estimated_question_levels.get(student_id, old_level)
            improvement = max(0, question_level - old_level)
            denom = max(1, StudentAgent.NUM_BLOOM_LEVELS - old_level)
            question_quality = improvement / denom
            w_q = 0.5

            # Reward proportional to the question level (scaled)
            q_reward = question_quality * w_q
            rewards[student_id] += q_reward
            student_bloom_bonus_total_delta += q_reward  # Accumulate for potential teacher reward

        # --- Goal Achievement Bonus (Reaching Max Bloom Level) ---
        if new_level == StudentAgent.NUM_BLOOM_LEVELS and old_level < StudentAgent.NUM_BLOOM_LEVELS:
            rewards[student_id] += 5.0  # Bonus for reaching highest level
            # TODO: 5 might be kinda crazy high - LOW PRIORITY to change
            
        return teacher_zpd_bonus_delta, teacher_zpd_penalty_delta, student_bloom_bonus_total_delta
    
    def _calculate_teacher_reward(self, rewards, old_bloom_levels, new_bloom_levels, level_advanced, 
                                  student_bloom_bonus_total, teacher_zpd_bonus, teacher_zpd_penalty):
        # TODO: all the numbers here being added/subtracted are arbitrary - need to find good values or tune them or wtv
        # Base reward for average advancement (simple version)
        avg_advancement = sum(level_advanced.values()) / self.num_students  # avg number of students that improved
        # TODO: level_advanced consists of boolean values (improve or not) - we can change this to be how much they improved - how many bloom levels they jumped
        rewards[self.teacher.agent_id] += avg_advancement * 1.0

        # Apply accumulated ZPD bonuses/penalties
        rewards[self.teacher.agent_id] += teacher_zpd_bonus
        rewards[self.teacher.agent_id] += teacher_zpd_penalty

        # Reward teacher for eliciting high-level questions (average Bloom bonus given to students)
        if self.num_students > 0:
            rewards[self.teacher.agent_id] += (
                student_bloom_bonus_total / self.num_students
            )

        # Goal achievement bonus for teacher (all students reach max level)
        if all(level == StudentAgent.NUM_BLOOM_LEVELS for level in new_bloom_levels.values()) \
            and not all(level == StudentAgent.NUM_BLOOM_LEVELS for level in old_bloom_levels.values()):
            rewards[self.teacher.agent_id] += 5.0

    def _calculate_rewards(
        self,
        teacher_action,
        student_actions,
        old_bloom_levels,
        new_bloom_levels,
        level_advanced,
        estimated_question_levels,
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
                                       student_bloom_bonus_total, teacher_zpd_bonus, teacher_zpd_penalty)

        # Track reward component breakdown for metrics
        teacher_id = self.teacher.agent_id
        total_teacher_reward = rewards[teacher_id]
        
        # Avoid division by zero
        if total_teacher_reward != 0:
            # Calculate contribution percentages
            advancement_contribution = (sum(level_advanced.values()) / self.num_students) / total_teacher_reward
            zpd_contribution = (teacher_zpd_bonus + teacher_zpd_penalty) / total_teacher_reward
            question_contribution = (student_bloom_bonus_total / self.num_students if self.num_students > 0 else 0) / total_teacher_reward
            
            # Store in infos for callback to access
            if "reward_breakdown" not in self.infos[teacher_id]:
                self.infos[teacher_id]["reward_breakdown"] = []
                
            self.infos[teacher_id]["reward_breakdown"].append({
                "zpd_contribution": zpd_contribution,
                "advancement_contribution": advancement_contribution,
                "question_contribution": question_contribution
            })

        return rewards

    def _calculate_step_metrics(self, teacher_action, student_actions, old_bloom_levels, new_bloom_levels, level_advanced):
        """Calculate per-step metrics for tracking"""
        from agents import TeacherAgent, StudentAgent
        
        metrics = {}
        
        # 1. Bloom level variance
        bloom_levels = list(new_bloom_levels.values())
        if len(bloom_levels) >= 2:
            metrics["bloom_level_variance"] = np.var(bloom_levels)
            metrics["knowledge_gap"] = max(bloom_levels) - min(bloom_levels)
        
        # 2. ZPD matching rate for this step
        zpd_matches = 0
        for student_id, old_level in old_bloom_levels.items():
            if ((teacher_action == TeacherAgent.ACTION_SIMPLE and old_level <= StudentAgent.BLOOM_UNDERSTAND) or
                (teacher_action == TeacherAgent.ACTION_COMPLEX and old_level >= StudentAgent.BLOOM_APPLY)):
                zpd_matches += 1
        
        metrics["step_zpd_match_rate"] = zpd_matches / self.num_students if self.num_students > 0 else 0
        
        # 3. Action effectiveness for this step
        advancements = sum(level_advanced.values())
        metrics["step_advancement_rate"] = advancements / self.num_students if self.num_students > 0 else 0
        
        # 4. Average bloom level
        metrics["avg_bloom_level"] = sum(new_bloom_levels.values()) / self.num_students if self.num_students > 0 else 0
        
        return metrics


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
