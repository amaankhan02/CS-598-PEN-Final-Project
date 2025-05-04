# train.py
# Trains the Teacher and Student agents in the ClassroomEnv using RLlib/PPO.

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

from agents import StudentAgent, TeacherAgent
from config import DEFAULT_ENV_CONFIG, DEFAULT_TRAINING_CONFIG
from environment import ClassroomEnv

class ClassroomCallbacks(DefaultCallbacks):
    """Callbacks to track classroom metrics across episodes"""
    
    def __init__(self):
        super().__init__()
        self.episode_count = 0
        print("ClassroomCallbacks initialized")
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Process metrics at the end of each episode"""
        print("--------------------------------")
        print(f"on_episode_end called for episode {self.episode_count + 1}")
        
        # Extract student IDs
        print(f"Episode rewards: {episode.agent_rewards}")
        
        # student_ids = [agent_id for agent_id in episode.agent_rewards.keys() 
        #               if isinstance(agent_id, str) and agent_id.startswith("student")]
        student_ids = [agent_id[0] for agent_id in episode.agent_rewards.keys() 
               if isinstance(agent_id, tuple) and agent_id[0].startswith("student")]
        teacher_id = "teacher_0"  # Assuming teacher ID is consistent
        
        print(f"Found {len(student_ids)} students and teacher_id: {teacher_id}")
        
        if not student_ids:
            print("No students found in episode, skipping metrics")
            return  # Skip metrics if no students
            
        # Calculate all metrics
        self._track_bloom_metrics(episode, student_ids, teacher_id)
        self._track_zpd_metrics(episode, student_ids, teacher_id)
        self._track_teacher_strategy(episode, student_ids, teacher_id)
        self._track_student_behavior(episode, student_ids)
        self._track_equity_metrics(episode, student_ids)
        self._track_reward_components(episode, student_ids, teacher_id)
        
        # Print custom metrics count to verify they were collected
        print(f"Collected {len(episode.custom_metrics)} custom metrics for episode {self.episode_count + 1}")
        
        # Increment episode counter
        self.episode_count += 1
    
    def _track_bloom_metrics(self, episode, student_ids, teacher_id):
        """Track metrics related to Bloom's Taxonomy levels"""
        print(f"_track_bloom_metrics called with {len(student_ids)} students")
        
        # 1. Time to Max Bloom - track step when each student first reaches max level
        if not hasattr(self, "time_to_max_bloom"):
            self.time_to_max_bloom = {}
            self.reached_max_bloom = set()
            
        from agents import StudentAgent
        max_bloom = StudentAgent.NUM_BLOOM_LEVELS
        
        # Collect final bloom levels
        final_bloom_levels = []
        for student_id in student_ids:
            last_info = episode.last_info_for(student_id)
            if last_info and "self_bloom_level" in last_info:
                bloom_level = last_info["self_bloom_level"]
                final_bloom_levels.append(bloom_level)
                print(f"Student {student_id} final bloom level: {bloom_level}")
                
                # Check if student reached max bloom for the first time this episode
                if bloom_level == max_bloom and student_id not in self.reached_max_bloom:
                    self.time_to_max_bloom[student_id] = episode.length
                    self.reached_max_bloom.add(student_id)
        
        if not final_bloom_levels:
            print("WARNING: No bloom levels found for any students!")
            
        # Calculate average time to max bloom
        if self.time_to_max_bloom:
            episode.custom_metrics["avg_time_to_max_bloom"] = np.mean(list(self.time_to_max_bloom.values()))
            episode.custom_metrics["students_reached_max_bloom"] = len(self.reached_max_bloom)
        
        # 2. Bloom Level Distribution - percentage of students at each level
        if final_bloom_levels:
            # Calculate average bloom level
            avg_bloom = np.mean(final_bloom_levels)
            episode.custom_metrics["final_avg_bloom_level"] = avg_bloom
            
            # Calculate distribution
            for level in range(1, max_bloom + 1):
                count = sum(1 for b in final_bloom_levels if b == level)
                episode.custom_metrics[f"bloom_level_{level}_pct"] = 100 * count / len(final_bloom_levels)
        
        # 3. Bloom Level Advancement Rate
        total_advancements = 0
        total_possible = 0
        
        # Track bloom levels across the episode to calculate advancements
        for student_id in student_ids:
            if student_id in episode.hist_data and "self_bloom_level" in episode.hist_data[student_id]:
                bloom_history = episode.hist_data[student_id]["self_bloom_level"]
                if len(bloom_history) > 1:
                    # Count advancements (increases in bloom level)
                    advancements = sum(1 for i in range(1, len(bloom_history)) 
                                     if bloom_history[i] > bloom_history[i-1])
                    total_advancements += advancements
                    total_possible += len(bloom_history) - 1  # Maximum possible advancements
        
        if total_possible > 0:
            episode.custom_metrics["bloom_advancement_rate"] = total_advancements / total_possible
            
    def _track_zpd_metrics(self, episode, student_ids, teacher_id):
        """Track metrics related to Zone of Proximal Development"""
        
        # ZPD Match Rate - percentage of teacher actions that match students' ZPD
        from agents import TeacherAgent, StudentAgent
        
        zpd_matches = 0
        total_actions = 0
        
        # Get teacher actions
        # teacher_actions = episode.agent_actions.get(teacher_id, [])
        teacher_actions = self.__get_actions(episode, teacher_id)
        
        # Process each step to determine ZPD matches
        for t in range(len(teacher_actions)):
            teacher_action = teacher_actions[t]
            zpd_matched_students = 0
            
            # Check each student's bloom level for ZPD match
            for student_id in student_ids:
                if (student_id in episode.hist_data and 
                    "self_bloom_level" in episode.hist_data[student_id] and
                    t < len(episode.hist_data[student_id]["self_bloom_level"])):
                    
                    bloom_level = episode.hist_data[student_id]["self_bloom_level"][t]
                    
                    # Apply ZPD matching rules
                    if ((teacher_action == TeacherAgent.ACTION_SIMPLE and 
                         bloom_level <= StudentAgent.BLOOM_UNDERSTAND) or
                        (teacher_action == TeacherAgent.ACTION_COMPLEX and 
                         bloom_level >= StudentAgent.BLOOM_APPLY)):
                        zpd_matched_students += 1
            
            # Calculate ZPD match rate for this step
            if student_ids:
                step_match_rate = zpd_matched_students / len(student_ids)
                zpd_matches += step_match_rate
                total_actions += 1
        
        if total_actions > 0:
            episode.custom_metrics["zpd_match_rate"] = zpd_matches / total_actions
        
    def _track_teacher_strategy(self, episode, student_ids, teacher_id):
        """Track metrics related to teacher strategy"""
        
        from agents import TeacherAgent, StudentAgent
        
        # Get teacher actions
        # teacher_actions = episode.agent_actions.get(teacher_id, [])
        # if not teacher_actions:
        #     return
        teacher_actions = self.__get_actions(episode, teacher_id)
        if not teacher_actions:
            return
            
        # 1. Teacher Action Distribution
        simple_actions = sum(1 for a in teacher_actions if a == TeacherAgent.ACTION_SIMPLE)
        complex_actions = sum(1 for a in teacher_actions if a == TeacherAgent.ACTION_COMPLEX)
        total_actions = len(teacher_actions)
        
        episode.custom_metrics["teacher_simple_action_rate"] = simple_actions / total_actions
        episode.custom_metrics["teacher_complex_action_rate"] = complex_actions / total_actions
        
        # 2. Action-to-Level Correlation
        low_level_actions = {"simple": 0, "complex": 0}
        high_level_actions = {"simple": 0, "complex": 0}
        
        for t in range(len(teacher_actions)):
            # Get average bloom level at this step
            if teacher_id in episode.hist_data and "class_avg_bloom" in episode.hist_data[teacher_id]:
                if t < len(episode.hist_data[teacher_id]["class_avg_bloom"]):
                    avg_bloom = episode.hist_data[teacher_id]["class_avg_bloom"][t]
                    action_type = "simple" if teacher_actions[t] == TeacherAgent.ACTION_SIMPLE else "complex"
                    
                    if avg_bloom <= 2.5:  # Low level threshold
                        low_level_actions[action_type] += 1
                    else:  # High level
                        high_level_actions[action_type] += 1
        
        # Calculate strategy adaptation metrics
        total_low = sum(low_level_actions.values())
        total_high = sum(high_level_actions.values())
        
        if total_low > 0:
            episode.custom_metrics["low_level_simple_rate"] = low_level_actions["simple"] / total_low
            episode.custom_metrics["low_level_complex_rate"] = low_level_actions["complex"] / total_low
            
        if total_high > 0:
            episode.custom_metrics["high_level_simple_rate"] = high_level_actions["simple"] / total_high
            episode.custom_metrics["high_level_complex_rate"] = high_level_actions["complex"] / total_high
        
        # 3. Action Effectiveness
        simple_advancements = 0
        complex_advancements = 0
        
        # Track bloom level changes after each action
        for student_id in student_ids:
            if (student_id in episode.hist_data and 
                "self_bloom_level" in episode.hist_data[student_id]):
                
                bloom_history = episode.hist_data[student_id]["self_bloom_level"]
                
                for t in range(len(teacher_actions) - 1):  # -1 because we need t+1
                    if t+1 < len(bloom_history):
                        # Check if bloom level increased after this action
                        if bloom_history[t+1] > bloom_history[t]:
                            if teacher_actions[t] == TeacherAgent.ACTION_SIMPLE:
                                simple_advancements += 1
                            else:
                                complex_advancements += 1
        
        # Calculate effectiveness rates
        if simple_actions > 0:
            episode.custom_metrics["simple_action_effectiveness"] = simple_advancements / simple_actions
            
        if complex_actions > 0:
            episode.custom_metrics["complex_action_effectiveness"] = complex_advancements / complex_actions
    
    def __get_actions(self, episode, agent_id):
        # actions = []
        # for i, (agent_id, action) in enumerate(episode.actions.items()):
        #     if agent_id == key_agent_id:
        #         actions.append(action)
        # return actions
        # for agent_id in agent_ids:
        if agent_id in episode.hist_data and "actions" in episode.hist_data[agent_id]:
            return episode.hist_data[agent_id]["actions"]
        return []

    
    def _track_student_behavior(self, episode, student_ids):
        """Track metrics related to student behavior"""
        
        from agents import StudentAgent
        
        # 1. Question Quality Trends
        all_question_levels = []
        
        for student_id in student_ids:
            if (student_id in episode.hist_data and 
                "question_bloom_level" in episode.hist_data[student_id]):
                
                question_levels = episode.hist_data[student_id]["question_bloom_level"]
                all_question_levels.extend(question_levels)
                
                # Track individual student question quality
                if question_levels:
                    avg_question_level = np.mean(question_levels)
                    episode.custom_metrics[f"{student_id}_avg_question_level"] = avg_question_level
                    
                    # Track question quality progression
                    if len(question_levels) >= 2:
                        # Calculate the change between consecutive questions
                        question_changes = [question_levels[i] - question_levels[i-1] 
                                          for i in range(1, len(question_levels))]
                        
                        # Average change in question quality
                        avg_question_change = np.mean(question_changes)
                        episode.custom_metrics[f"{student_id}_avg_question_change"] = avg_question_change
                        
                        # Last-vs-first question level difference (overall trajectory)
                        question_trajectory = question_levels[-1] - question_levels[0]
                        episode.custom_metrics[f"{student_id}_question_trajectory"] = question_trajectory
        
        # Track class-wide question quality
        if all_question_levels:
            episode.custom_metrics["class_avg_question_level"] = np.mean(all_question_levels)
            
            # Aggregate question levels by step across all students
            # First, determine the maximum step with questions
            max_step = 0
            for student_id in student_ids:
                if (student_id in episode.hist_data and 
                    "question_bloom_level" in episode.hist_data[student_id]):
                    max_step = max(max_step, len(episode.hist_data[student_id]["question_bloom_level"]))
            
            # Group questions by step
            questions_by_step = [[] for _ in range(max_step)]
            for student_id in student_ids:
                if (student_id in episode.hist_data and 
                    "question_bloom_level" in episode.hist_data[student_id]):
                    
                    for i, level in enumerate(episode.hist_data[student_id]["question_bloom_level"]):
                        if i < max_step:
                            questions_by_step[i].append(level)
            
            # Calculate class average question level at each step
            step_averages = [np.mean(levels) if levels else 0 for levels in questions_by_step]
            
            # Track class-wide question quality changes
            if len(step_averages) >= 2:
                # Calculate changes between consecutive steps
                step_changes = [step_averages[i] - step_averages[i-1] 
                              for i in range(1, len(step_averages))]
                
                if step_changes:
                    episode.custom_metrics["class_avg_question_change"] = np.mean(step_changes)
                
                # Overall trajectory (last vs first)
                non_zero_averages = [avg for avg in step_averages if avg > 0]
                if len(non_zero_averages) >= 2:
                    episode.custom_metrics["class_question_trajectory"] = non_zero_averages[-1] - non_zero_averages[0]
        
        # 2. Student Action Distribution
        for student_id in student_ids:
            # student_actions = episode.agent_actions.get(student_id, [])
            student_actions = self.__get_actions(episode, student_id)
            
            if student_actions:
                study_count = sum(1 for a in student_actions if a == StudentAgent.ACTION_STUDY)
                ask_count = sum(1 for a in student_actions if a == StudentAgent.ACTION_ASK)
                total = len(student_actions)
                
                episode.custom_metrics[f"{student_id}_study_rate"] = study_count / total
                episode.custom_metrics[f"{student_id}_ask_rate"] = ask_count / total
        
        # 3. Question-to-Advancement Correlation
        # This requires tracking questions and subsequent advancements
        advancement_after_question = 0
        advancement_after_study = 0
        total_questions = 0
        total_study = 0
        
        # Track advancements by bloom level
        advancements_by_level = {}
        actions_by_level = {}
        
        # Track advancements by bloom level and student type
        advancements_by_level_type = {}
        actions_by_level_type = {}
        
        for student_id in student_ids:
            student_actions = self.__get_actions(episode, student_id)
            # if (student_id in episode.agent_actions and 
            #     student_id in episode.hist_data and 
            #     "self_bloom_level" in episode.hist_data[student_id]):
            if student_actions and student_id in episode.hist_data and "self_bloom_level" in episode.hist_data[student_id]:
                actions = student_actions
                bloom_history = episode.hist_data[student_id]["self_bloom_level"]
                
                # Get student type if available
                student_type = None
                last_info = episode.last_info_for(student_id)
                if last_info and "student_type" in last_info:
                    student_type = last_info["student_type"]
                
                for t in range(len(actions) - 1):  # -1 because we need t+1
                    if t+1 < len(bloom_history):
                        current_level = bloom_history[t]
                        action_type = "ask" if actions[t] == StudentAgent.ACTION_ASK else "study"
                        
                        # Initialize counters for this level if not exists
                        if current_level not in advancements_by_level:
                            advancements_by_level[current_level] = {"ask": 0, "study": 0}
                            actions_by_level[current_level] = {"ask": 0, "study": 0}
                        
                        # Increment action counter for this level
                        actions_by_level[current_level][action_type] += 1
                        
                        # If student type is available, track by level and type
                        if student_type:
                            level_type_key = (current_level, student_type)
                            if level_type_key not in advancements_by_level_type:
                                advancements_by_level_type[level_type_key] = {"ask": 0, "study": 0}
                                actions_by_level_type[level_type_key] = {"ask": 0, "study": 0}
                            
                            # Increment action counter for this level and type
                            actions_by_level_type[level_type_key][action_type] += 1
                        
                        # Check if advancement occurred
                        if bloom_history[t+1] > bloom_history[t]:
                            # Increment general advancement counters
                            if actions[t] == StudentAgent.ACTION_ASK:
                                advancement_after_question += 1
                                total_questions += 1
                            else:
                                advancement_after_study += 1
                                total_study += 1
                            
                            # Increment level-specific advancement counter
                            advancements_by_level[current_level][action_type] += 1
                            
                            # Increment level-and-type-specific counter if available
                            if student_type:
                                advancements_by_level_type[level_type_key][action_type] += 1
                        else:
                            # Count actions that didn't lead to advancement
                            if actions[t] == StudentAgent.ACTION_ASK:
                                total_questions += 1
                            else:
                                total_study += 1
        
        # Calculate advancement rates by action type
        if total_questions > 0:
            episode.custom_metrics["advancement_rate_after_question"] = advancement_after_question / total_questions
            
        if total_study > 0:
            episode.custom_metrics["advancement_rate_after_study"] = advancement_after_study / total_study
        
        # Calculate and record level-specific advancement rates
        for level, advancements in advancements_by_level.items():
            for action_type in ["ask", "study"]:
                total_actions = actions_by_level[level][action_type]
                if total_actions > 0:
                    advancement_rate = advancements[action_type] / total_actions
                    episode.custom_metrics[f"bloom_{level}_advancement_rate_after_{action_type}"] = advancement_rate
        
        # Calculate and record level-and-type-specific advancement rates
        for (level, student_type), advancements in advancements_by_level_type.items():
            for action_type in ["ask", "study"]:
                total_actions = actions_by_level_type[(level, student_type)][action_type]
                if total_actions > 0:
                    advancement_rate = advancements[action_type] / total_actions
                    episode.custom_metrics[f"{student_type}_bloom_{level}_advancement_rate_after_{action_type}"] = advancement_rate
    
    def _track_equity_metrics(self, episode, student_ids):
        """Track metrics related to educational equity"""
        
        # 1. Bloom Level Variance - how dispersed are student knowledge levels
        if len(student_ids) <= 1:
            return  # Skip equity metrics for single student
            
        # Track variance over time
        variance_history = []
        
        # Get the first student's bloom history to determine episode length
        first_student = student_ids[0]
        if (first_student in episode.hist_data and 
            "self_bloom_level" in episode.hist_data[first_student]):
            
            num_steps = len(episode.hist_data[first_student]["self_bloom_level"])
            
            # Calculate variance at each step
            for t in range(num_steps):
                step_bloom_levels = []
                
                for student_id in student_ids:
                    if (student_id in episode.hist_data and 
                        "self_bloom_level" in episode.hist_data[student_id] and
                        t < len(episode.hist_data[student_id]["self_bloom_level"])):
                        
                        step_bloom_levels.append(episode.hist_data[student_id]["self_bloom_level"][t])
                
                if len(step_bloom_levels) >= 2:  # Need at least 2 students for variance
                    variance_history.append(np.var(step_bloom_levels))
            
            if variance_history:
                episode.custom_metrics["avg_bloom_variance"] = np.mean(variance_history)
                episode.custom_metrics["final_bloom_variance"] = variance_history[-1]
        
        # 2. Knowledge Gap - difference between highest and lowest bloom levels
        gap_history = []
        
        # Calculate knowledge gap at each step
        for t in range(num_steps):
            step_bloom_levels = []
            
            for student_id in student_ids:
                if (student_id in episode.hist_data and 
                    "self_bloom_level" in episode.hist_data[student_id] and
                    t < len(episode.hist_data[student_id]["self_bloom_level"])):
                    
                    step_bloom_levels.append(episode.hist_data[student_id]["self_bloom_level"][t])
            
            if step_bloom_levels:
                gap_history.append(max(step_bloom_levels) - min(step_bloom_levels))
        
        if gap_history:
            episode.custom_metrics["avg_knowledge_gap"] = np.mean(gap_history)
            episode.custom_metrics["final_knowledge_gap"] = gap_history[-1]
        
        # 3. Learning Coefficient Impact
        # Track performance by student type
        student_types = {}
        
        for student_id in student_ids:
            last_info = episode.last_info_for(student_id)
            if last_info and "student_type" in last_info and "self_bloom_level" in last_info:
                student_type = last_info["student_type"]
                bloom_level = last_info["self_bloom_level"]
                
                if student_type not in student_types:
                    student_types[student_type] = []
                    
                student_types[student_type].append(bloom_level)
        
        # Calculate average bloom level by student type
        for student_type, bloom_levels in student_types.items():
            if bloom_levels:
                episode.custom_metrics[f"{student_type}_avg_bloom"] = np.mean(bloom_levels)
    
    def _track_reward_components(self, episode, student_ids, teacher_id):
        """Track metrics related to reward components"""
        
        # Analyze teacher reward components if provided in episode info
        if teacher_id in episode.hist_data and "reward_breakdown" in episode.hist_data[teacher_id]:
            reward_breakdowns = episode.hist_data[teacher_id]["reward_breakdown"]
            
            # Average the components across the episode
            zpd_contributions = []
            advancement_contributions = []
            question_contributions = []
            
            for breakdown in reward_breakdowns:
                if "zpd_contribution" in breakdown:
                    zpd_contributions.append(breakdown["zpd_contribution"])
                if "advancement_contribution" in breakdown:
                    advancement_contributions.append(breakdown["advancement_contribution"])
                if "question_contribution" in breakdown:
                    question_contributions.append(breakdown["question_contribution"])
            
            # Record average contribution of each component
            if zpd_contributions:
                episode.custom_metrics["zpd_reward_contribution"] = np.mean(zpd_contributions)
            if advancement_contributions:
                episode.custom_metrics["advancement_reward_contribution"] = np.mean(advancement_contributions)
            if question_contributions:
                episode.custom_metrics["question_reward_contribution"] = np.mean(question_contributions)
        
                
def get_tmp_spaces(env_config):
    temp_env = ClassroomEnv(env_config)

    teacher_obs_space = temp_env.observation_spaces["teacher_0"]
    teacher_action_space = temp_env.action_spaces["teacher_0"]

    # All students share the same observation and action space, so just grab the first one
    student_id = list(temp_env.students.keys())[0] if temp_env.students else "student_0"
    student_obs_space = temp_env.observation_spaces[student_id]
    student_action_space = temp_env.action_spaces[student_id]

    return (
        teacher_obs_space,
        teacher_action_space,
        student_obs_space,
        student_action_space,
    )


def print_metrics(result):
    """Print key metrics from training results"""
    # Basic reward metrics
    reward_mean = result.get("env_runners", {}).get("episode_return_mean", "N/A")
    policy_reward_mean = result.get("env_runners", {}).get("policy_return_mean", {})
    
    print("\n=== BASIC METRICS ===")
    print(f"Mean Episode Reward: {reward_mean}")
    if policy_reward_mean:
        print(f"Teacher Policy Mean Reward: {policy_reward_mean.get('teacher_policy', 'N/A')}")
        print(f"Student Policy Mean Reward: {policy_reward_mean.get('student_policy', 'N/A')}")
    
    # Get custom metrics if available
    custom_metrics = result.get("custom_metrics", {})
    if not custom_metrics:
        return
    
    # Learning effectiveness metrics
    print("\n=== LEARNING EFFECTIVENESS ===")
    if "final_avg_bloom_level" in custom_metrics:
        print(f"Average Bloom Level: {custom_metrics['final_avg_bloom_level']:.2f}")
    if "bloom_advancement_rate" in custom_metrics:
        print(f"Bloom Advancement Rate: {custom_metrics['bloom_advancement_rate']:.2f}")
    if "avg_time_to_max_bloom" in custom_metrics:
        print(f"Avg Time to Max Bloom: {custom_metrics['avg_time_to_max_bloom']:.1f} steps")
    if "students_reached_max_bloom" in custom_metrics:
        print(f"Students at Max Bloom: {custom_metrics['students_reached_max_bloom']}")
    
    # ZPD metrics
    print("\n=== TEACHING STRATEGY ===")
    if "zpd_match_rate" in custom_metrics:
        print(f"ZPD Match Rate: {custom_metrics['zpd_match_rate']:.2f}")
    if "teacher_simple_action_rate" in custom_metrics:
        print(f"Simple Explanation Rate: {custom_metrics['teacher_simple_action_rate']:.2f}")
    if "teacher_complex_action_rate" in custom_metrics:
        print(f"Complex Explanation Rate: {custom_metrics['teacher_complex_action_rate']:.2f}")
    
    # Student behavior metrics
    print("\n=== STUDENT BEHAVIOR ===")
    if "class_avg_question_level" in custom_metrics:
        print(f"Average Question Level: {custom_metrics['class_avg_question_level']:.2f}")
    if "advancement_rate_after_question" in custom_metrics:
        print(f"Advancement After Questions: {custom_metrics['advancement_rate_after_question']:.2f}")
    if "advancement_rate_after_study" in custom_metrics:
        print(f"Advancement After Study: {custom_metrics['advancement_rate_after_study']:.2f}")
    
    # Equity metrics
    print("\n=== EQUITY METRICS ===")
    if "avg_bloom_variance" in custom_metrics:
        print(f"Bloom Level Variance: {custom_metrics['avg_bloom_variance']:.2f}")
    if "avg_knowledge_gap" in custom_metrics:
        print(f"Knowledge Gap: {custom_metrics['avg_knowledge_gap']:.2f}")
    
    # Print student types performance if available
    student_types = set()
    for key in custom_metrics:
        if key.endswith("_avg_bloom"):
            student_type = key.replace("_avg_bloom", "")
            student_types.add(student_type)
    
    if student_types:
        print("\n=== PERFORMANCE BY STUDENT TYPE ===")
        for student_type in student_types:
            key = f"{student_type}_avg_bloom"
            if key in custom_metrics:
                print(f"{student_type.capitalize()} Avg Bloom: {custom_metrics[key]:.2f}")

def train(num_iterations, algo):
    """Run the main training loop."""
    print(f"\nStarting training for {num_iterations} iterations...")
    
    # Initialize metrics collection
    metrics_by_iteration = []

    for i in range(num_iterations):
        print(f"\nStarting iteration {i+1}/{num_iterations}...")
        result = algo.train()  # one iteration of training
        
        # Debug: Print raw training results
        print(f"Training result keys: {list(result.keys())}")
        if "env_runners" in result:
            print(f"env_runners keys: {list(result['env_runners'].keys())}")
            if "episodes_this_iter" in result["env_runners"]:
                print(f"Episodes this iteration: {result['env_runners']['episodes_this_iter']}")
            if "custom_metrics" in result["env_runners"]:
                print(f"Custom metrics keys: {list(result['env_runners']['custom_metrics'].keys())}")
            else:
                print("No custom_metrics found in result!")

        # Print results
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        print_metrics(result)
        
        # Store iteration metrics
        metrics_by_iteration.append({
            "iteration": i+1,
            "metrics": result["env_runners"].get("custom_metrics", {})
        })
        
        # Save metrics to JSON after each iteration
        import json
        import os
        os.makedirs("metrics", exist_ok=True)
        
        # Save current iteration metrics
        with open(f"metrics/metrics_iteration_{i+1}.json", "w") as f:
            json.dump(metrics_by_iteration[-1], f, indent=2)
        
        # Save all metrics so far
        with open("metrics/metrics_all.json", "w") as f:
            json.dump(metrics_by_iteration, f, indent=2)

    checkpoint_dir = algo.save()
    algo.stop()
    ray.shutdown()

    print(f"Checkpoint saved in directory: {checkpoint_dir}. Training finished.")
    print(f"All metrics saved to metrics/metrics_all.json")


def create_algo_config(env_config):
    teacher_obs_space, teacher_action_space, student_obs_space, student_action_space = (
        get_tmp_spaces(env_config)
    )
    algo_config = (
        PPOConfig()
        .environment(
            env="classroom_v1",
            env_config=env_config,
            disable_env_checking=True,  # Disable for custom envs if needed, but check if issues arise
        )
        .callbacks(ClassroomCallbacks)
        .framework("torch")  # Or "tf2"
        .rollouts(
            num_rollout_workers=1,  # Number of parallel workers for collecting samples
            # Set the fragment length to the max steps in an episode
            rollout_fragment_length='auto',
            # Set the number of parallel environments per worker
            num_envs_per_worker=DEFAULT_ENV_CONFIG["num_students"],
            # Set batch mode to complete episodes
            batch_mode="complete_episodes",
        )
        # TODO: when we make our environment, obs and action space more complex, we need to make the policy network model more complex as well
        .training(
            gamma=0.99,
            lr=5e-5,
            lambda_=0.95,
            # Set the train batch size to control number of episodes
            # 2 episodes x max_steps x num_students
            train_batch_size=2 * DEFAULT_ENV_CONFIG["max_steps"],
            sgd_minibatch_size=10,  # TODO: changed from 32 to 10 for testing
            # num_sgd_iter=10,
            num_sgd_iter=1,
            model={
                "fcnet_hiddens": [64, 64],
            },
        )
        .multi_agent(
            # Define the policies (agents). Here, both use the same PPO policy class
            # but will have separate instances and weights during training.
            policies={
                "teacher_policy": PolicySpec(
                    policy_class=None,  # Use default PPO policy
                    observation_space=teacher_obs_space,
                    action_space=teacher_action_space,
                ),
                "student_policy": PolicySpec(
                    policy_class=None,
                    observation_space=student_obs_space,
                    action_space=student_action_space,
                ),
            },
            # Function mapping agent IDs to policy IDs
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kwargs: (
                    "teacher_policy" if "teacher" in agent_id else "student_policy"
                )
            ),
            policies_to_train=["teacher_policy", "student_policy"],
        )
        .resources(num_gpus=0)
        # For debugging or simpler setups:
        .debugging(log_level="INFO")  # Set log level (INFO, DEBUG, WARN, ERROR)
    )
    return algo_config


def main(num_iterations=DEFAULT_TRAINING_CONFIG["num_iterations"]):
    # shutdown previous instances if any
    ray.shutdown()

    # local_mode=True can be helpful for debugging but runs sequentially.  # todo: read on what local_mode is
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)

    # This makes the environment name ("classroom_v1") available to RLlib.
    tune.register_env("classroom_v1", lambda config: ClassroomEnv(config))

    algo_config = create_algo_config(DEFAULT_ENV_CONFIG)

    print("Full PPO config:\n", pretty_print(algo_config.to_dict()))
    print(f"Training with {DEFAULT_ENV_CONFIG['num_students']} students of " \
          + f"types: {DEFAULT_ENV_CONFIG['student_types']}")
    algo = algo_config.build()
    train(num_iterations=num_iterations, algo=algo)


if __name__ == "__main__":
    try:
        main(1)  # Run just 1 iteration for testing
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        ray.shutdown()  # shutdown in case of an error
