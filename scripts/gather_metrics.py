import json
import matplotlib.pyplot as plt
import numpy as np
import collections # For defaultdict
import math # For checking NaN

# --- Constants ---
# Predefined agent IDs based on the JSON structure
STUDENT_AGENT_IDS = ['student_0', 'student_1', 'student_2']
TEACHER_AGENT_ID = 'teacher_0'
PREDEFINED_AGENT_IDS = STUDENT_AGENT_IDS + [TEACHER_AGENT_ID]

# Bloom Taxonomy Levels
BLOOM_REMEMBER = 1
BLOOM_UNDERSTAND = 2
BLOOM_APPLY = 3
BLOOM_ANALYZE = 4
BLOOM_EVALUATE = 5
BLOOM_CREATE = 6
NUM_BLOOM_LEVELS = 6 # Max level (used as MAX_BLOOM_LEVEL)
MAX_BLOOM_LEVEL = NUM_BLOOM_LEVELS

BLOOM_LEVEL_MAP = {
    1: "Remember",        # Simplified from "Remembering basic facts"
    2: "Understand",      # Simplified from "Understanding concepts"
    3: "Apply",           # Simplified from "Applying knowledge"
    4: "Analyze",         # Simplified from "Analyzing information"
    5: "Evaluate",        # Simplified from "Evaluating ideas"
    6: "Create",          # Simplified from "Creating new work"
}

# Action Types (for consistency)
TEACHER_ACTIONS = ['Simple', 'Complex']
STUDENT_ACTIONS = ['Study', 'Ask']

# --- Utility Functions ---
def load_data(file_path):
    """ Loads JSON data from a file. """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

# --- Data Extraction ---
def get_all_episodes_data(data):
    """
    Extracts rewards, Bloom levels, actions, ZPD, and termination/truncation
    for all agents for each step of each episode. Uses 'total_episode_number' as global ID.
    """
    if not data:
        return []

    all_episodes_data_temp = []
    processed_global_ids = set()

    for iter_name, iter_data in data.items():
        if not isinstance(iter_data, dict) or "episodes" not in iter_data:
            continue

        for ep_name, ep_data in iter_data.get("episodes", {}).items():
            current_global_ep_id = ep_data.get("total_episode_number")
            if current_global_ep_id is None: continue
            current_global_ep_id = int(current_global_ep_id)
            if current_global_ep_id in processed_global_ids: continue
            processed_global_ids.add(current_global_ep_id)

            episode_info = {
                "episode_global_id": current_global_ep_id,
                "iteration_id": iter_name,
                "episode_in_iteration_id": ep_name,
                "steps": [],
                "terminated": False, # Episode level flag
                "truncated": False   # Episode level flag
            }

            sorted_steps = sorted(
                ep_data.get("steps", {}).items(),
                key=lambda item: int(item[0].split('_')[-1]) if item[0].startswith('step_') else float('inf')
            )

            max_bloom_step_tracker = {sid: None for sid in STUDENT_AGENT_IDS}
            last_step_data = None # To capture episode end state

            for step_idx, (step_name, step_data) in enumerate(sorted_steps):
                try:
                    actual_step_number = int(step_name.split('_')[-1])
                except (ValueError, IndexError): continue

                step_details = {
                    "step_number": actual_step_number,
                    "rewards_calculated": {aid: 0.0 for aid in PREDEFINED_AGENT_IDS},
                    "team_reward_at_step": 0.0,
                    "bloom_levels_after": {sid: None for sid in STUDENT_AGENT_IDS},
                    "current_bloom_levels": {sid: None for sid in STUDENT_AGENT_IDS}, # Added for action/bloom analysis
                    "class_avg_bloom_rounded": None,
                    "student_actions": {sid: None for sid in STUDENT_AGENT_IDS},
                    "teacher_action": None, # Added
                    "student_question_bloom": {sid: None for sid in STUDENT_AGENT_IDS},
                    "student_zpd_alignment": {sid: None for sid in STUDENT_AGENT_IDS}, # Added
                    "student_max_bloom_step": max_bloom_step_tracker.copy()
                }

                if isinstance(step_data, dict):
                    # Rewards
                    rewards_dict = step_data.get("rewards_calculated")
                    if isinstance(rewards_dict, dict):
                        team_reward = 0.0
                        for aid in PREDEFINED_AGENT_IDS:
                            reward = float(rewards_dict.get(aid, 0.0))
                            step_details["rewards_calculated"][aid] = reward
                            team_reward += reward
                        step_details["team_reward_at_step"] = team_reward

                    # Bloom Levels
                    bloom_after_dict = step_data.get("bloom_levels_after")
                    if isinstance(bloom_after_dict, dict):
                        for sid in STUDENT_AGENT_IDS: step_details["bloom_levels_after"][sid] = bloom_after_dict.get(sid)
                    step_details["class_avg_bloom_rounded"] = step_data.get("class_avg_bloom_level_rounded")

                    # Teacher Action
                    step_details["teacher_action"] = step_data.get("teacher_action")

                    # Student Data (Action, Current Bloom, Question, ZPD, Max Bloom Step)
                    students_dict = step_data.get("students")
                    if isinstance(students_dict, dict):
                        for sid in STUDENT_AGENT_IDS:
                            s_data = students_dict.get(sid)
                            if isinstance(s_data, dict):
                                action = s_data.get("action")
                                current_bloom = s_data.get("current_bloom_level")
                                step_details["student_actions"][sid] = action
                                step_details["current_bloom_levels"][sid] = current_bloom # Store current level
                                step_details["student_zpd_alignment"][sid] = s_data.get("zpd_alignment")
                                if action == "Ask": step_details["student_question_bloom"][sid] = s_data.get("question_bloom_level")
                                if s_data.get("max_bloom_reached_at_step") == actual_step_number:
                                    if max_bloom_step_tracker[sid] is None:
                                        max_bloom_step_tracker[sid] = actual_step_number
                                        step_details["student_max_bloom_step"] = max_bloom_step_tracker.copy()

                    # Store last step data for termination/truncation flags
                    if step_idx == len(sorted_steps) - 1:
                        last_step_data = step_data

                episode_info["steps"].append(step_details)

            # Set episode termination/truncation flags from the last step
            if isinstance(last_step_data, dict):
                 term_dict = last_step_data.get("terminated", {})
                 trunc_dict = last_step_data.get("truncated", {})
                 if isinstance(term_dict, dict):
                      episode_info["terminated"] = term_dict.get("__all__", False)
                 if isinstance(trunc_dict, dict):
                      episode_info["truncated"] = trunc_dict.get("__all__", False)
                 # Ensure only one flag is true (termination overrides truncation if both somehow true)
                 if episode_info["terminated"]:
                      episode_info["truncated"] = False


            all_episodes_data_temp.append(episode_info)

    all_episodes_data_temp.sort(key=lambda x: x["episode_global_id"])
    return all_episodes_data_temp


# --- Reward Calculation Functions (Unchanged) ---
def get_cumulative_rewards_per_episode(all_episodes_data):
    """ Calculates cumulative rewards per episode. """
    episode_returns = []
    for ep_data in all_episodes_data:
        current_episode_cumulative_rewards = {agent_id: 0.0 for agent_id in PREDEFINED_AGENT_IDS}
        current_episode_cumulative_rewards['team'] = 0.0
        episode_length = len(ep_data["steps"])
        for step in ep_data["steps"]:
            step_team_reward_from_agents = 0.0
            for agent_id in PREDEFINED_AGENT_IDS:
                reward = step["rewards_calculated"].get(agent_id, 0.0)
                current_episode_cumulative_rewards[agent_id] += reward
                step_team_reward_from_agents += reward
            current_episode_cumulative_rewards['team'] += step_team_reward_from_agents
        episode_returns.append({
            "episode_global_id": ep_data["episode_global_id"],
            "iteration_id": ep_data["iteration_id"],
            "episode_in_iteration_id": ep_data["episode_in_iteration_id"],
            "cumulative_rewards": current_episode_cumulative_rewards,
            "episode_length": episode_length
        })
    return episode_returns

def get_mean_rewards_per_timestep_per_episode(cumulative_rewards_per_episode_data):
    """ Calculates mean rewards per timestep per episode. """
    episode_mean_rewards = []
    for ep_data in cumulative_rewards_per_episode_data:
        mean_rewards = {agent_id: 0.0 for agent_id in PREDEFINED_AGENT_IDS}
        mean_rewards['team'] = 0.0
        episode_length = ep_data["episode_length"]
        if episode_length > 0:
            for agent_id in PREDEFINED_AGENT_IDS:
                mean_rewards[agent_id] = ep_data["cumulative_rewards"].get(agent_id, 0.0) / episode_length
            mean_rewards['team'] = ep_data["cumulative_rewards"].get('team', 0.0) / episode_length
        episode_mean_rewards.append({
            "episode_global_id": ep_data["episode_global_id"],
            "iteration_id": ep_data["iteration_id"],
            "episode_in_iteration_id": ep_data["episode_in_iteration_id"],
            "mean_reward_per_timestep": mean_rewards
        })
    return episode_mean_rewards

def get_agent_specific_cumulative_rewards(cumulative_rewards_per_episode_data):
    """ Aggregates cumulative rewards per agent over episodes. """
    agent_rewards_over_episodes = {agent_id: [] for agent_id in PREDEFINED_AGENT_IDS}
    agent_rewards_over_episodes['team'] = []
    episode_global_ids = []
    sorted_ep_data = sorted(cumulative_rewards_per_episode_data, key=lambda x: x["episode_global_id"])
    for ep_data in sorted_ep_data:
        episode_global_ids.append(ep_data["episode_global_id"])
        for agent_id in PREDEFINED_AGENT_IDS:
            agent_rewards_over_episodes[agent_id].append(ep_data["cumulative_rewards"].get(agent_id, 0.0))
        agent_rewards_over_episodes['team'].append(ep_data["cumulative_rewards"].get('team', 0.0))
    return agent_rewards_over_episodes, episode_global_ids

def get_agent_specific_mean_rewards_per_timestep(mean_rewards_per_timestep_data):
    """ Aggregates mean rewards per timestep per agent over episodes. """
    agent_mean_step_rewards_over_episodes = {agent_id: [] for agent_id in PREDEFINED_AGENT_IDS}
    agent_mean_step_rewards_over_episodes['team'] = []
    episode_global_ids = []
    sorted_ep_data = sorted(mean_rewards_per_timestep_data, key=lambda x: x["episode_global_id"])
    for ep_data in sorted_ep_data:
        episode_global_ids.append(ep_data["episode_global_id"])
        for agent_id in PREDEFINED_AGENT_IDS:
            agent_mean_step_rewards_over_episodes[agent_id].append(ep_data["mean_reward_per_timestep"].get(agent_id, 0.0))
        agent_mean_step_rewards_over_episodes['team'].append(ep_data["mean_reward_per_timestep"].get('team', 0.0))
    return agent_mean_step_rewards_over_episodes, episode_global_ids

def calculate_overall_mean_from_agent_specific_data(agent_specific_rewards):
    """ Calculates overall mean reward from aggregated lists. """
    overall_means = {}
    for agent_id, rewards_list in agent_specific_rewards.items():
        valid_rewards = [r for r in rewards_list if r is not None and not (isinstance(r, float) and math.isnan(r))]
        if valid_rewards:
            overall_means[agent_id] = sum(valid_rewards) / len(valid_rewards)
        else:
            overall_means[agent_id] = None # Indicate no valid data
    return overall_means


# --- Bloom Metrics Calculation Functions (Unchanged) ---
def get_bloom_metrics_per_episode(all_episodes_data):
    """ Calculates average Bloom levels and collects max bloom steps per episode. """
    bloom_metrics_list = []
    for ep_data in all_episodes_data:
        episode_length = len(ep_data["steps"])
        episode_bloom_metrics = {
            "episode_global_id": ep_data["episode_global_id"],
            "avg_student_bloom_levels": {sid: 0.0 for sid in STUDENT_AGENT_IDS},
            "avg_class_bloom_level": 0.0,
            "avg_student_question_bloom": {sid: None for sid in STUDENT_AGENT_IDS},
            "steps_to_max_bloom": {sid: None for sid in STUDENT_AGENT_IDS}
        }
        student_bloom_sum = collections.defaultdict(float)
        student_bloom_count = collections.defaultdict(int)
        class_bloom_sum = 0.0
        class_bloom_count = 0
        student_question_bloom_sum = collections.defaultdict(float)
        student_question_count = collections.defaultdict(int)
        final_step_max_bloom_tracker = {sid: None for sid in STUDENT_AGENT_IDS}
        for step in ep_data["steps"]:
            for sid in STUDENT_AGENT_IDS:
                level = step["bloom_levels_after"].get(sid)
                if level is not None:
                    student_bloom_sum[sid] += float(level)
                    student_bloom_count[sid] += 1
            class_level = step.get("class_avg_bloom_rounded")
            if class_level is not None:
                class_bloom_sum += float(class_level)
                class_bloom_count += 1
            for sid in STUDENT_AGENT_IDS:
                q_level = step["student_question_bloom"].get(sid)
                if q_level is not None:
                    student_question_bloom_sum[sid] += float(q_level)
                    student_question_count[sid] += 1
            if step.get("student_max_bloom_step"):
                 current_tracker = step["student_max_bloom_step"]
                 for sid in STUDENT_AGENT_IDS:
                      if current_tracker.get(sid) is not None: final_step_max_bloom_tracker[sid] = current_tracker[sid]
        for sid in STUDENT_AGENT_IDS:
             if student_bloom_count[sid] > 0: episode_bloom_metrics["avg_student_bloom_levels"][sid] = student_bloom_sum[sid] / student_bloom_count[sid]
             else: episode_bloom_metrics["avg_student_bloom_levels"][sid] = None
        if class_bloom_count > 0: episode_bloom_metrics["avg_class_bloom_level"] = class_bloom_sum / class_bloom_count
        else: episode_bloom_metrics["avg_class_bloom_level"] = None
        for sid in STUDENT_AGENT_IDS:
            if student_question_count[sid] > 0: episode_bloom_metrics["avg_student_question_bloom"][sid] = student_question_bloom_sum[sid] / student_question_count[sid]
        episode_bloom_metrics["steps_to_max_bloom"] = final_step_max_bloom_tracker
        bloom_metrics_list.append(episode_bloom_metrics)
    return bloom_metrics_list

def get_agent_specific_bloom_metrics(bloom_metrics_per_episode_data):
    """ Aggregates average Bloom metrics per agent/class across all episodes. """
    avg_student_bloom = {sid: [] for sid in STUDENT_AGENT_IDS}
    avg_class_bloom = []
    avg_student_question_bloom = {sid: [] for sid in STUDENT_AGENT_IDS}
    steps_to_max_bloom = {sid: [] for sid in STUDENT_AGENT_IDS}
    episode_global_ids = []
    sorted_ep_data = sorted(bloom_metrics_per_episode_data, key=lambda x: x["episode_global_id"])
    for ep_data in sorted_ep_data:
        episode_global_ids.append(ep_data["episode_global_id"])
        avg_class_bloom.append(ep_data.get("avg_class_bloom_level"))
        for sid in STUDENT_AGENT_IDS:
            avg_student_bloom[sid].append(ep_data["avg_student_bloom_levels"].get(sid))
            avg_student_question_bloom[sid].append(ep_data["avg_student_question_bloom"].get(sid))
            steps_to_max_bloom[sid].append(ep_data["steps_to_max_bloom"].get(sid))
    return avg_student_bloom, avg_class_bloom, avg_student_question_bloom, steps_to_max_bloom, episode_global_ids

def calculate_avg_steps_to_max_bloom(steps_to_max_bloom_data):
    """ Calculates the average steps taken to reach max Bloom level. """
    avg_steps = {}
    for sid, steps_list in steps_to_max_bloom_data.items():
        valid_steps = [s for s in steps_list if s is not None]
        if valid_steps: avg_steps[sid] = np.mean(valid_steps)
        else: avg_steps[sid] = None
    return avg_steps


# --- Action, ZPD, Termination Metrics ---

def get_action_zpd_term_metrics_per_episode(all_episodes_data):
    """
    Calculates action proportions, avg ZPD, and termination status per episode.
    """
    metrics_list = []
    for ep_data in all_episodes_data:
        episode_length = len(ep_data["steps"])
        ep_metrics = {
            "episode_global_id": ep_data["episode_global_id"],
            "teacher_action_proportions": {action: 0.0 for action in TEACHER_ACTIONS},
            "student_action_proportions": {sid: {action: 0.0 for action in STUDENT_ACTIONS} for sid in STUDENT_AGENT_IDS},
            "avg_student_zpd": {sid: None for sid in STUDENT_AGENT_IDS},
            "terminated": ep_data.get("terminated", False),
            "truncated": ep_data.get("truncated", False),
            "episode_length": episode_length
        }
        teacher_action_counts = collections.defaultdict(int)
        student_action_counts = {sid: collections.defaultdict(int) for sid in STUDENT_AGENT_IDS}
        student_zpd_sum = collections.defaultdict(float)
        student_zpd_count = collections.defaultdict(int)
        for step in ep_data["steps"]:
            teacher_action = step.get("teacher_action")
            if teacher_action in TEACHER_ACTIONS: teacher_action_counts[teacher_action] += 1
            for sid in STUDENT_AGENT_IDS:
                student_action = step["student_actions"].get(sid)
                if student_action in STUDENT_ACTIONS: student_action_counts[sid][student_action] += 1
                zpd = step["student_zpd_alignment"].get(sid)
                if zpd is not None and not (isinstance(zpd, float) and math.isnan(zpd)):
                    student_zpd_sum[sid] += float(zpd)
                    student_zpd_count[sid] += 1
        if episode_length > 0:
            for action, count in teacher_action_counts.items(): ep_metrics["teacher_action_proportions"][action] = count / episode_length
            for sid in STUDENT_AGENT_IDS:
                total_student_actions = sum(student_action_counts[sid].values())
                if total_student_actions > 0:
                    for action, count in student_action_counts[sid].items(): ep_metrics["student_action_proportions"][sid][action] = count / total_student_actions
                if student_zpd_count[sid] > 0: ep_metrics["avg_student_zpd"][sid] = student_zpd_sum[sid] / student_zpd_count[sid]
        metrics_list.append(ep_metrics)
    return metrics_list

def get_agent_specific_action_zpd_term_metrics(action_zpd_term_metrics_per_episode):
    """
    Aggregates action proportions, ZPD averages, and termination/truncation info over episodes.
    """
    teacher_action_props = {action: [] for action in TEACHER_ACTIONS}
    student_action_props = {sid: {action: [] for action in STUDENT_ACTIONS} for sid in STUDENT_AGENT_IDS}
    avg_student_zpd = {sid: [] for sid in STUDENT_AGENT_IDS}
    terminated_flags, truncated_flags, episode_lengths, episode_global_ids = [], [], [], []
    sorted_ep_data = sorted(action_zpd_term_metrics_per_episode, key=lambda x: x["episode_global_id"])
    for ep_data in sorted_ep_data:
        episode_global_ids.append(ep_data["episode_global_id"])
        terminated_flags.append(ep_data["terminated"])
        truncated_flags.append(ep_data["truncated"])
        episode_lengths.append(ep_data["episode_length"])
        for action in TEACHER_ACTIONS: teacher_action_props[action].append(ep_data["teacher_action_proportions"].get(action, 0.0))
        for sid in STUDENT_AGENT_IDS:
            for action in STUDENT_ACTIONS: student_action_props[sid][action].append(ep_data["student_action_proportions"][sid].get(action, 0.0))
            avg_student_zpd[sid].append(ep_data["avg_student_zpd"].get(sid))
    return (teacher_action_props, student_action_props, avg_student_zpd,
            terminated_flags, truncated_flags, episode_lengths, episode_global_ids)

def aggregate_student_action_by_bloom(all_episodes_data):
    """ Aggregates counts of student actions based on their current Bloom level. """
    action_bloom_counts = {level: collections.defaultdict(int) for level in range(BLOOM_REMEMBER, MAX_BLOOM_LEVEL + 1)}
    for ep_data in all_episodes_data:
        for step in ep_data["steps"]:
            for sid in STUDENT_AGENT_IDS:
                action = step["student_actions"].get(sid)
                bloom_level = step["current_bloom_levels"].get(sid)
                if action in STUDENT_ACTIONS and bloom_level in action_bloom_counts:
                    action_bloom_counts[bloom_level][action] += 1
    return action_bloom_counts


# --- Plotting Functions ---

# plot_rewards_over_episodes (Unchanged)
def plot_rewards_over_episodes(agent_rewards, episode_ids, title, ylabel):
    """ Plots raw rewards over episodes, filtering None values. """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(PREDEFINED_AGENT_IDS)))
    agent_color_map = {agent_id: colors[i] for i, agent_id in enumerate(PREDEFINED_AGENT_IDS)}
    agent_color_map['team'] = 'red'
    has_data = False
    for agent_id in PREDEFINED_AGENT_IDS + ['team']:
        if agent_id in agent_rewards:
            rewards_data = agent_rewards[agent_id]
            valid_ids = [ep_id for ep_id, r in zip(episode_ids, rewards_data) if r is not None]
            valid_rewards = [r for r in rewards_data if r is not None]
            if valid_ids:
                label = agent_id if agent_id != 'team' else 'Team'
                linewidth = 1 if agent_id != 'team' else 2
                linestyle = '-' if agent_id != 'team' else '--'
                plt.plot(valid_ids, valid_rewards, label=label, color=agent_color_map[agent_id], alpha=0.7, linewidth=linewidth, linestyle=linestyle)
                has_data = True
    plt.title(title)
    plt.xlabel("Global Episode ID")
    plt.ylabel(ylabel)
    if has_data: plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


# calculate_moving_average (FIXED)
def calculate_moving_average(data, window_size):
    """ Calculates moving average. Handles lists and numpy arrays. """
    # Check if data is None or empty (works for lists and numpy arrays via size/len)
    is_empty = False
    if data is None:
         is_empty = True
    elif isinstance(data, np.ndarray):
        if data.size == 0:
            is_empty = True
    elif not data: # Handles empty lists/tuples etc.
        is_empty = True

    # Also check if window size is valid and data is long enough
    if is_empty or window_size <= 0 or len(data) < window_size:
        return [] # Return empty list for invalid inputs or insufficient data

    # Ensure data is a numpy array for convolution
    data_np = np.asarray(data)
    # Calculate moving average using convolution
    return np.convolve(data_np, np.ones(window_size)/window_size, mode='valid')


# plot_moving_average_rewards (Unchanged)
def plot_moving_average_rewards(agent_rewards, episode_ids, window_size, title, ylabel):
    """ Plots moving average of rewards over episodes. """
    if window_size <= 0: return
    if not agent_rewards or not episode_ids: return
    filtered_agent_rewards, filtered_episode_ids, max_len = {}, {}, 0
    for agent_id, rewards in agent_rewards.items():
        valid_ids = [ep_id for ep_id, r in zip(episode_ids, rewards) if r is not None]
        valid_rewards = [r for r in rewards if r is not None]
        if valid_rewards:
            filtered_agent_rewards[agent_id], filtered_episode_ids[agent_id] = valid_rewards, valid_ids
            max_len = max(max_len, len(valid_rewards))
    if window_size > max_len:
        print(f"Warning: MA window ({window_size}) > max data points ({max_len}). Plotting raw.")
        plot_rewards_over_episodes(agent_rewards, episode_ids, title + " (Raw Data)", ylabel)
        return
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(PREDEFINED_AGENT_IDS)))
    agent_color_map = {aid: colors[i] for i, aid in enumerate(PREDEFINED_AGENT_IDS)}
    agent_color_map['team'] = 'red'
    has_ma_data = False
    for agent_id in PREDEFINED_AGENT_IDS + ['team']:
        if agent_id in filtered_agent_rewards and len(filtered_agent_rewards[agent_id]) >= window_size:
            moving_avg = calculate_moving_average(filtered_agent_rewards[agent_id], window_size)
            ma_episode_ids = filtered_episode_ids[agent_id][window_size-1:]
            label = f"{agent_id} (MA)" if agent_id != 'team' else 'Team (MA)'
            linewidth = 1 if agent_id != 'team' else 2
            linestyle = '-' if agent_id != 'team' else '--'
            plt.plot(ma_episode_ids, moving_avg, label=label, color=agent_color_map[agent_id], alpha=0.9, linewidth=linewidth, linestyle=linestyle)
            has_ma_data = True
    plt.title(f"{title} (Moving Average, Window={window_size})")
    plt.xlabel("Global Episode ID")
    plt.ylabel(ylabel)
    if has_ma_data: plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


# plot_bloom_levels_over_episodes (Unchanged)
def plot_bloom_levels_over_episodes(avg_student_bloom, avg_class_bloom, episode_ids, title):
    """ Plots average Bloom levels (students and class) over episodes with descriptive y-axis. """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(STUDENT_AGENT_IDS)))
    student_color_map = {sid: colors[i] for i, sid in enumerate(STUDENT_AGENT_IDS)}
    has_data = False
    for sid in STUDENT_AGENT_IDS:
        valid_ids = [ep_id for ep_id, level in zip(episode_ids, avg_student_bloom.get(sid, [])) if level is not None]
        valid_levels = [level for level in avg_student_bloom.get(sid, []) if level is not None]
        if valid_ids:
             plt.plot(valid_ids, valid_levels, label=f"{sid} Avg Bloom", color=student_color_map[sid], alpha=0.8)
             has_data = True
    valid_class_ids = [ep_id for ep_id, level in zip(episode_ids, avg_class_bloom) if level is not None]
    valid_class_levels = [level for level in avg_class_bloom if level is not None]
    if valid_class_ids:
        plt.plot(valid_class_ids, valid_class_levels, label="Class Avg Bloom", color='black', linewidth=2, linestyle=':')
        has_data = True
    plt.title(title)
    plt.xlabel("Global Episode ID")
    plt.ylabel("Average Bloom Level")
    tick_positions = range(BLOOM_REMEMBER, MAX_BLOOM_LEVEL + 1)
    tick_labels = [f"{i}: {BLOOM_LEVEL_MAP.get(i, '')}" for i in tick_positions]
    plt.yticks(tick_positions, tick_labels)
    plt.ylim(bottom=0.5, top=MAX_BLOOM_LEVEL + 0.5)
    if has_data: plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

# plot_question_bloom_levels_over_episodes (Unchanged)
def plot_question_bloom_levels_over_episodes(avg_student_question_bloom, episode_ids, title):
    """ Plots average question Bloom levels per student over episodes with descriptive y-axis. """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.autumn(np.linspace(0, 1, len(STUDENT_AGENT_IDS)))
    student_color_map = {sid: colors[i] for i, sid in enumerate(STUDENT_AGENT_IDS)}
    has_data = False
    for sid in STUDENT_AGENT_IDS:
        student_data = avg_student_question_bloom.get(sid, [])
        valid_episode_ids = [ep_id for ep_id, q_level in zip(episode_ids, student_data) if q_level is not None]
        valid_q_levels = [q_level for q_level in student_data if q_level is not None]
        if valid_episode_ids:
            plt.plot(valid_episode_ids, valid_q_levels, label=f"{sid} Avg Q Level", color=student_color_map[sid], marker='o', linestyle='-', markersize=4, alpha=0.7)
            has_data = True
    plt.title(title)
    plt.xlabel("Global Episode ID")
    plt.ylabel("Average Question Bloom Level")
    tick_positions = range(BLOOM_REMEMBER, MAX_BLOOM_LEVEL + 1)
    tick_labels = [f"{i}: {BLOOM_LEVEL_MAP.get(i, '')}" for i in tick_positions]
    plt.yticks(tick_positions, tick_labels)
    plt.ylim(bottom=0.5, top=MAX_BLOOM_LEVEL + 0.5)
    if has_data: plt.legend(loc='best')
    else: plt.text(0.5, 0.5, 'No question data available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

# plot_steps_to_max_bloom_histogram (Unchanged)
def plot_steps_to_max_bloom_histogram(steps_to_max_bloom_data, title):
    """ Plots histograms of the number of steps taken to reach the max Bloom level. """
    plt.figure(figsize=(10, 7))
    num_students = len(STUDENT_AGENT_IDS)
    colors = plt.cm.winter(np.linspace(0, 1, num_students))
    student_color_map = {sid: colors[i] for i, sid in enumerate(STUDENT_AGENT_IDS)}
    plot_data, max_step, has_data = {}, 0, False
    for sid in STUDENT_AGENT_IDS:
        valid_steps = [s for s in steps_to_max_bloom_data.get(sid, []) if s is not None]
        if valid_steps:
            plot_data[sid], max_step, has_data = valid_steps, max(max_step, max(valid_steps)), True
    if not has_data:
         plt.text(0.5, 0.5, 'No max bloom data available', ha='center', va='center', transform=plt.gca().transAxes)
         plt.title(title + " (No Data)")
         plt.tight_layout(); return
    bin_width = 1
    bins = np.arange(0, max_step + bin_width + 1, bin_width) - 0.5
    bar_width_factor = 0.8 / max(1, len(plot_data)) # Adjust width based on actual students with data
    plotted_students = list(plot_data.keys()) # Get students who actually have data
    for i, sid in enumerate(plotted_students):
         hist, _ = np.histogram(plot_data[sid], bins=bins)
         offset = (i - (len(plotted_students) - 1) / 2) * bin_width * bar_width_factor
         plt.bar(bins[:-1] + bin_width/2 + offset, hist, width=bin_width * bar_width_factor, label=sid, color=student_color_map[sid], alpha=0.7)
    plt.title(title)
    plt.xlabel(f"Steps Taken to Reach Max Bloom Level ({MAX_BLOOM_LEVEL}: {BLOOM_LEVEL_MAP[MAX_BLOOM_LEVEL]})")
    plt.ylabel("Number of Episodes")
    plt.xticks(np.arange(0, max_step + 1, max(1, (max_step+1) // 10)))
    plt.xlim(left=-0.5)
    plt.legend(loc='best')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()


# --- Action, ZPD, Termination Plotting Functions (Unchanged) ---

def plot_teacher_action_proportions(teacher_action_props, episode_ids, title):
    """ Plots the proportion of Simple vs Complex teacher actions over episodes. """
    plt.figure(figsize=(12, 6))
    colors = {'Simple': 'skyblue', 'Complex': 'salmon'}
    bottom = np.zeros(len(episode_ids))
    for action in TEACHER_ACTIONS:
        proportions = teacher_action_props.get(action, [])
        if proportions:
             plt.bar(episode_ids, proportions, bottom=bottom, label=action, color=colors.get(action, 'gray'), width=0.8)
             bottom += [p if p is not None else 0 for p in proportions]
    plt.title(title)
    plt.xlabel("Global Episode ID")
    plt.ylabel("Proportion of Actions")
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_student_action_proportions(student_action_props, episode_ids, title_prefix):
    """ Plots the proportion of Study vs Ask actions for each student over episodes. """
    num_students = len(STUDENT_AGENT_IDS)
    fig, axes = plt.subplots(num_students, 1, figsize=(12, 5 * num_students), sharex=True)
    if num_students == 1: axes = [axes]
    colors = {'Study': 'lightgreen', 'Ask': 'gold'}
    for i, sid in enumerate(STUDENT_AGENT_IDS):
        ax = axes[i]
        bottom = np.zeros(len(episode_ids))
        has_data = False
        props_for_student = student_action_props.get(sid, {})
        for action in STUDENT_ACTIONS:
            proportions = props_for_student.get(action, [])
            if proportions:
                 valid_ids_idx = [idx for idx, p in enumerate(proportions) if p is not None]
                 valid_ids = [episode_ids[idx] for idx in valid_ids_idx]
                 valid_props = [proportions[idx] for idx in valid_ids_idx]
                 if valid_ids:
                     current_bottom = [bottom[idx] for idx in valid_ids_idx]
                     ax.bar(valid_ids, valid_props, bottom=current_bottom, label=action, color=colors.get(action, 'gray'), width=0.8)
                     for idx, p_idx in enumerate(valid_ids_idx): bottom[p_idx] += valid_props[idx]
                     has_data = True
        ax.set_title(f"{title_prefix} - {sid}")
        ax.set_ylabel("Proportion of Actions")
        ax.set_ylim(0, 1)
        if has_data: ax.legend(loc='best')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[-1].set_xlabel("Global Episode ID")
    plt.tight_layout()

def plot_average_zpd_alignment(avg_student_zpd, episode_ids, title):
    """ Plots the average ZPD alignment per student over episodes. """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(STUDENT_AGENT_IDS)))
    student_color_map = {sid: colors[i] for i, sid in enumerate(STUDENT_AGENT_IDS)}
    has_data = False
    for sid in STUDENT_AGENT_IDS:
        zpd_data = avg_student_zpd.get(sid, [])
        valid_ids = [ep_id for ep_id, zpd in zip(episode_ids, zpd_data) if zpd is not None]
        valid_zpd = [zpd for zpd in zpd_data if zpd is not None]
        if valid_ids:
            plt.plot(valid_ids, valid_zpd, label=f"{sid} Avg ZPD", color=student_color_map[sid], marker='.', linestyle='-', markersize=5, alpha=0.7)
            has_data = True
    plt.title(title)
    plt.xlabel("Global Episode ID")
    plt.ylabel("Average ZPD Alignment")
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='ZPD=0 (Aligned)')
    plt.ylim(-1.1, 1.1)
    if has_data: plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_termination_truncation_rate(terminated_flags, truncated_flags, episode_ids, window_size, title):
    """ Plots the moving average rate of episode termination and truncation. """
    if window_size <= 0 or len(episode_ids) < window_size: # Check length against original episode IDs
        print(f"Warning: Invalid window size ({window_size}) or insufficient data ({len(episode_ids)}) for termination plot. Skipping.")
        return

    plt.figure(figsize=(12, 6))
    term_numeric = np.array([1 if t else 0 for t in terminated_flags])
    trunc_numeric = np.array([1 if tr else 0 for tr in truncated_flags])

    # Ensure we have enough data points *before* calculating MA
    if len(term_numeric) < window_size or len(trunc_numeric) < window_size:
         print(f"Warning: Insufficient data points ({len(term_numeric)}) for MA window ({window_size}). Skipping termination plot.")
         plt.close() # Close the figure created
         return


    term_rate_ma = calculate_moving_average(term_numeric, window_size)
    trunc_rate_ma = calculate_moving_average(trunc_numeric, window_size)
    ma_episode_ids = episode_ids[window_size-1:] # MA corresponds to the end of the window

    plt.plot(ma_episode_ids, term_rate_ma, label=f"Termination Rate (MA {window_size})", color='green', linewidth=2)
    plt.plot(ma_episode_ids, trunc_rate_ma, label=f"Truncation Rate (MA {window_size})", color='orange', linewidth=2)

    plt.title(title + f" (Moving Average, Window={window_size})")
    plt.xlabel("Global Episode ID")
    plt.ylabel("Rate (Proportion of Episodes)")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_episode_lengths(episode_lengths, terminated_flags, truncated_flags, episode_ids, title):
    """ Plots episode lengths over time, colored by termination/truncation status. """
    plt.figure(figsize=(12, 6))
    colors = {'terminated': 'green', 'truncated': 'orange', 'other': 'grey'}
    markers = {'terminated': 'o', 'truncated': 'x', 'other': '.'}
    labels = {'terminated': 'Terminated (Goal Reached)', 'truncated': 'Truncated (Max Steps)', 'other':'Other/Unknown'}
    data_by_status = collections.defaultdict(lambda: {'ids': [], 'lengths': []})
    for i, length in enumerate(episode_lengths):
        status = 'other'
        if terminated_flags[i]: status = 'terminated'
        elif truncated_flags[i]: status = 'truncated'
        data_by_status[status]['ids'].append(episode_ids[i])
        data_by_status[status]['lengths'].append(length)
    plotted_statuses = []
    for status, data in data_by_status.items():
        if data['ids']:
             plt.scatter(data['ids'], data['lengths'], color=colors[status], marker=markers[status], alpha=0.6, label=labels[status], s=30)
             plotted_statuses.append(status)
    plt.title(title)
    plt.xlabel("Global Episode ID")
    plt.ylabel("Episode Length (Steps)")
    plt.ylim(bottom=0)
    if plotted_statuses: plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_student_actions_by_bloom(action_bloom_counts, title):
    """ Plots a bar chart of student actions (Study/Ask) grouped by Bloom level. """
    plt.figure(figsize=(12, 7))
    levels = sorted(action_bloom_counts.keys())
    study_counts = [action_bloom_counts[level].get('Study', 0) for level in levels]
    ask_counts = [action_bloom_counts[level].get('Ask', 0) for level in levels]
    x = np.arange(len(levels))
    width = 0.35
    rects1 = plt.bar(x - width/2, study_counts, width, label='Study', color='lightgreen')
    rects2 = plt.bar(x + width/2, ask_counts, width, label='Ask', color='gold')
    plt.ylabel('Total Count Across All Steps')
    plt.xlabel('Student Bloom Level at Step')
    plt.title(title)
    plt.xticks(x, [f"{lvl}: {BLOOM_LEVEL_MAP.get(lvl, '')}" for lvl in levels], rotation=45, ha="right")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()


# --- Main Execution ---
if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    raw_data = load_data(file_path)

    if raw_data:
        # --- Data Processing ---
        all_episodes_data = get_all_episodes_data(raw_data)
        print(f"Processed {len(all_episodes_data)} episodes with step-by-step data.")

        # --- Reward Metrics ---
        cumulative_rewards_data = get_cumulative_rewards_per_episode(all_episodes_data)
        print(f"Calculated cumulative rewards for {len(cumulative_rewards_data)} episodes.")
        mean_step_rewards_data = get_mean_rewards_per_timestep_per_episode(cumulative_rewards_data)
        print(f"Calculated mean reward per timestep for {len(mean_step_rewards_data)} episodes.")
        agent_cumulative_returns, ep_ids_rewards = get_agent_specific_cumulative_rewards(cumulative_rewards_data)
        print("Aggregated agent-specific cumulative rewards.")
        agent_mean_step_rewards, _ = get_agent_specific_mean_rewards_per_timestep(mean_step_rewards_data) # Reuse ep_ids_rewards
        print("Aggregated agent-specific mean rewards per timestep.")
        overall_avg_cumulative = calculate_overall_mean_from_agent_specific_data(agent_cumulative_returns)
        print("\nOverall Mean Cumulative Reward (Return) per Agent/Team:")
        for agent, avg_reward in overall_avg_cumulative.items(): print(f"  {agent}: {avg_reward:.4f}" if avg_reward is not None else f"  {agent}: N/A")
        print("-" * 30)
        overall_avg_mean_step = calculate_overall_mean_from_agent_specific_data(agent_mean_step_rewards)
        print("\nOverall Mean Reward per Timestep per Agent/Team:")
        for agent, avg_reward in overall_avg_mean_step.items(): print(f"  {agent}: {avg_reward:.4f}" if avg_reward is not None else f"  {agent}: N/A")
        print("-" * 30)

        # --- Bloom Metrics ---
        bloom_metrics_episodes = get_bloom_metrics_per_episode(all_episodes_data)
        print(f"Calculated Bloom metrics for {len(bloom_metrics_episodes)} episodes.")
        avg_student_bloom, avg_class_bloom, avg_student_question_bloom, steps_to_max_bloom, bloom_ep_ids = get_agent_specific_bloom_metrics(bloom_metrics_episodes)
        print("Aggregated agent-specific Bloom metrics.")
        avg_steps_max = calculate_avg_steps_to_max_bloom(steps_to_max_bloom)
        print("\nAverage Steps to Reach Max Bloom (Level 6):")
        for sid, avg_steps in avg_steps_max.items():
            if avg_steps is not None: print(f"  {sid}: {avg_steps:.2f} steps (averaged over episodes where max was reached)")
            else: print(f"  {sid}: Max Bloom level ({MAX_BLOOM_LEVEL}) never reached.")
        print("-" * 30)

        # --- Action, ZPD, Termination Metrics ---
        action_zpd_term_metrics = get_action_zpd_term_metrics_per_episode(all_episodes_data)
        print(f"Calculated Action/ZPD/Termination metrics for {len(action_zpd_term_metrics)} episodes.")
        (teacher_action_props, student_action_props, avg_student_zpd,
         terminated_flags, truncated_flags, episode_lengths, action_zpd_ep_ids) = get_agent_specific_action_zpd_term_metrics(action_zpd_term_metrics)
        print("Aggregated agent-specific Action/ZPD/Termination metrics.")
        action_bloom_counts = aggregate_student_action_by_bloom(all_episodes_data)
        print("Aggregated student actions by Bloom level.")
        print("-" * 30)


        # --- Generate Plots ---
        print("\nGenerating plots...")

        # -- Reward Plots --
        plot_rewards_over_episodes(agent_cumulative_returns, ep_ids_rewards, "Cumulative Reward (Return) per Episode", "Cumulative Reward")
        plot_moving_average_rewards(agent_cumulative_returns, ep_ids_rewards, 5, "Cumulative Reward (Return) per Episode", "Moving Average Cumulative Reward")
        plot_rewards_over_episodes(agent_mean_step_rewards, ep_ids_rewards, "Mean Reward per Timestep per Episode", "Mean Reward per Timestep")
        plot_moving_average_rewards(agent_mean_step_rewards, ep_ids_rewards, 5, "Mean Reward per Timestep per Episode", "Moving Average Mean Reward per Timestep")

        # -- Bloom Plots --
        plot_bloom_levels_over_episodes(avg_student_bloom, avg_class_bloom, bloom_ep_ids, "Average Bloom Level per Episode")
        plot_question_bloom_levels_over_episodes(avg_student_question_bloom, bloom_ep_ids, "Average Question Bloom Level per Episode (When Asking)")
        plot_steps_to_max_bloom_histogram(steps_to_max_bloom, "Distribution of Steps to Reach Max Bloom Level")

        # -- Action, ZPD, Termination Plots --
        plot_teacher_action_proportions(teacher_action_props, action_zpd_ep_ids, "Teacher Action Proportions per Episode")
        plot_student_action_proportions(student_action_props, action_zpd_ep_ids, "Student Action Proportions per Episode")
        plot_average_zpd_alignment(avg_student_zpd, action_zpd_ep_ids, "Average Student ZPD Alignment per Episode")
        plot_termination_truncation_rate(terminated_flags, truncated_flags, action_zpd_ep_ids, 5, "Episode End Status Rate")
        plot_episode_lengths(episode_lengths, terminated_flags, truncated_flags, action_zpd_ep_ids, "Episode Lengths Over Time")
        plot_student_actions_by_bloom(action_bloom_counts, "Student Actions Taken at Each Bloom Level (Global Count)")


        # Display all generated plots
        plt.show()
        print("Plots generated and displayed.")

    else:
        print("Could not load or process data.")

