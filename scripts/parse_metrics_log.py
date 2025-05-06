import ast  # For safely evaluating dictionary-like strings
import json
import re
import sys


def parse_log_to_json(log_file_path, json_file_path):
    """
    Parses a multi-agent RL log file and converts it into a structured JSON format.

    Args:
        log_file_path (str): The path to the input log file.
        json_file_path (str): The path where the output JSON file will be saved.
    """
    parsed_data = {}
    current_iteration = None
    current_episode_total = None
    current_episode_in_iteration = None
    current_step = None
    episode_base = 0 # To calculate episode number within an iteration
    total_episodes_in_prev_iterations = 0 # Alternative way to track episode_base

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- Iteration Start ---
        iteration_match = re.match(r'------- Starting Iteration (\d+)/(\d+) -------', line)
        if iteration_match:
            current_iteration = int(iteration_match.group(1))
            total_iterations = int(iteration_match.group(2))
            # Reset episode base using the last total episode count if available
            if current_episode_total is not None:
                 total_episodes_in_prev_iterations = current_episode_total
            parsed_data[f"iteration_{current_iteration}"] = {
                "total_iterations": total_iterations,
                "episodes": {}
            }
            print(f"Processing Iteration {current_iteration}/{total_iterations}")
            i += 1
            continue

        # --- Episode Start ---
        episode_match = re.match(r'\*+Episode (\d+) has started\*+', line)
        if episode_match and current_iteration is not None:
            current_episode_total = int(episode_match.group(1))
            current_episode_in_iteration = current_episode_total - total_episodes_in_prev_iterations
            current_step = None # Reset step for new episode
            iteration_key = f"iteration_{current_iteration}"
            episode_key = f"episode_{current_episode_in_iteration}"
            if iteration_key in parsed_data:
                 parsed_data[iteration_key]["episodes"][episode_key] = {
                     "total_episode_number": current_episode_total,
                     "steps": {}
                 }
            print(f"  Processing Episode {current_episode_in_iteration} (Total: {current_episode_total})")
            i += 1
            continue

        # --- Step Start ---
        step_match = re.match(r'-----Step (\d+) has started-----', line)
        if step_match and current_iteration is not None and current_episode_in_iteration is not None:
            current_step = int(step_match.group(1))
            iteration_key = f"iteration_{current_iteration}"
            episode_key = f"episode_{current_episode_in_iteration}"
            step_key = f"step_{current_step}"

            # Initialize step data structure
            step_data = {"students": {}}

            # --- Parse Step Data ---
            i += 1 # Move to the next line after step start
            while i < len(lines):
                step_line = lines[i].strip()

                # Break if we hit the next step, episode, or iteration marker, or EOF
                if not step_line or \
                   step_line.startswith('-----Step') or \
                   step_line.startswith('**********Episode') or \
                   step_line.startswith('------- Starting Iteration'):
                    break # End of current step's data

                # Teacher Action
                teacher_action_match = re.match(r'Teacher Action: (.*)', step_line)
                if teacher_action_match:
                    step_data['teacher_action'] = teacher_action_match.group(1).strip()
                    i += 1
                    continue

                # Student Data (Action, Bloom, Advanced)
                student_action_match = re.match(r'(student_\d+) Action: (Study|Ask) \(Current Bloom: (\d+) - (.*?)\)', step_line)
                if student_action_match:
                    student_id = student_action_match.group(1)
                    action = student_action_match.group(2)
                    bloom_level = int(student_action_match.group(3))
                    bloom_desc = student_action_match.group(4).strip(')') # Remove trailing parenthesis if present

                    if student_id not in step_data["students"]:
                        step_data["students"][student_id] = {}

                    step_data["students"][student_id]['action'] = action
                    step_data["students"][student_id]['current_bloom_level'] = bloom_level
                    step_data["students"][student_id]['current_bloom_desc'] = bloom_desc
                    i += 1
                    continue

                # Student Level Advanced
                student_advanced_match = re.match(r'(student_\d+) Level Advanced: (TRUE|FALSE)', step_line)
                if student_advanced_match:
                    student_id = student_advanced_match.group(1)
                    level_advanced = student_advanced_match.group(2) == 'TRUE'
                    if student_id in step_data["students"]:
                        step_data["students"][student_id]['level_advanced'] = level_advanced
                    i += 1
                    continue

                # Student Current Bloom Level Desc (Only for 'Ask')
                student_bloom_desc_match = re.match(r'(student_\d+) Current Student Bloom Level Desc: (.*)', step_line)
                if student_bloom_desc_match:
                    student_id = student_bloom_desc_match.group(1)
                    desc = student_bloom_desc_match.group(2).strip()
                    if student_id in step_data["students"]:
                         # Simple categorization based on the first word
                        step_data["students"][student_id]['current_student_bloom_level_desc_full'] = desc
                        step_data["students"][student_id]['current_student_bloom_level_desc_simple'] = desc.split(' ')[0].replace('Applying', 'Apply').replace('Understanding', 'Understand').replace('Remembering', 'Remember').replace('Analyzing', 'Analyze').replace('Evaluating', 'Evaluate').replace('Creating', 'Create')

                    i += 1
                    continue

                # Student Estimated Question Bloom Level (Only for 'Ask')
                student_est_q_match = re.match(r'(student_\d+): Estimated Question Bloom Level and Description: (\d+) and (.*)', step_line)
                if student_est_q_match:
                    student_id = student_est_q_match.group(1)
                    level = int(student_est_q_match.group(2))
                    desc = student_est_q_match.group(3).strip()
                    if student_id in step_data["students"]:
                        step_data["students"][student_id]['estimated_question_bloom_level'] = level
                        step_data["students"][student_id]['estimated_question_bloom_desc'] = desc
                    i += 1
                    continue

                # Current Class Avg Bloom Level
                avg_bloom_match = re.match(r'Current Class Avg Bloom Level: ([\d.]+) -> (\d+)', step_line)
                if avg_bloom_match:
                    step_data['class_avg_bloom_level_unrounded'] = float(avg_bloom_match.group(1))
                    step_data['class_avg_bloom_level_rounded'] = int(avg_bloom_match.group(2))
                    i += 1
                    continue

                
                # NEW Bloom levels update
                # Regex captures the two dicts and optional trailing period/space
                bloom_update_match = re.match(r'NEW Bloom levels update: (\{.*?\}) -> (\{.*?\})\.?\s*$', step_line)
                if bloom_update_match:
                    # print(f"DEBUG: Matched Bloom update line {i+1}: '{step_line}'") # Uncomment for debugging
                    try:
                        # Extract the dictionary strings
                        before_str = bloom_update_match.group(1)
                        after_str = bloom_update_match.group(2)
                        # Use ast.literal_eval to safely parse them into Python dicts
                        step_data['bloom_levels_before'] = ast.literal_eval(before_str)
                        step_data['bloom_levels_after'] = ast.literal_eval(after_str)
                    except (ValueError, SyntaxError) as e:
                        # Handle cases where the string might not be a perfect dictionary literal
                        print(f"Warning: Could not parse bloom update string at line {i+1}: '{step_line}' - Error: {e}")
                        step_data['bloom_levels_before'] = None
                        step_data['bloom_levels_after'] = None
                    i += 1
                    continue # Move to the next line
                
                # Parse "student asked question" line
                student_asked_match = re.match(r'\t(student_\d+) asked question with estimated level: (\d+) \(current level: (\d+)\)', step_line)
                if student_asked_match:
                    student_id = student_asked_match.group(1)
                    est_level = int(student_asked_match.group(2))
                    curr_level = int(student_asked_match.group(3))
                    if student_id in step_data["students"]:
                        # Store as a dictionary or individual keys
                        step_data["students"][student_id]['asked_question_details'] = {
                            'estimated_level': est_level,
                            'current_level_at_ask': curr_level
                        }
                    else:
                        print(f"Warning: Found 'asked question' for unknown {student_id} at line {i+1}")
                    i += 1
                    continue

                # Parse "student received question quality reward" line
                question_reward_match = re.match(r'\t(student_\d+) received question quality reward: \+?(-?[\d.]+)', step_line)
                if question_reward_match:
                    student_id = question_reward_match.group(1)
                    reward_value_str = question_reward_match.group(2)
                    try:
                        reward_value = float(reward_value_str)
                        if student_id in step_data["students"]:
                            step_data["students"][student_id]['question_quality_reward'] = reward_value
                        else:
                            print(f"Warning: Found 'question quality reward' for unknown {student_id} at line {i+1}")
                    except ValueError:
                        print(f"Warning: Could not convert question quality reward '{reward_value_str}' to float at line {i+1}")
                    i += 1
                    continue

                # Parse "student received advancement reward" line
                advancement_reward_match = re.match(r'\t(student_\d+) received advancement reward: \+?(-?[\d.]+)', step_line)
                if advancement_reward_match:
                    student_id = advancement_reward_match.group(1)
                    reward_value_str = advancement_reward_match.group(2)
                    try:
                        reward_value = float(reward_value_str)
                        if student_id in step_data["students"]:
                            step_data["students"][student_id]['advancement_reward'] = reward_value
                        else:
                            print(f"Warning: Found 'advancement reward' for unknown {student_id} at line {i+1}")
                    except ValueError:
                        print(f"Warning: Could not convert advancement reward '{reward_value_str}' to float at line {i+1}")

                    i += 1
                    continue

                # Reward Breakdown Header
                if step_line == 'Reward Breakdown: (in teacher rewards)' or step_line == 'teacher reward breakdown':
                    # print("B"*10) # Uncomment for debugging
                    # Initialize the dictionary when the header is found
                    if 'reward_breakdown' not in step_data: # reward_breakdown is for TEACHER
                         step_data['reward_breakdown'] = {}
                    i += 1
                    continue # Move to the next line (the first item)
                
                if 'reward_breakdown' in step_data:
                    if 'Progress:' in step_line:
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['progress'] = value
                        i += 1
                        continue
                    if 'Equity Penalty:' in step_line:
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['equity_penalty'] = value
                        i += 1
                        continue
                    if 'Engagement:' in step_line:
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['engagement'] = value
                        i += 1
                        continue
                    if 'Explanation Norm:' in step_line:
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['explanation_norm'] = value
                        i += 1
                        continue
                    if 'Time Penalty:' in step_line:
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['time_penalty'] = value
                        i += 1
                        continue
                    if 'teacher_0 - Advancement Contribution' in step_line:
                        print("A"*10) # Uncomment for debugging
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['advancement_contribution'] = value
                        i += 1
                        continue
                    if 'teacher_0 - ZPD Contribution' in step_line:
                        print("B"*10) # Uncomment for debugging
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['zpd_contribution'] = value
                        i += 1
                        continue    
                    if 'teacher_0 - Question Contribution' in step_line:
                        print("C"*10) # Uncomment for debugging
                        value = float(step_line.split(':')[1].strip())
                        step_data['reward_breakdown']['question_contribution'] = value
                        i += 1
                        continue

                # Rewards Calculated
                rewards_calc_match = re.match(r'Rewards calculated: (\{.*?\})', step_line)
                if rewards_calc_match:
                    try:
                        rewards_str = rewards_calc_match.group(1)
                        step_data['rewards_calculated'] = ast.literal_eval(rewards_str)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse rewards calculated string at line {i+1}: {step_line} - Error: {e}")
                        step_data['rewards_calculated'] = None
                    i += 1
                    continue

                # Terminated Status
                terminated_match = re.match(r'Terminated: (\{.*?\})', step_line)
                if terminated_match:
                    try:
                        term_str = terminated_match.group(1)
                        step_data['terminated'] = ast.literal_eval(term_str)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse terminated string at line {i+1}: {step_line} - Error: {e}")
                        step_data['terminated'] = None
                    i += 1
                    continue

                # Truncated Status
                truncated_match = re.match(r'Truncated: (\{.*?\})', step_line)
                if truncated_match:
                    try:
                        trunc_str = truncated_match.group(1)
                        step_data['truncated'] = ast.literal_eval(trunc_str)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse truncated string at line {i+1}: {step_line} - Error: {e}")
                        step_data['truncated'] = None
                    i += 1
                    continue

                # Student ZPD Alignment
                zpd_match = re.match(r'(student_\d+) - ZPD Alignment: (-?\d+)', step_line)
                if zpd_match:
                    student_id = zpd_match.group(1)
                    alignment = int(zpd_match.group(2))
                    if student_id in step_data["students"]:
                        step_data["students"][student_id]['zpd_alignment'] = alignment
                    i += 1
                    continue

                # Student Question Bloom Level (Actual/Final for the step - Only for 'Ask')
                q_bloom_match = re.match(r'(student_\d+) - Question Bloom Level: (\d+)', step_line)
                if q_bloom_match:
                    student_id = q_bloom_match.group(1)
                    level = int(q_bloom_match.group(2))
                    # This seems to duplicate 'estimated_question_bloom_level',
                    # but we'll store it as specified just in case.
                    if student_id in step_data["students"]:
                         step_data["students"][student_id]['question_bloom_level'] = level
                    i += 1
                    continue

                # Student Max Bloom Reached (Specific end-of-episode line)
                max_bloom_match = re.match(r'(student_\d+) - Max Bloom Level Reached at Step: (\d+)', step_line)
                if max_bloom_match:
                    student_id = max_bloom_match.group(1)
                    step_num = int(max_bloom_match.group(2))
                    # This info belongs to the episode level, but appears within the last step's log.
                    # We'll add it to the step for now, or could move it up later.
                    if student_id in step_data["students"]:
                         step_data["students"][student_id]['max_bloom_reached_at_step'] = step_num
                    i += 1
                    continue
                
                


                # If no match, just advance
                # print(f"  Skipping line {i+1}: {step_line}") # Uncomment for debugging unmatched lines
                i += 1
            # --- End of Step Data Parsing ---

            # Store the collected step data
            if iteration_key in parsed_data and episode_key in parsed_data[iteration_key]["episodes"]:
                parsed_data[iteration_key]["episodes"][episode_key]["steps"][step_key] = step_data
            # Don't increment i here, the outer loop handles it based on the break condition

        else:
            # If the line didn't match any major section, just move to the next line
            i += 1

    # --- Save to JSON ---
    try:
        with open(json_file_path, 'w', encoding='utf-8') as jf:
            json.dump(parsed_data, jf, indent=4)
        print(f"\nSuccessfully parsed log and saved to {json_file_path}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    log_file = 'log_base_experiment_1.txt'
    json_output_file = 'parsed_log_data.json'
    
    if len(sys.argv) != 3:
        print("Usage: python parse_metrics_log.py <log_file> <json_output_file>")
        sys.exit(1)
    else:
        log_file = sys.argv[1]
        json_output_file = sys.argv[2]

    parse_log_to_json(log_file, json_output_file)





# Plan:
# episode_base = 0
# read line by line

#    > if line contains "Starting Iteration" then read the value "Starting Iteration {num}/{total_num_iterations}" num and total_num_iterations
#        * set current_iteration = num
#        * episode_base = curr_episode_in_iteration (that way the next episode value you read immediately will be 1)
#    > if line contains "**********Episode " then the next character is the current episode number in TOTAL across all iterations
#        * curr_episode_in_iteration = that value - episode_base
#    > if line contains "-----Step " then the next character is the current step number in the episode
#        * dont need to do any offset subtraction for this - the step number is the current step number in the episode
#        > then in this step there's a certain set of lines that are there that i must read and parse and store maybe in a dictionary, and then
#          it'll be easier to iterate over the dictionary to calculate whatever metrics i need. maybe i can save the dictionary as a json file
#        > the next lines will contains:
#             Teacher Action: Complex
#             student_{i} Action: {Study or Ask} (Current Bloom: {num} - {one of the bloom scores})
#             student_{i} Level Advanced: {FALSE or TRUE}
#               ^ repeat for 3 students (i = 0, 1, and 2)
#               ^ but if the action is Ask then there is also the "Current Student Bloom Level Desc" like below
#                 Categorize that value to like "Applying knowledge" to like "Apply" or the basic bloom value
#                 and then it also has the estimated question bloom level
#             student_{i} Current Student Bloom Level Desc: Applying knowledge
#             student_{i}: Estimated Question Bloom Level and Description: 3 and Apply
            
#             Current Class Avg Bloom Level: 2.0 -> 2
#                ^ then current class average bloom level is there (unrounded) -> (rounded)
#             NEW Bloom levels update: {'student_0': 1, 'student_1': 2, 'student_2': 3} -> {'student_0': 1, 'student_1': 2, 'student_2': 3}. 
#                ^ shows for this step how the bloom levels updated for each student
#             Reward Breakdown: (in teacher rewards)
#                 teacher_0 - Advancement Contribution: -0.0
#                 teacher_0 - ZPD Contribution: 0.5454545454545454
#                 teacher_0 - Question Contribution: -0.0
#             Rewards calculated: {'student_2': 0.0, 'teacher_0': -0.36666666666666664, 'student_0': 0.0, 'student_1': 0.0}
#                 ^ calculated rewards for each student during this step
#             Terminated: {'student_2': False, 'teacher_0': False, 'student_0': False, 'student_1': False, '__all__': False}
#             Truncated: {'student_2': False, 'teacher_0': False, 'student_0': False, 'student_1': False, '__all__': False}
#                ^ the __all__ will tell us if this step causes the episode to end. False means it doesnt end, and True would 
#                mean it reached the goal if terminated __all__ is true, otherwise it means it got clipped / truncated early 
#                if truncated __all__ is True
#             student_0 - ZPD Alignment: -1
#             student_1 - ZPD Alignment: -1
#             student_2 - ZPD Alignment: 1
#             student_2 - Question Bloom Level: 3
#                 ^ if the student asked a question then they also have this Question bloom level with the value, but this is 
#                 just repeat from above so skip
