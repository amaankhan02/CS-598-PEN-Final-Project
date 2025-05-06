# train.py
# Trains the Teacher and Student agents in the ClassroomEnv using RLlib/PPO.

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np
import os
import json
import sys

from agents import StudentAgent, TeacherAgent
from config import DEFAULT_ENV_CONFIG, DEFAULT_TRAINING_CONFIG, METRICS_DIR, LOG_FILE_NAME, LOG_DIR
from environment import ClassroomEnv


class ClassroomCallbacks(DefaultCallbacks):
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        print("INSIDE ON_EPISODE_END")
        # gather statistics at the end of each episode
        
        # get final avg bloom levels for each student
        final_bloom_levels = []
        student_ids = [agent_id for agent_id in episode.agent_rewards.keys() if agent_id.startswith("student")]

        print(f"student_ids: {student_ids}")
        print(f"episode.last_info_for: {episode.last_info_for}")
        
        
        for agent_id in student_ids:
            last_info = episode.last_info_for(agent_id)
            if last_info and "self_bloom_level" in last_info:
                final_bloom_levels.append(last_info["self_bloom_level"])
            else:
                # TODO: handle this case 
                print("-"*40 + "ERROR A" + "-"*40)
                print(f"No final bloom level for student {agent_id}")
                print(f"last_info: {last_info}")
                print("-"*40 + "ERROR A" + "-"*40)
                
            
        if final_bloom_levels:
            avg_final_bloom = np.mean(final_bloom_levels)
            episode.custom_metrics["avg_final_bloom"] = avg_final_bloom
        else:
            print("-"*40 + "ERROR B" + "-"*40)
            print(f"No final bloom levels for any students")
            print("-"*40 + "ERROR B" + "-"*40)
            
        # get avg question bloom level for the class and per student
        all_class_question_levels = []
        hist_data = episode.hist_data
        
        print(f"hist_data: {hist_data}")
        print(f"len(hist_data): {len(hist_data)}")
        print(f"hist_data keys: {hist_data.keys()}")
        
        for agent_id in student_ids:
            if agent_id in hist_data and 'question_bloom_level' in hist_data[agent_id]:
                print(f"hist_data[agent_id]['question_bloom_level']: {hist_data[agent_id]['question_bloom_level']}")
                all_class_question_levels.extend(hist_data[agent_id]['question_bloom_level'])
                student_avg_question_level = np.mean(hist_data[agent_id]['question_bloom_level'])
                episode.custom_metrics[f"{agent_id}_avg_question_bloom_level"] = student_avg_question_level
        if all_class_question_levels:
            avg_question_level = np.mean(all_class_question_levels)
            episode.custom_metrics["class_avg_question_bloom_level"] = avg_question_level
        
        print(f"episode.custom_metrics: {episode.custom_metrics}")
        print(f"episode.custom_metrics keys: {episode.custom_metrics.keys()}")
        
        # save episode.custom_metrics to file
        metrics_file_path = os.path.join(METRICS_DIR, f"metrics_episode_{episode.episode_id}.json")
        with open(metrics_file_path, "w") as f:
            print("We're on line 77 - saving episode.custom_metrics to file in ClassroomCallbacks")
            json.dump(episode.custom_metrics, f)
        
        print(f"Episode {episode.episode_id} completed. Saved metrics to {metrics_file_path}")
        
                
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
    # TODO: double check these, where can i get the loss (policy loss)
    reward_mean = result.get("env_runners", {}).get("episode_return_mean", "N/A")
    policy_reward_mean = result.get("env_runners", {}).get("policy_return_mean", {})
    print(f"Mean Episode Reward: {reward_mean}")
    if policy_reward_mean:
        print(
            f"Teacher Policy Mean Reward: {policy_reward_mean.get('teacher_policy', 'N/A')}"
        )
        print(
            f"Student Policy Mean Reward: {policy_reward_mean.get('student_policy', 'N/A')}"
        )


def train(num_iterations, algo):
    """Run the main training loop."""
    print(f"\nStarting training for {num_iterations} iterations...")

    for i in range(num_iterations):
        result = algo.train()  # one iteration of training

        # Print results using RLlib's pretty_print function
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        print(pretty_print(result))

        # You can access specific metrics like this:
        print_metrics(result)
        
        # save metrics to file
        metrics_file_path = os.path.join(METRICS_DIR, f"metrics_iter_{i+1}.json")
        env_runners = result.get("env_runners", {})
        if not env_runners:
            print("-"*40 + "ERROR C" + "-"*40)
            print(f"No env_runners found in result")
            print(f"result: {result}")
            print("-"*40 + "ERROR C" + "-"*40)
        else:
            if "custom_metrics" in env_runners:
                with open(metrics_file_path, "w") as f:
                    json.dump(env_runners.get("custom_metrics", {}), f)
            else:
                print("-"*40 + "ERROR D" + "-"*40)
                print(f"No custom_metrics found in env_runners")
                print(f"env_runners: {env_runners}")
                print("-"*40 + "ERROR D" + "-"*40)
        
        print(f"Metrics for iteration {i+1} saved to {metrics_file_path}")

    # checkpoint_dir = algo.save()
    algo.stop()
    ray.shutdown()

    # print(f"Checkpoint saved in directory: {checkpoint_dir}. Training finished.")


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
        # .callbacks(ClassroomCallbacks)
        .framework("torch")  # Or "tf2"
        .rollouts(
            num_rollout_workers=1,  # Number of parallel workers for collecting samples
            rollout_fragment_length="auto",  # Let RLlib determine fragment length
        )
        # TODO: when we make our environment, obs and action space more complex, we need to make the policy network model more complex as well
        .training(
            gamma=0.99,
            lr=DEFAULT_TRAINING_CONFIG["lr"],
            lambda_=0.95,
            train_batch_size=DEFAULT_TRAINING_CONFIG["train_batch_size"],
            sgd_minibatch_size=64,
            num_sgd_iter=DEFAULT_TRAINING_CONFIG["num_sgd_iter"],
            model={
                # "fcnet_hiddens": [64, 64],
                "fcnet_hiddens": [32, 32],
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
    # global LOG_FILE_NAME
    # create metrics directory if it doesn't exist
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if len(sys.argv) != 2:
        print("Usage: python train.py <num_iterations>")
        sys.exit(1)
    
    num_iterations = int(sys.argv[1])
    
    try:
        main(num_iterations)
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        ray.shutdown()  # shutdown in case of an error
