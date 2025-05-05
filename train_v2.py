# train.py
# Trains the Teacher and Student agents in the ClassroomEnv using RLlib/PPO.

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.train import CheckpointConfig
import numpy as np
import sys
from ray.rllib.connectors.env_to_module import FlattenObservations

from agents import StudentAgent, TeacherAgent
from config import DEFAULT_ENV_CONFIG, DEFAULT_TRAINING_CONFIG
from environment import ClassroomEnv

# class ClassroomCallbacks(DefaultCallbacks):
    
#     def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
#         # gather statistics at the end of each episode
        
#         # # get final avg bloom levels for each student
#         # final_bloom_levels = []
#         # student_ids = [agent_id for agent_id in episode.agent_rewards.keys() if agent_id.startswith("student")]

#         # for agent_id in student_ids:
#         #     last_info = episode.last_info_for(agent_id)
#         #     if last_info and "self_bloom_level" in last_info:
#         #         final_bloom_levels.append(last_info["self_bloom_level"])
#         #     else:
#         #         # TODO: handle this case 
#         #         pass
            
#         # if final_bloom_levels:
#         #     avg_final_bloom = np.mean(final_bloom_levels)
#         #     episode.custom_metrics["avg_final_bloom"] = avg_final_bloom
            
#         # # get avg question bloom level for the class and per student
#         # all_class_question_levels = []
#         # hist_data = episode.hist_data
#         # for agent_id in student_ids:
#         #     if agent_id in hist_data and 'question_bloom_level' in hist_data[agent_id]:
#         #         all_class_question_levels.extend(hist_data[agent_id]['question_bloom_level'])
#         #         student_avg_question_level = np.mean(hist_data[agent_id]['question_bloom_level'])
#         #         episode.custom_metrics[f"{agent_id}_avg_question_bloom_level"] = student_avg_question_level
#         # if all_class_question_levels:
#         #     avg_question_level = np.mean(all_class_question_levels)
#         #     episode.custom_metrics["class_avg_question_bloom_level"] = avg_question_level
        
        
        
                
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
    # reward_mean = result.get("env_runners", {}).get("episode_return_mean", "N/A")
    # policy_reward_mean = result.get("env_runners", {}).get("policy_return_mean", {})
    reward_mean = result.get("episode_reward_mean", "N/A")
    policy_reward_mean = result.get("policy_reward_mean", {})
    
    print(f"Mean Episode Reward: {reward_mean}")
    if policy_reward_mean:
        print(f"Teacher Policy Mean Reward: {policy_reward_mean.get('teacher_policy', 'N/A')}")
        print(f"Student Policy Mean Reward: {policy_reward_mean.get('student_policy', 'N/A')}")


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
        .framework("torch")  # Or "tf2"
        .env_runners(
            # num_rollout_workers=1,  # Number of parallel workers for collecting samples
            num_env_runners=1,
            rollout_fragment_length="auto", 
            env_to_module_connector=lambda env: FlattenObservations(), # Let RLlib determine fragment length
        )
        .training(
            gamma=0.99,
            lr=5e-5,
            lambda_=0.95,
            train_batch_size_per_learner=DEFAULT_TRAINING_CONFIG["train_batch_size"],   # * changed from train_batch_size
            minibatch_size=DEFAULT_TRAINING_CONFIG["sgd_minibatch_size"],
            num_sgd_iter=DEFAULT_TRAINING_CONFIG["num_sgd_iter"],
            # model={
            #     "fcnet_hiddens": [64, 64],
            # },
        )
    # * moved policy model to rl_module
        .rl_module(
            model_config={
                "fcnet_hiddens": DEFAULT_TRAINING_CONFIG["fcnet_hiddens"],
                "encoder_configs": {
            "default": {
                "type": "categorical",
                "vocab_size": 6,        # Discrete(6) → vocab size 6
                "embed_dim": 32,        # small embedding size
            },
            "teacher_policy": {
                "type": "mlp",
                "hidden_layers": [32, 32],
            },
            "student_policy": {
                "type": "categorical",
                "vocab_size": 6,
                "embed_dim": 32,
            },
        
                            
            }},
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
                lambda agent_id, episode, **kwargs: (
                    "teacher_policy" if "teacher" in agent_id else "student_policy"
                )
            ),
            policies_to_train=["teacher_policy", "student_policy"],
        )
        # .resources(num_cpus=1)
        # For debugging or simpler setups:
        .debugging(
            log_level="INFO",
            # checkpoint_config=CheckpointConfig(
            #     num_to_keep=0,
            #     checkpoint_frequency=0
            # ),
            # min_time_s_per_iteration=0, # * added for debugging
            # min_sample_timesteps_per_iteration=100,  # * added for debugging
            # TODO: add tensorboard logger, below, then run tensorboard --logdir=./tensorboard_logs
            # logger_config={
            #     "type": "ray.tune.logger.TBXLogger",
            #     "logdir": "./tensorboard_logs",
            # }
        )  # Set log level (INFO, DEBUG, WARN, ERROR)
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
    if len(sys.argv) != 2:
        print("Usage: python train_v2.py <num_iterations>")
        sys.exit(1)
    
    num_iterations = int(sys.argv[1])

    try:
        main(num_iterations)
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        ray.shutdown()  # shutdown in case of an error
