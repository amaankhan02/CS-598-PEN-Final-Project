# train.py
# Trains the Teacher and Student agents in the ClassroomEnv using RLlib/PPO.

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

from agents import StudentAgent, TeacherAgent
from config import DEFAULT_ENV_CONFIG, DEFAULT_TRAINING_CONFIG
from environment import ClassroomEnv


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

    checkpoint_dir = algo.save()
    algo.stop()
    ray.shutdown()

    print(f"Checkpoint saved in directory: {checkpoint_dir}. Training finished.")


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
        .rollouts(
            num_rollout_workers=1,  # Number of parallel workers for collecting samples
            rollout_fragment_length="auto",  # Let RLlib determine fragment length
        )
        # TODO: when we make our environment, obs and action space more complex, we need to make the policy network model more complex as well
        .training(
            gamma=0.99,
            lr=5e-5,
            lambda_=0.95,
            train_batch_size=512,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
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


def main():
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
    train(num_iterations=DEFAULT_TRAINING_CONFIG["num_iterations"], algo=algo)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        ray.shutdown()  # shutdown in case of an error
