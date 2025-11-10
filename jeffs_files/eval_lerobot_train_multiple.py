# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import torch
from tqdm import tqdm
import imageio
import json

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Configuration
NUM_TESTS = 1000  # Number of tests to run - change this value as needed
BATCH_SIZE = 100  # Number of parallel environments to run simultaneously
THRESHOLD = 0.97

for experiment in [
    "auto",
    "rawvideo",
    "libaom-av1",
    "libx264",
    "libx265",
    # "ffv1",
]:
    # Create a directory to store the evaluation results
    output_directory = Path(f"outputs/eval/wandb/new_eval/{experiment}-100k-1e-4-overnight-saturday-97")
    output_directory.mkdir(parents=True, exist_ok=True)

    # create directory to store rollout videos
    video_directory = output_directory / "rollout_videos"
    video_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add LeRobot evaluation optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
    # pretrained_policy_path = "lerobot/diffusion_pusht"
    # OR a path to a local outputs/train folder.
    pretrained_policy_path = Path(f"outputs/train/wandb/{experiment}-100k-1e-4-overnight-saturday")

    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

    # Initialize batched evaluation environments for parallel processing
    env = gym.vector.make(
        "gym_pusht/PushT-v0",
        num_envs=BATCH_SIZE,
        obs_type="pixels_agent_pos",
        max_episode_steps=300)


    # env.unwrapped.success_threshold = 0.8

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    # print(policy.config.input_features)
    # print(env.observation_space)

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    # print(policy.config.output_features)
    # print(env.action_space)

    # Initialize lists to track results across all tests
    all_results = []
    all_rewards = []

    reward_history = {}

    # Calculate number of batches needed
    n_batches = NUM_TESTS // BATCH_SIZE + int((NUM_TESTS % BATCH_SIZE) != 0)

    # Run batched evaluation with progress bar
    progress_bar = tqdm(range(n_batches), desc="Evaluating", unit="batch")
    for batch_idx in progress_bar:
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        
        # Reset environments with different seeds for each environment in the batch
        seeds = [42 + batch_idx * BATCH_SIZE + i for i in range(BATCH_SIZE)]
        numpy_observation, info = env.reset(seed=seeds)

        # initialize video writers for this batch
        batch_video_writers = []
        batch_frames = []
        for i in range(BATCH_SIZE):
            episode_idx = batch_idx * BATCH_SIZE + i
            if episode_idx < NUM_TESTS:
                batch_video_writers.append(str(video_directory / f"rollout_{episode_idx}.mp4"))
                batch_frames.append([])
            else:
                batch_video_writers.append(None)
                batch_frames.append(None)

        # Prepare to collect rewards for all environments in batch
        batch_rewards = [[] for _ in range(BATCH_SIZE)]
        batch_dones = [False] * BATCH_SIZE
        batch_successes = [False] * BATCH_SIZE

        step = 0
        
        # Create progress bar for this batch
        episode_bar = tqdm(total=300, desc=f"Batch {batch_idx + 1}/{n_batches}", leave=False, unit="step")
        
        while not all(batch_dones):
            # Prepare observation for the policy running in Pytorch (batched)
            state = torch.from_numpy(numpy_observation["agent_pos"])
            image = torch.from_numpy(numpy_observation["pixels"])

            # Convert to float32 with image from channel first in [0,255]
            # to channel last in [0,1]
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(0, 3, 1, 2)  # (batch, channels, height, width)

            # Send data tensors from CPU to GPU with optimized transfers
            state = state.to(device, non_blocking=device.type == "cuda")
            image = image.to(device, non_blocking=device.type == "cuda")

            # Create the policy input dictionary (already batched)
            observation = {
                "observation.state": state,
                "observation.image": image,
            }

            # Predict the next action with respect to the current observation
            # Use autocast for mixed precision if available
            with torch.inference_mode(), torch.autocast(device_type=device.type):
                action = policy.select_action(observation)

            # Prepare the action for the environment (already batched)
            numpy_action = action.to("cpu").numpy()

            # Step through the environment and receive a new observation
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            
            # record frames for video
            for i in range(BATCH_SIZE):
                if batch_frames[i] is not None and not batch_dones[i]:
                    batch_frames[i].append(numpy_observation["pixels"][i])

            # Keep track of rewards for each environment in the batch
            for i in range(BATCH_SIZE):
                if not batch_dones[i]:
                    batch_rewards[i].append(reward[i])

                    episode_idx = batch_idx * BATCH_SIZE + i
                    if episode_idx < NUM_TESTS:
                        if episode_idx not in reward_history:
                            reward_history[episode_idx] = []
                        reward_history[episode_idx].append(reward[i])

                    if terminated[i] or truncated[i] or reward[i] >= THRESHOLD:
                        batch_dones[i] = True
                        batch_successes[i] = terminated[i] or reward[i] >= THRESHOLD

            # The rollout is considered done when all environments are done
            step += 1

            # Update episode progress bar
            episode_bar.update(1)
            running_rewards = [sum(rewards) for rewards in batch_rewards if rewards]
            avg_reward = sum(running_rewards) / len(running_rewards) if running_rewards else 0
            episode_bar.set_postfix({
                "avg_reward": f"{avg_reward:.2f}",
                "done": f"{sum(batch_dones)}/{BATCH_SIZE}",
                "successes": f"{sum(batch_successes)}"
            })

        # Close episode progress bar
        episode_bar.close()
        
        # save videos
        for i, (video_path, frames) in enumerate(zip(batch_video_writers, batch_frames)):
            if video_path is not None and frames is not None:
                imageio.mimsave(video_path, frames, fps=30)

        # Record results for all environments in this batch
        for i in range(BATCH_SIZE):
            if i < NUM_TESTS - batch_idx * BATCH_SIZE:  # Only count up to NUM_TESTS
                success = batch_successes[i]
                total_reward = sum(batch_rewards[i])
                all_results.append(success)
                all_rewards.append(total_reward)

        # Update main progress bar with current success rate
        current_success_rate = sum(all_results) / len(all_results) * 100 if all_results else 0
        progress_bar.set_postfix({
            "success_rate": f"{current_success_rate:.1f}%",
            "episodes": f"{len(all_results)}/{NUM_TESTS}",
            "batch_successes": f"{sum(batch_successes)}/{BATCH_SIZE}"
        })

    # Print summary statistics
    print(f"\n=== SUMMARY OF {NUM_TESTS} TESTS ===")
    success_count = sum(all_results)
    success_rate = success_count / NUM_TESTS * 100
    avg_reward = sum(all_rewards) / NUM_TESTS
    min_reward = min(all_rewards)
    max_reward = max(all_rewards)

    print(f"Success rate: {success_count}/{NUM_TESTS} ({success_rate:.1f}%)")
    print(f"Average total reward: {avg_reward:.2f}")
    print(f"Min total reward: {min_reward:.2f}")
    print(f"Max total reward: {max_reward:.2f}")
    print(f"Results saved in: {output_directory}")
    print(f"Rollout videos saved in: {video_directory}")

    # save reward history to file
    reward_maxes = dict()
    for episode_idx in reward_history:
        reward_maxes[episode_idx] = max(reward_history[episode_idx])

    reward_history_path = output_directory / "reward_history.json"
    with open(reward_history_path, 'w') as f:
        json.dump(reward_maxes, f, indent=2)
        json.dump(reward_history, f, indent=2)
    total_timesteps = sum(len(timesteps) for timesteps in reward_history.values())

    print(f"Reward history saved in: {reward_history_path}")
    print(f"Total episodes recorded: {len(reward_history)}")
    print(f"Total timesteps recorded: {total_timesteps}")

    # Close the environment
    env.close()
