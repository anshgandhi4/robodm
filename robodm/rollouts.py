from pathlib import Path
import multiprocessing as mp
import time

import gym_pusht
import gymnasium as gym
import imageio
import numpy
import random
import torch
import wandb

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

device = 'cuda'

def run_single_rollout(rollout_id, pretrained_policy_path, device, max_episode_steps, output_directory, seed=None):
    try:
        # Load policy for this process
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
        policy.reset()

        # Initialize environment for this rollout
        env = gym.make(
            'gym_pusht/PushT-v0',
            obs_type='pixels_agent_pos',
            max_episode_steps=max_episode_steps,
        )

        # Reset environment with seed
        numpy_observation, info = env.reset(seed=seed)

        # Prepare to collect rewards and frames
        rewards = []
        frames = []

        # Render initial frame
        frames.append(env.render())

        step = 0
        done = False

        while not done:
            # Prepare observation for the policy
            state = torch.from_numpy(numpy_observation['agent_pos'])
            image = torch.from_numpy(numpy_observation['pixels'])

            # Convert to float32 with image from channel first in [0,255] to channel last in [0,1]
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)

            # Send data tensors to device
            state = state.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)

            # Add batch dimension
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            # Create observation dictionary
            observation = {
                'observation.state': state,
                'observation.image': image,
            }

            # Predict action
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Prepare action for environment
            numpy_action = action.squeeze(0).to('cpu').numpy()

            # Step environment
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)

            # Track rewards and frames
            rewards.append(reward)
            frames.append(env.render())

            wandb.log({
                f'eval/reward_{rollout_id}': reward
            })

            # Check if done
            done = terminated | truncated
            step += 1

            # Print progress for this rollout
            print(f'Rollout {rollout_id}: step={step} reward={reward} terminated={terminated}')

        # Close environment
        env.close()

        # Save video for this rollout
        fps = env.metadata['render_fps']
        video_path = output_directory / f'rollout_{rollout_id}.mp4'
        imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

        # Calculate results
        total_reward = sum(rewards)
        success = terminated

        return {
            'rollout_id': rollout_id,
            'success': success,
            'total_reward': total_reward,
            'steps': step,
            'video_path': str(video_path),
            'seed': seed
        }

    except Exception as e:
        print(f'Error in rollout {rollout_id}: {e}')
        return {
            'rollout_id': rollout_id,
            'success': False,
            'total_reward': 0,
            'steps': 0,
            'video_path': None,
            'seed': seed,
            'error': str(e)
        }

def run_parallel_rollouts(num_rollouts, max_episode_steps, codec, output_directory, iter=0, seeds=None):
    pretrained_policy_path = Path(f'outputs/train/wandb/{codec}')
    # pretrained_policy_path = 'lerobot/diffusion_pusht'

    if seeds is None:
        import random
        seeds = [random.randint(0, 10000) for _ in range(num_rollouts)]

    rollout_args = [
        (num_rollouts * iter + i, pretrained_policy_path, device, max_episode_steps, output_directory, seeds[num_rollouts * iter + i])
        for i in range(num_rollouts)
    ]

    print(f'Starting {num_rollouts} parallel rollouts...')
    print(f'Seeds: {seeds}')

    # run rollouts in parallel
    with mp.Pool(processes=num_rollouts) as pool:
        results = pool.starmap(run_single_rollout, rollout_args)

    print('\n')
    print('============================================================')
    print('=                      ROLLOUT SUMMARY                     =')
    print('============================================================')

    successful_rollouts = 0
    total_rewards = []

    for result in results:
        status = 'SUCCESS' if result['success'] else 'FAILURE'
        print(f'Rollout {result["rollout_id"]} (seed {result["seed"]}): {status}')
        print(f'  Steps: {result["steps"]}, Total Reward: {result["total_reward"]:.2f}')
        if result['video_path']:
            print(f'  Video: {result["video_path"]}')
        if 'error' in result:
            print(f'  Error: {result["error"]}')
        print()

        if result['success']:
            successful_rollouts += 1
        total_rewards.append(result['total_reward'])

    # print stats
    success_rate = successful_rollouts / num_rollouts * 100
    avg_reward = sum(total_rewards) / len(total_rewards)

    print(f'Overall Success Rate: {success_rate:.1f}% ({successful_rollouts}/{num_rollouts})')
    print(f'Average Total Reward: {avg_reward:.2f}')
    print(f'All videos saved in: {output_directory}')

def main(codec, iters, seeds):
    run = wandb.init(project='robodm-x-lerobot', name=f'eval-{codec}')

    num_rollouts = 10
    max_episode_steps = 300

    output_directory = Path(f'outputs/eval/wandb/{codec}')
    output_directory.mkdir(parents=True, exist_ok=True)

    # parallel rollouts
    start_time = time.time()
    for i in range(iters):
        run_parallel_rollouts(
            num_rollouts=num_rollouts,
            max_episode_steps=max_episode_steps,
            codec=codec,
            output_directory=output_directory,
            iter=i,
            seeds=seeds
        )
    print(f'Time taken to run rollouts: {time.time() - start_time:.2f} seconds')

    wandb.save(str(output_directory / '*'))
    run.alert('rollouts complete', text=f'{codec} finished rollouts')
    wandb.finish()

if __name__ == '__main__':
    for codec in ['auto', 'rawvideo', 'libaom-av1', 'libx264', 'libx265', 'ffv1']:
        main(codec, 10, [random.randint(0, 10000) for _ in range(100)])
