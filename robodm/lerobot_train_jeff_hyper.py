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

"""This script demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import robodm
from robodm.dataset import load_trajectory_dataset, load_slice_dataset
import time
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from robodm.lerobot.lerobot_trajectory_dataset import LeRobotRobodmDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from robodm.lerobot.lerobot_converter_single_images import LerobotConverter

import wandb
import weave

WANDB = True


@weave.op()
def train_with_compression_params(codec='auto', g=None, crf=None):
    """
    Training function that uses compression parameters g and crf.
    
    Args:
        codec: Video codec to use
        g: Group of pictures (GOP) size for video compression
        crf: Constant Rate Factor for compression quality (lower = better quality)
    """
    compression_config = {
        'g': g,  # GOP size
        'crf': crf,  # Constant Rate Factor
    }
    
    # ========================================
    # PLACEHOLDER: Apply compression parameters
    # ========================================
    # TODO: Use g and crf values here when loading/processing video data
    # For example, when creating the dataset or encoding videos:
    # - Pass g to set keyframe interval (GOP size)
    # - Pass crf to control compression quality/bitrate
    # Example usage might look like:
    #   dataset = create_dataset(codec=codec, g=g, crf=crf)
    #   or
    #   video_encoder.configure(gop_size=g, crf=crf)
    print(f"Using compression params: codec={codec}, g={g}, crf={crf}")
    # ========================================
    
    if WANDB:
        run = wandb.init(
            project='robodm-x-lerobot', 
            name=f'train-{codec}-g{g}-crf{crf}',
            config=compression_config,
            reinit=True
        )

    # Create a directory to store the training checkpoint.
    output_directory = Path(f'outputs/train/wandb/{codec}-g{g}-crf{crf}-100k-1e-4')
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 100000
    warmup_steps = 1000  # Warmup steps for learning rate scheduling
    log_freq = 100

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht_image")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(input_features)


    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(
        input_features=input_features, 
        output_features=output_features,
        num_train_timesteps=100,
        optimizer_lr=1e-4,
    )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

   
    # ========================================
    #   Dataset creation from single images
    # ========================================
  

    output_path = f"./tmp/single_imagesMasonWeaveFixPath/single_images_demo_{codec}_{g}_{crf}.vla"
    print(f"Converting {codec} and saving to {output_path}...")

    start_time = time.time()
    dataset_name = "lerobot/pusht_image" # Modify this to the dataset you want to convert
    converter = LerobotConverter(dataset_name) # this the import from the lerobot_converter_single_images.py file
    
    load_start = time.time()
    dataset = converter.load_dataset()
    load_time = time.time() - load_start
    
    convert_start = time.time()
    roboDM_dataset = converter.convert_to_roboDM(output_path=output_path, codec=codec, g=g, crf=crf)
    convert_time = time.time() - convert_start
    
    total_time = time.time() - start_time
    print(f"\nTiming Results:")
    print(f"Dataset loading: {load_time:.2f}s")
    print(f"Conversion: {convert_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")

    # ========================================


    dataset = LeRobotRobodmDataset(
        trajectory_path=output_path, 
        delta_timestamps=delta_timestamps, 
        dataset_metadata=dataset_metadata
    )
    
    print(f"Dataset length: {dataset.__len__()}")
    
    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)  # Use the same LR as in config
    
    # Create learning rate schedulers
    # Warmup scheduler: linearly increase LR from 0 to target LR over warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
    # Main scheduler: cosine annealing after warmup
    main_scheduler = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps, eta_min=1e-6)
    
    # Combined scheduler: warmup followed by cosine annealing
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps]
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,  # Enable shuffling for better training
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    print("Starting training loop...")
    start_time = time.time()
    epoch_start_time = time.time()
    running_loss = 0.0
    loss_count = 0
    
    while not done:
        print(f"Starting epoch, step {step}")
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            running_loss += loss.item()
            loss_count += 1

            if step % log_freq == 0:
                avg_loss = running_loss / loss_count

                if WANDB:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/epoch': step,
                        'train/time': time.time() - start_time,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'compression/g': g,
                        'compression/crf': crf,
                    })

                running_loss = 0.0
                loss_count = 0

                current_time = time.time()
                epoch_time = current_time - epoch_start_time
                steps_per_sec = (step + 1) / epoch_time if epoch_time > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                print(f"step: {step} loss: {loss.item():.3f} | lr: {current_lr:.2e} | {steps_per_sec:.2f} steps/sec")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)

    if WANDB:
        wandb.save(str(output_directory / '*'))
        run.alert('training complete', text=f'{codec} with g={g}, crf={crf} finished training')
        wandb.finish()
    
    # Benchmark results
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return {
        'final_loss': loss.item(),
        'total_time': total_time,
        'compression_g': g,
        'compression_crf': crf,
    }


def main(codec='auto'):
    if WANDB:
        run = wandb.init(project='robodm-x-lerobot', name=f'train-{codec}-mason')

    # Create a directory to store the training checkpoint.
    output_directory = Path(f'outputs/train/wandb/{codec}-100k-1e-4')
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 100000
    warmup_steps = 1000  # Warmup steps for learning rate scheduling
    log_freq = 100

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht_image")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(input_features)


    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(
        input_features=input_features, 
        output_features=output_features,
        num_train_timesteps=100,
        optimizer_lr=1e-4,
    )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    # dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    dataset = LeRobotRobodmDataset(trajectory_path=f"./data/single_images_demo_{codec}.vla", delta_timestamps=delta_timestamps, dataset_metadata=dataset_metadata)
    print(f"Dataset length: {dataset.__len__()}")
    
    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)  # Use the same LR as in config
    
    # Create learning rate schedulers
    # Warmup scheduler: linearly increase LR from 0 to target LR over warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
    # Main scheduler: cosine annealing after warmup
    main_scheduler = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps, eta_min=1e-6)
    
    # Combined scheduler: warmup followed by cosine annealing
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps]
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,  # Enable shuffling for better training
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    print("Starting training loop...")
    start_time = time.time()
    epoch_start_time = time.time()
    running_loss = 0.0
    loss_count = 0
    
    while not done:
        print(f"Starting epoch, step {step}")
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            running_loss += loss.item()
            loss_count += 1

            if step % log_freq == 0:
                avg_loss = running_loss / loss_count

                if WANDB:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/epoch': step,
                        'train/time': time.time() - start_time,
                        'train/learning_rate': scheduler.get_last_lr()[0]
                    })

                running_loss = 0.0
                loss_count = 0

                current_time = time.time()
                epoch_time = current_time - epoch_start_time
                steps_per_sec = (step + 1) / epoch_time if epoch_time > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                print(f"step: {step} loss: {loss.item():.3f} | lr: {current_lr:.2e} | {steps_per_sec:.2f} steps/sec")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)

    if WANDB:
        wandb.save(str(output_directory / '*'))
        run.alert('training complete', text=f'{codec} finished training')
        wandb.finish()
    
    # Benchmark results
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


def run_compression_sweep():
    """
    Run a weave sweep to test different compression parameters.
    Tests various combinations of codec, g (GOP size), and crf (quality) values.
    Only tests g/crf for codecs that support them.
    """
    # Initialize weave
    weave.init('robodm-compression-sweep')
    
    # Define codec capabilities
    codec_config = {
        # 'auto': {'supports_crf': False, 'supports_g': False},
        # 'rawvideo': {'supports_crf': False, 'supports_g': False},
        'libx264': {'supports_crf': True, 'supports_g': True},
        'libx265': {'supports_crf': True, 'supports_g': True},
        'ffv1': {'supports_crf': False, 'supports_g': False},
    }
    
    # Define parameter ranges to test
    g_values = [10, 30, 60]  # GOP sizes to test (keyframe intervals)
    crf_values = list(range(18, 29))  # CRF values to test (lower = better quality, larger file)
    
    # Build experiment list based on codec capabilities
    experiments = []
    for codec, capabilities in codec_config.items():
        if capabilities['supports_crf'] and capabilities['supports_g']:
            # Test all combinations of g and crf
            for g in g_values:
                for crf in crf_values:
                    experiments.append({'codec': codec, 'g': g, 'crf': crf})
        else:
            # Codec doesn't support g/crf, test with None values
            experiments.append({'codec': codec, 'g': None, 'crf': None})
    
    results = []
    
    print("=" * 80)
    print("Starting Weave Compression Parameter Sweep")
    print("=" * 80)
    print(f"Testing {len(codec_config)} codec(s):")
    for codec, caps in codec_config.items():
        support_str = []
        if caps['supports_g']:
            support_str.append(f"g={g_values}")
        if caps['supports_crf']:
            support_str.append(f"crf={list(crf_values)}")
        if not support_str:
            support_str.append("no params")
        print(f"  - {codec}: {', '.join(support_str)}")
    print(f"Total experiments: {len(experiments)}")
    print("=" * 80)
    
    # Run experiments
    for i, exp in enumerate(experiments, 1):
        codec = exp['codec']
        g = exp['g']
        crf = exp['crf']
        
        print(f"\n{'=' * 80}")
        print(f"Experiment {i}/{len(experiments)}: codec={codec}, g={g}, crf={crf}")
        print(f"{'=' * 80}\n")
        
        try:
            result = train_with_compression_params(codec=codec, g=g, crf=crf)
            results.append({
                'codec': codec,
                'g': g,
                'crf': crf,
                'status': 'success',
                **result
            })
            print(f"\n✓ Completed: codec={codec}, g={g}, crf={crf}")
        except Exception as e:
            print(f"\n✗ Failed: codec={codec}, g={g}, crf={crf}")
            print(f"Error: {str(e)}")
            results.append({
                'codec': codec,
                'g': g,
                'crf': crf,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    for result in results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_symbol} codec={result['codec']}, g={result['g']}, crf={result['crf']}: {result['status']}")
        if result['status'] == 'success':
            print(f"  └─ Loss: {result.get('final_loss', 'N/A'):.4f}, Time: {result.get('total_time', 0):.1f}s")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Choose which mode to run:
    # 1. Run compression parameter sweep with weave
    # 2. Run original main function with multiple codecs
    # 3. Run original main with single codec
    
    USE_WEAVE_SWEEP = True  # Set to True to use weave parameter sweep
    
    if USE_WEAVE_SWEEP:
        # Run the weave-based parameter sweep
        run_compression_sweep()
    elif WANDB:
        # Original behavior: iterate through codecs
        for codec in [
            'rawvideo', 'auto', 'libaom-av1', 'libx264', 'libx265', 
            # 'ffv1'
            ]:
            main(codec)
    else:
        # Run single training
        main()