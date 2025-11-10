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
from robodm.lerobot.lerobot_trajectory_dataset import LeRobotRobodmDataset as RoboDMTrajectoryDataset



def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_pusht_diffusion_bench")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    log_freq = 100
    
    # Benchmarking flag - set to True to enable detailed timing analysis
    enable_benchmarking = True

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht_image")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(input_features)

    time.sleep(10)


    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

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
    dataset = RoboDMTrajectoryDataset(trajectory_path="./tmp/temporal/temporal_demo_auto.vla", delta_timestamps=delta_timestamps, dataset_metadata=dataset_metadata)
    print(f"Dataset length: {dataset.__len__()}")

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    print("Starting training loop...")
    start_time = time.time()
    epoch_start_time = time.time()
    
    # Benchmarking variables
    if enable_benchmarking:
        batch_data_loading_time = 0.0
        batch_device_transfer_time = 0.0
        batch_forward_pass_time = 0.0
        batch_backward_pass_time = 0.0
        batch_optimizer_step_time = 0.0
        batch_optimizer_zero_grad_time = 0.0
        batch_step_time = 0.0
        batch_steps = 0
    
    last_iteration_time = time.time()  # Track time from previous iteration
    
    while not done:
        print(f"Starting epoch, step {step}")
        for batch_idx, batch in enumerate(dataloader):
            current_time = time.time()
            
            # Data loading time is the time between this iteration and the last
            data_loading_time = current_time - last_iteration_time
            last_iteration_time = current_time  # Update for next iteration
            
            step_start_time = time.time()
            
            # Device transfer timing
            device_transfer_start = time.time()
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            device_transfer_time = time.time() - device_transfer_start
            
            # Forward pass timing
            forward_start = time.time()
            loss, _ = policy.forward(batch)
            forward_time = time.time() - forward_start
            
            # Backward pass timing
            backward_start = time.time()
            loss.backward()
            backward_time = time.time() - backward_start
            
            # Optimizer step timing
            optimizer_step_start = time.time()
            optimizer.step()
            optimizer_step_time = time.time() - optimizer_step_start
            
            # Optimizer zero_grad timing
            zero_grad_start = time.time()
            optimizer.zero_grad()
            zero_grad_time = time.time() - zero_grad_start
            
            # Calculate step time as sum of components (excluding timing overhead)
            total_step_time_measured = (data_loading_time + device_transfer_time + 
                                      forward_time + backward_time + 
                                      optimizer_step_time + zero_grad_time)
            
            # Accumulate benchmarking data for this batch
            if enable_benchmarking:
                batch_data_loading_time += data_loading_time
                batch_device_transfer_time += device_transfer_time
                batch_forward_pass_time += forward_time
                batch_backward_pass_time += backward_time
                batch_optimizer_step_time += optimizer_step_time
                batch_optimizer_zero_grad_time += zero_grad_time
                batch_step_time += total_step_time_measured
                batch_steps += 1

            if step % log_freq == 0:
                current_time = time.time()
                epoch_time = current_time - epoch_start_time
                steps_per_sec = (step + 1) / epoch_time if epoch_time > 0 else 0
                
                if enable_benchmarking and batch_steps > 0:
                    # Calculate averages for this batch (last log_freq steps)
                    avg_data_loading = batch_data_loading_time / batch_steps
                    avg_device_transfer = batch_device_transfer_time / batch_steps
                    avg_forward_pass = batch_forward_pass_time / batch_steps
                    avg_backward_pass = batch_backward_pass_time / batch_steps
                    avg_optimizer_step = batch_optimizer_step_time / batch_steps
                    avg_zero_grad = batch_optimizer_zero_grad_time / batch_steps
                    avg_total_step = batch_step_time / batch_steps
                    
                    print(f"step: {step} loss: {loss.item():.3f} | {steps_per_sec:.2f} steps/sec")
                    print(f"  BENCHMARK (avg over last {batch_steps} steps):")
                    print(f"    Data Loading: {avg_data_loading*1000:.5f}ms ({avg_data_loading/avg_total_step*100:.1f}%)")
                    print(f"    Device Transfer: {avg_device_transfer*1000:.2f}ms ({avg_device_transfer/avg_total_step*100:.1f}%)")
                    print(f"    Forward Pass: {avg_forward_pass*1000:.2f}ms ({avg_forward_pass/avg_total_step*100:.1f}%)")
                    print(f"    Backward Pass: {avg_backward_pass*1000:.2f}ms ({avg_backward_pass/avg_total_step*100:.1f}%)")
                    print(f"    Optimizer Step: {avg_optimizer_step*1000:.2f}ms ({avg_optimizer_step/avg_total_step*100:.1f}%)")
                    print(f"    Zero Grad: {avg_zero_grad*1000:.2f}ms ({avg_zero_grad/avg_total_step*100:.1f}%)")
                    print(f"    Total Step: {avg_total_step*1000:.2f}ms")
                    
                    # Reset batch counters for next batch
                    batch_data_loading_time = 0.0
                    batch_device_transfer_time = 0.0
                    batch_forward_pass_time = 0.0
                    batch_backward_pass_time = 0.0
                    batch_optimizer_step_time = 0.0
                    batch_optimizer_zero_grad_time = 0.0
                    batch_step_time = 0.0
                    batch_steps = 0
                else:
                    print(f"step: {step} loss: {loss.item():.3f} | {steps_per_sec:.2f} steps/sec")
                    
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)
    
    # Benchmark results
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()