from pathlib import Path

import numpy as np
import time
import torch
from tqdm import tqdm
import wandb
import robodm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

WANDB = True

def main(codec='auto'):
    if WANDB:
        run = wandb.init(project='robodm-x-lerobot', name=f'train-{codec}')

    output_directory = Path(f'outputs/train/wandb/{codec}')
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')

    training_steps = 100000
    log_freq = 1
    batch_size = 64
    lr = 1e-4

    dataset_metadata = LeRobotDatasetMetadata('lerobot/pusht')
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    delta_timestamps = {
        'observation.image': [-0.1, 0.0],
        'observation.state': [-0.1, 0.0],
        'action': [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    dataset = LeRobotDataset('lerobot/pusht', delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != 'cpu',
        drop_last=True,
    )

    #########################################################################
    # save lerobot dataset as robodm, then load robodm dataset              #
    #########################################################################

    trajectory = robodm.Trajectory(path=Path(f'model/{codec}/robot_demo.vla'), mode='w', video_codec=codec)

    start_time = time.time()
    for item in tqdm(dataset):
        for i in range(len(item['observation.image'])):
            trajectory.add(f'observation.image_{i}', item['observation.image'][i].numpy().transpose(1, 2, 0))
        trajectory.add('observation.state', item['observation.state'].numpy())
        trajectory.add('action', item['action'].numpy())

    print(f'time to save dataset as robodm: {time.time() - start_time:.2f} seconds')

    start_time = time.time()
    trajectory.close()
    print(f'time to close dataset: {time.time() - start_time:.2f} seconds')

    start_time = time.time()
    trajectory = robodm.Trajectory(path=Path(f'model/{codec}/robot_demo.vla'), mode='r')
    data = trajectory.load()
    data['observation.image'] = np.stack([data.pop(k) for k in list(data.keys()) if 'observation.image_' in k], axis=1).transpose(0, 1, 4, 2, 3)
    print(f'time to load dataset: {time.time() - start_time:.2f} seconds')

    # dataset2 = robodm.dataset.VLADataset.create_trajectory_dataset(path='/tmp/robodm/robot_demo.vla',
    #         config=robodm.dataset.DatasetConfig(ray_init_kwargs={'log_to_driver': False})).get_ray_dataset()
    # dataloader2 = dataset2.iter_torch_batches(batch_size=batch_size, drop_last=True)

    # rest of lerobot example code
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000)

    start_time = time.time()
    running_loss = 0.0
    loss_count = 0

    for step in range(training_steps):
        i = np.random.choice(len(data['observation.image']), size=batch_size, replace=False)
        batch = {}
        for k, v in data.items():
            # Get the slice of arrays
            arrays = v[i]

            # Find the maximum shape for padding
            max_shape = max(item.shape for item in arrays)

            # Pad arrays and create mask
            padded_arrays = []
            masks = []
            for item in arrays:
                item_tensor = torch.from_numpy(item)

                pad_width = [(0, max_dim - curr_dim) for curr_dim, max_dim in zip(item.shape, max_shape)]

                # Pad the tensor using torch.nn.functional.pad
                # torch.nn.functional.pad expects padding in reverse order (last dim first)
                pad_width_torch = [item for sublist in reversed(pad_width) for item in sublist]
                padded_arr = torch.nn.functional.pad(item_tensor, pad_width_torch, mode='constant', value=0)
                padded_arrays.append(padded_arr)

                mask = torch.ones(item.shape, dtype=torch.bool)
                mask = torch.nn.functional.pad(mask, pad_width_torch, mode='constant', value=False)
                masks.append(mask)

            # Convert to torch tensors
            batch[k] = torch.stack(padded_arrays).to(device)
            batch[f'{k}_is_pad'] = torch.stack(masks).to(device)

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
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                })
            else:
                print(f"step: {step} loss: {loss.item():.3f}")

            running_loss = 0.0
            loss_count = 0

    print(f'time taken to train: {time.time() - start_time:.2f} seconds')

    policy.save_pretrained(output_directory)

    if WANDB:
        wandb.save(str(output_directory / '*'))
        run.alert('training complete', text=f'{codec} finished training')
        wandb.finish()

if __name__ == '__main__':
    if WANDB:
        for codec in ['auto', 'rawvideo', 'libaom-av1', 'libx264', 'libx265', 'ffv1']:
            main(codec)
    else:
        main()
