"""Custom PyTorch Dataset for RoboDM trajectories."""

import numpy as np
import torch
import robodm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_delta_indices, get_episode_data_index

class LeRobotRobodmDataset(torch.utils.data.Dataset):
    def __init__(self, trajectory_path: str, delta_timestamps: dict, dataset_metadata: LeRobotDatasetMetadata, return_type: str = "numpy"):
        self.trajectory_path = trajectory_path
        # Load trajectory data
        traj = robodm.Trajectory(self.trajectory_path, mode="r")
        self.data = traj.load(return_type=return_type)

        self.delta_timestamps = delta_timestamps
        self.return_type = return_type
        self.dataset_metadata = dataset_metadata

        image_keys = [k for k in list(self.data.keys()) if 'observation.image_' in k and not k.endswith('_is_pad')]
        if image_keys:
            # Stack frames: [N_frames, H, W, C] -> [N_frames, T, H, W, C]
            # Keep the original format [N_frames, T, H, W, C] - don't transpose channels yet
            stacked_images = np.stack([self.data.pop(k) for k in sorted(image_keys)], axis=1)
            self.data['observation.image'] = stacked_images
        self.fps = dataset_metadata.fps
        self.delta_indices = get_delta_indices(delta_timestamps, dataset_metadata.fps)
        if "episode_index" in self.data:
            self.episodes = np.unique(self.data["episode_index"])
        else:
            self.episodes = np.array([0])
        self.episode_data_index = get_episode_data_index(dataset_metadata.episodes, self.episodes)

        print("Episode data index:", self.episode_data_index)
        print("Delta indices:", self.delta_indices)

    def __len__(self):
        if "frame_index" in self.data:
            return len(self.data["frame_index"])
        else:
            return len(self.data["observation.image"])

    def __getitem__(self, idx):
        item = {}

        # Get episode index for current sample
        ep_idx = self.data["episode_index"][idx]            
        
        # Get query indices and padding for temporal sequences
        query_indices, padding = self._get_query_indices(idx, ep_idx)

        # print(self.data["observation.image"].ndim)
        # print(self.data["observation.image"].shape)

        if self.data["observation.image"][idx].ndim == 3:

        # Extract data using query indices and convert to float32 tensors
        # Note: self.data["observation.image"] is already a temporal sequence [T, H, W, C]
        # We need to index the temporal dimension, not the frame dimension
            item["observation.image"] = torch.from_numpy(
                self.data["observation.image"][query_indices["observation.image"]]  # Get the temporal sequence for this frame
            ).float()
            
            item["observation.state"] = torch.from_numpy(
                self.data["observation.state"][query_indices["observation.state"]]  # Get the temporal sequence for this frame
            ).float()
            
            item["action"] = torch.from_numpy(
                self.data["action"][query_indices["action"]]  # Get the temporal sequence for this frame
            ).float()

        else:
            item["observation.image"] = self.data["observation.image"][idx]  # Get the temporal sequence for this frame
            # item["observation.image"] = item["observation.image"].transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            # item["observation.image"] = item["observation.image"] / 255.0 # normalize the image
            
            # Add state and action data if available
            if "observation.state" in self.data:
                item["observation.state"] = self.data["observation.state"][idx]
            if "action" in self.data:
                item["action"] = self.data["action"][idx]

        # Handle both PyTorch tensors and NumPy arrays for image permutation
        if hasattr(item["observation.image"], 'permute'):
            # PyTorch tensor
            item["observation.image"] = item["observation.image"].permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        else:
            # NumPy array
            item["observation.image"] = torch.from_numpy(item["observation.image"]).permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        item["observation.image"] = item["observation.image"] / 255.0 # normalize the image
        
        # Add padding information - use saved padding flags if available, otherwise compute them
        if 'observation.state_is_pad' in self.data:
            item['observation.state_is_pad'] = torch.from_numpy(self.data['observation.state_is_pad'][idx]).bool()
        if 'observation.image_is_pad' in self.data:
            item['observation.image_is_pad'] = torch.from_numpy(self.data['observation.image_is_pad'][idx]).bool()
        if 'action_is_pad' in self.data:
            item['action_is_pad'] = torch.from_numpy(self.data['action_is_pad'][idx]).bool()
        
        # # Fallback to computed padding if not available in saved data
        if not any(key in item for key in ['observation.state_is_pad', 'observation.image_is_pad', 'action_is_pad']):
            item.update(padding)

        # Add metadata
        item["episode_index"] = self.data["episode_index"][idx]
        item["frame_index"] = self.data["frame_index"][idx]
        if "timestamp" in self.data:
            item["timestamp"] = self.data["timestamp"][idx]

        return item

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        # Convert ep_idx to int if it's a tensor/array
        if hasattr(ep_idx, 'item'):
            ep_idx = ep_idx.item()
        elif isinstance(ep_idx, np.ndarray):
            ep_idx = int(ep_idx)
        
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        
        # Convert to int if needed
        if hasattr(ep_start, 'item'):
            ep_start = ep_start.item()
        if hasattr(ep_end, 'item'):
            ep_end = ep_end.item()
            
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

# Test the dataset
if __name__ == "__main__":
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
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht_image")
    robodmDataset = LeRobotRobodmDataset(trajectory_path="./tmp/robot_demoNew25650Temporal1Compressed.vla", delta_timestamps=delta_timestamps, dataset_metadata=dataset_metadata)
    lerobotDataset = LeRobotDataset("lerobot/pusht_image", delta_timestamps=delta_timestamps)


    print("len(robodmDataset):", len(robodmDataset))
    print("len(lerobotDataset):", len(lerobotDataset))
    print("Running tests...")

    # compare the two datasets
    from tqdm import tqdm
    maxVal = 10
    for i in tqdm(range(min(len(robodmDataset), maxVal)), desc="Comparing datasets"):
        # extract images from both datasets
        robodm_img_raw = robodmDataset[i]["observation.image"][0]
        lerobot_img_raw = lerobotDataset[i]["observation.image"][0]
        
        # Debug: print shapes before transpose
        robodm_shape = robodm_img_raw.numpy().shape
        lerobot_shape = lerobot_img_raw.numpy().shape
        print(f"RoboDM image shape: {robodm_shape}")
        print(f"LeRobot image shape: {lerobot_shape}")
        
        # Handle different possible shapes
        if len(robodm_shape) == 3 and robodm_shape[0] == 3:
            img = robodm_img_raw.numpy().transpose(1, 2, 0)  # (3, 96, 96) -> (96, 96, 3)
        elif len(robodm_shape) == 3 and robodm_shape[-1] == 3:
            img = robodm_img_raw.numpy()  # Already (96, 96, 3)
        else:
            print(f"Unexpected RoboDM image shape: {robodm_shape}")
            img = robodm_img_raw.numpy()
            
        if len(lerobot_shape) == 3 and lerobot_shape[0] == 3:
            img1 = lerobot_img_raw.numpy().transpose(1, 2, 0)  # (3, 96, 96) -> (96, 96, 3)
        elif len(lerobot_shape) == 3 and lerobot_shape[-1] == 3:
            img1 = lerobot_img_raw.numpy()  # Already (96, 96, 3)
        else:
            print(f"Unexpected LeRobot image shape: {lerobot_shape}")
            img1 = lerobot_img_raw.numpy()
                
        # convert to float64 for precise comparison
        img_float = img.astype(np.float64)
        img1_float = img1.astype(np.float64)
        
        # calculate difference metrics
        abs_diff = np.abs(img_float - img1_float)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        assert mean_abs_diff < 0.01 # Allow for compression artifacts
        
        # verify other data
        assert robodmDataset[i]["observation.image"].shape == lerobotDataset[i]["observation.image"].shape
        
        # Only verify state and action if they exist in both datasets
        if "observation.state" in robodmDataset[i] and "observation.state" in lerobotDataset[i]:
            assert robodmDataset[i]["observation.state"].shape == lerobotDataset[i]["observation.state"].shape
            assert torch.all(robodmDataset[i]["observation.state"] == lerobotDataset[i]["observation.state"])
        
        if "action" in robodmDataset[i] and "action" in lerobotDataset[i]:
            assert robodmDataset[i]["action"].shape == lerobotDataset[i]["action"].shape
            assert torch.all(robodmDataset[i]["action"] == lerobotDataset[i]["action"])
        
        # Verify metadata
        assert robodmDataset[i]["episode_index"] == lerobotDataset[i]["episode_index"]
        assert robodmDataset[i]["frame_index"] == lerobotDataset[i]["frame_index"]
        assert robodmDataset[i]["timestamp"] == lerobotDataset[i]["timestamp"]
        
        # Verify padding flags if they exist in both datasets
        if "observation.state_is_pad" in robodmDataset[i] and "observation.state_is_pad" in lerobotDataset[i]:
            assert torch.all(robodmDataset[i]["observation.state_is_pad"] == lerobotDataset[i]["observation.state_is_pad"])
        if "observation.image_is_pad" in robodmDataset[i] and "observation.image_is_pad" in lerobotDataset[i]:
            assert torch.all(robodmDataset[i]["observation.image_is_pad"] == lerobotDataset[i]["observation.image_is_pad"])
        if "action_is_pad" in robodmDataset[i] and "action_is_pad" in lerobotDataset[i]:
            assert torch.all(robodmDataset[i]["action_is_pad"] == lerobotDataset[i]["action_is_pad"])

    print("RoboDM converted Dataset and original LeRobot Dataset are the same")
