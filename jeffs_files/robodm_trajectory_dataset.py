"""Custom PyTorch Dataset for RoboDM trajectories."""

from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import robodm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_delta_indices, get_episode_data_index
import matplotlib.pyplot as plt

class RoboDMTrajectoryDataset(Dataset):

    def __init__(self, trajectory_path: str, delta_timestamps: dict, dataset_metadata: LeRobotDatasetMetadata, return_type: str = "numpy"):
        self.trajectory_path = trajectory_path
        # Load trajectory data
        traj = robodm.Trajectory(self.trajectory_path, mode="r")
        self.data = traj.load(return_type=return_type)

        self.delta_timestamps = delta_timestamps
        self.return_type = return_type
        self.dataset_metadata = dataset_metadata
        self.fps = dataset_metadata.fps
        self.delta_indices = get_delta_indices(delta_timestamps, dataset_metadata.fps)
        self.episodes = np.unique(self.data["episode_index"])
        self.episode_data_index = get_episode_data_index(dataset_metadata.episodes, self.episodes)

        print("Episode data index:", self.episode_data_index)
        print("Delta indices:", self.delta_indices)

    def __len__(self):
        return len(self.data["frame_index"])

    def __getitem__(self, idx):
        item = {}

        # Get episode index for current sample
        ep_idx = self.data["episode_index"][idx]
        
        # Get query indices and padding for temporal sequences
        query_indices, padding = self._get_query_indices(idx, ep_idx)

        # Extract data using query indices and convert to float32 tensors
        item["observation.image"] = torch.from_numpy(
            self.data["observation.image"][query_indices["observation.image"]]
        ).float()  # Convert to float32
        item["observation.state"] = torch.from_numpy(
            self.data["observation.state"][query_indices["observation.state"]]
        ).float()  # Convert to float32
        item["action"] = torch.from_numpy(
            self.data["action"][query_indices["action"]]
        ).float()  # Convert to float32

        item["observation.image"] = item["observation.image"].permute(0, 3, 1, 2) # T, H, W, C -> T, C, H, W
        item["observation.image"] = item["observation.image"] / 255.0 # normalize the image
        
        # Add padding information
        item.update(padding)
        
        # Add metadata
        item["episode_index"] = self.data["episode_index"][idx]
        item["frame_index"] = self.data["frame_index"][idx]
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
    robodmDataset = RoboDMTrajectoryDataset(trajectory_path="./tmp/robot_demoNew25650.vla", delta_timestamps=delta_timestamps, dataset_metadata=dataset_metadata)
    lerobotDataset = LeRobotDataset("lerobot/pusht_image", delta_timestamps=delta_timestamps)


    print("len(robodmDataset):", len(robodmDataset))
    print("len(lerobotDataset):", len(lerobotDataset))
    print("Running tests...")

    # compare the two datasets
    from tqdm import tqdm
    for i in tqdm(range(len(robodmDataset)), desc="Comparing datasets"):
        # extract images from both datasets
        img = robodmDataset[i]["observation.image"][0].transpose(1, 2, 0)
        img1 = lerobotDataset[i]["observation.image"][0].numpy().transpose(1, 2, 0)
        
        # convert to float64 for precise comparison
        img_float = img.astype(np.float64)
        img1_float = img1.astype(np.float64)
        
        # calculate difference metrics
        abs_diff = np.abs(img_float - img1_float)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        assert max_abs_diff < 1e-6
        
        # verify other data
        assert robodmDataset[i]["observation.image"].shape == lerobotDataset[i]["observation.image"].shape
        assert robodmDataset[i]["observation.state"].shape == lerobotDataset[i]["observation.state"].shape
        assert torch.all(robodmDataset[i]["observation.state"] == lerobotDataset[i]["observation.state"])
        assert robodmDataset[i]["action"].shape == lerobotDataset[i]["action"].shape
        assert torch.all(robodmDataset[i]["action"] == lerobotDataset[i]["action"])
        assert robodmDataset[i]["episode_index"] == lerobotDataset[i]["episode_index"]
        assert robodmDataset[i]["frame_index"] == lerobotDataset[i]["frame_index"]
        assert robodmDataset[i]["timestamp"] == lerobotDataset[i]["timestamp"]

    print("RoboDM converted Dataset and original LeRobot Dataset are the same")
