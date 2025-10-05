from PIL.ImageFile import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
import robodm
from robodm.ingestion import create_vla_dataset_from_source, PyTorchDatasetAdapter
import numpy as np
import time
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_delta_indices

class LerobotConverter:
    def __init__(self, dataset_name, delta_timestamps=None, fps=30):
        self.dataset_name = dataset_name
        self.dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")

        self.delta_timestamps = delta_timestamps or {
            "observation.image": [-0.1, 0.0],
            "observation.state": [-0.1, 0.0],
            "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        }
        self.fps = fps
        self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)


    def load_dataset(self):
        # Load dataset with temporal sequences
        print(f"Loading dataset '{self.dataset_name}'...")
        with tqdm(total=1, desc="Loading dataset", unit="dataset") as pbar:
            self.dataset = LeRobotDataset(self.dataset_name, delta_timestamps=self.delta_timestamps)
            pbar.update(1)
        print(f"Dataset loaded successfully! Size: {len(self.dataset)} items")
        return self.dataset

    def convert_to_roboDM(self, codec='auto', output_path="./tmp/roboDM_dataset.vla"):
        print(self.dataset)
        # hf_dataset = self.dataset["train"]
        print(f"Dataset size: {len(self.dataset)}")

        trajectory = robodm.Trajectory(path=output_path, mode="w", video_codec=codec)
        # print(self.dataset.features)

        for item in tqdm(self.dataset):
            for i in range(len(item['observation.image'])):
                trajectory.add(f'observation.image_{i}', item['observation.image'][i].numpy().transpose(1, 2, 0))
            trajectory.add('observation.state', item['observation.state'].numpy())
            trajectory.add('action', item['action'].numpy())
            trajectory.add('episode_index', item['episode_index'].numpy())
            trajectory.add('frame_index', item['frame_index'].numpy())
            trajectory.add('timestamp', item['timestamp'].numpy())

        trajectory.close()

        trajectory = robodm.Trajectory(path=output_path, mode='r')
        data = trajectory.load()
        # Reconstruct observation.image from individual frames
        image_keys = [k for k in list(data.keys()) if 'observation.image_' in k]
        if image_keys:
            # Stack frames: [N_frames, H, W, C] -> [N_frames, T, H, W, C] -> [N_frames, T, C, H, W]
            stacked_images = np.stack([data.pop(k) for k in sorted(image_keys)], axis=1)
            data['observation.image'] = stacked_images.transpose(0, 1, 4, 2, 3)
        print(f'time to load dataset: {time.time() - start_time:.2f} seconds')

        trajectory.close()

if __name__ == "__main__":
    start_time = time.time()
    dataset_name = "lerobot/pusht" # Modify this to the dataset you want to convert
    converter = LerobotConverter(dataset_name)
    
    load_start = time.time()
    dataset = converter.load_dataset()
    load_time = time.time() - load_start
    
    convert_start = time.time()
    roboDM_dataset = converter.convert_to_roboDM(codec='h264', output_path="./tmp/robot_demoNew25650Temporal1.vla")
    convert_time = time.time() - convert_start
    
    total_time = time.time() - start_time
    print(f"\nTiming Results:")
    print(f"Dataset loading: {load_time:.2f}s")
    print(f"Conversion: {convert_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")

