from PIL.ImageFile import Image
from datasets import load_dataset
import matplotlib.pyplot as plt
import robodm
from robodm.ingestion import create_vla_dataset_from_source, PyTorchDatasetAdapter
import numpy as np
import time

class LerobotConverter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name)
        return self.dataset

    def convert_to_roboDM(self):
        print(self.dataset)
        hf_dataset = self.dataset["train"]
        print(f"Dataset size: {len(hf_dataset)}")

        trajectory = robodm.Trajectory(path="./tmp/robot_demoNew25650.vla", mode="w")
        print(hf_dataset.features)

        for item in hf_dataset.take(25650):
            print(item)
            image = item["observation.image"]
            image_array = np.array(image)

            trajectory.add("observation.image", image_array)
            trajectory.add("observation.state", item["observation.state"])
            trajectory.add("action", item["action"])
            trajectory.add("episode_index", item["episode_index"])
            trajectory.add("frame_index", item["frame_index"])
            trajectory.add("timestamp", item["timestamp"])
            trajectory.add("next.reward", item["next.reward"])
            trajectory.add("next.done", item["next.done"])
            trajectory.add("next.success", item["next.success"])
            trajectory.add("index", item["index"])
            trajectory.add("task_index", item["task_index"])

            action = item["action"]

        trajectory.close()

if __name__ == "__main__":
    start_time = time.time()
    dataset_name = "lerobot/pusht_image" # Modify this to the dataset you want to convert
    converter = LerobotConverter(dataset_name)
    
    load_start = time.time()
    dataset = converter.load_dataset()
    load_time = time.time() - load_start
    
    convert_start = time.time()
    roboDM_dataset = converter.convert_to_roboDM()
    convert_time = time.time() - convert_start
    
    total_time = time.time() - start_time
    print(f"\nTiming Results:")
    print(f"Dataset loading: {load_time:.2f}s")
    print(f"Conversion: {convert_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")

