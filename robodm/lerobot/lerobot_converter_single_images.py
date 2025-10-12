from datasets import load_dataset
import robodm
import numpy as np
import time
from tqdm import tqdm

class LerobotConverter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name)
        return self.dataset

    def convert_to_roboDM(self, output_path, codec):
        print(self.dataset)
        hf_dataset = self.dataset["train"]
        print(f"Dataset size: {len(hf_dataset)}")

        trajectory = robodm.Trajectory(path=output_path, mode="w", video_codec=codec)
        print(hf_dataset.features)

        for item in tqdm(hf_dataset.take(25650)):
            # print(item)
            image = item["observation.image"]
            image_array = np.array(image)
            # print(image_array.shape)

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
    for codec in [
        # 'auto',
        # 'rawvideo',
        'libaom-av1',
        # 'libx264',
        # 'libx265',
        # 'ffv1',
    ]:

        output_path = f"./tmp/single_imagesNew/single_images_demo_{codec}.vla"
        print(f"Converting {codec} and saving to {output_path}...")

        start_time = time.time()
        dataset_name = "lerobot/pusht_image" # Modify this to the dataset you want to convert
        converter = LerobotConverter(dataset_name)
        
        load_start = time.time()
        dataset = converter.load_dataset()
        load_time = time.time() - load_start
        
        convert_start = time.time()
        roboDM_dataset = converter.convert_to_roboDM(output_path=output_path, codec=codec)
        convert_time = time.time() - convert_start
        
        total_time = time.time() - start_time
        print(f"\nTiming Results:")
        print(f"Dataset loading: {load_time:.2f}s")
        print(f"Conversion: {convert_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

