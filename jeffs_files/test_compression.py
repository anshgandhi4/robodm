import robodm
import time
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import os
import cv2


def main(codec='auto'):
    delta_timestamps = {
        'observation.image': [-0.1, 0.0],
        'observation.state': [-0.1, 0.0],
        'action': [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset = LeRobotDataset('lerobot/pusht', delta_timestamps=delta_timestamps)
    trajectory = robodm.Trajectory(path=f'./tmp/test_compression/robot_demo_{codec}.vla', mode='w', video_codec=codec)

    start_time = time.time()
    for item in tqdm(dataset):
        for i in range(len(item['observation.image'])):
            trajectory.add(f'observation.image_{i}', item['observation.image'][i].numpy().transpose(1, 2, 0))
            # save the image to a file
            cv2.imwrite(f'./tmp/test_compression/images/robot_demo_{codec}_{i}.png', item['observation.image'][i].numpy().transpose(1, 2, 0))
        trajectory.add('observation.state', item['observation.state'].numpy())
        trajectory.add('action', item['action'].numpy())

    print(f'time to save dataset as robodm: {time.time() - start_time:.2f} seconds')

    trajectory.close()

    # print the size of the trajectory
    print(f'size of trajectory: {os.path.getsize(f"./tmp/test_compression/robot_demo_{codec}.vla")}')

if __name__ == '__main__':
    for codec in [
        # 'libx264', 'libx265', 
    # 'libaom-av1',
    #  'ffv1', 
     'rawvideo']:
        main(codec)