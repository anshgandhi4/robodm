from robodm.lerobot.lerobot_trajectory_dataset import LeRobotRobodmDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np

delta_timestamps = {
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}
dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht_image")


def main(codec, vmin=0, vmax=1):
    lerobotDataset = LeRobotDataset("lerobot/pusht_image", delta_timestamps=delta_timestamps)
    robodmDataset = LeRobotRobodmDataset(trajectory_path=f"./tmp/single_images_ansh_good/single_images_demo_{codec}.vla", delta_timestamps=delta_timestamps, dataset_metadata=dataset_metadata)
    print(f"len(robodmDataset): {len(robodmDataset)}")

    IMAGE_INDEX = 0
    image = robodmDataset[IMAGE_INDEX]["observation.image"][0]
    image = image.numpy().transpose(1, 2, 0)

    lerobot_image = lerobotDataset[IMAGE_INDEX]["observation.image"][0]
    lerobot_image = lerobot_image.numpy().transpose(1, 2, 0)

    # Normalize images to [0, 1] for consistent comparison
    image_norm = image / 255.0
    lerobot_image_norm = lerobot_image / 255.0
    
    # Calculate difference image to highlight compression artifacts
    diff_image = np.abs(image_norm - lerobot_image_norm)
    
    # Create focused visualization with only 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Compression Artifacts Analysis - Codec: {codec}', fontsize=16)
    
    # Original LeRobot image
    axes[0].imshow(lerobot_image)
    axes[0].set_title("Original LeRobot Image")
    axes[0].axis('off')
    
    # Compressed RoboDM image
    axes[1].imshow(image)
    axes[1].set_title(f"Compressed RoboDM Image ({codec})")
    axes[1].axis('off')
    
    # Error heatmap with consistent scale
    error_map = np.mean(diff_image, axis=2)  # Convert to grayscale error
    im = axes[2].imshow(error_map, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title("Pixel-wise Error Map")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

    # save the figure
    plt.savefig(f"./tmp/single_images_ansh_good/compression_diagram_{codec}.png")
    
    # Print compression statistics
    mse = np.mean((image_norm - lerobot_image_norm) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    max_error = np.max(diff_image)
    mean_error = np.mean(diff_image)
    
    print(f"\nCompression Statistics for {codec}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Max Pixel Error: {max_error:.6f}")
    print(f"  Mean Pixel Error: {mean_error:.6f}")
    print("-" * 50)



if __name__ == "__main__":
    # Define consistent scale for all error maps (0 to 0.1 for better visualization)
    vmin, vmax = 0, 0.0008
    
    for codec in [
        'auto', 
    'rawvideo', 
    'libaom-av1', 'libx264', 'libx265', 
    # 'ffv1'
    ]:
        main(codec, vmin=vmin, vmax=vmax)