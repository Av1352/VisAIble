import numpy as np
from PIL import Image
import os
from scipy.stats import mannwhitneyu


def compute_image_metrics(directory, base_name):
    """
    Compute intersection, union, and IoU for two binary grayscale images.

    Parameters:
    - directory (str): Path to the directory containing the images.
    - base_name (str): Base name of the image files (e.g., 'fake_image_1').

    Returns:
    - tuple: (intersection_area, union_area, iou)
        - intersection_area (int): Number of pixels where both images have value 1.
        - union_area (int): Number of pixels where at least one image has value 1.
        - iou (float): Intersection over Union (intersection_area / union_area).
    """
    # Construct file paths
    gt_path = os.path.join(directory, f"{base_name}_gt.png")
    map_path = os.path.join(directory, f"{base_name}_map_gcam.png")

    # Load images
    image1 = np.array(Image.open(gt_path).convert('L'))  # Ground truth
    image2 = np.array(Image.open(map_path).convert('L'))  # Map (e.g., Grad-CAM)

    # Ensure binary (0 or 1)
    image1 = (image1 > 0).astype(np.uint8)  # Threshold to 0/1
    image2 = (image2 > 0).astype(np.uint8)

    # Intersection
    intersection = image1 & image2
    intersection_area = np.sum(intersection)

    # Union
    union = image1 | image2
    union_area = np.sum(union)

    # IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def compute_iou_array(directory, start_idx=1, end_idx=10):
    """
    Compute IoU for a range of image pairs and store results in an array.

    Parameters:
    - directory (str): Path to the directory containing the images.
    - start_idx (int): Starting index for base names (default: 1).
    - end_idx (int): Ending index for base names (default: 10).

    Returns:
    - np.ndarray: Array of IoU values for each image pair.
    """
    iou_values = []
    for i in range(start_idx, end_idx + 1):
        base_name = f"fake_image_{i}"
        iou = compute_image_metrics(directory, base_name)
        iou_values.append(iou)

    return np.array(iou_values)

# Image dimensions (256x256)
image_width, image_height = 256, 256
# Number of simulations
num_simulations = 100

# Function to generate a fixed ground truth mask (about 10% of the image)
def generate_fixed_ground_truth_mask(image_width, image_height, mask_area_percentage=0.1):
    # Calculate the mask area (10% of the image)
    mask_area = int(mask_area_percentage * image_width * image_height)

    # Try to generate a fixed mask that covers around 10% of the image
    width = int(np.sqrt(mask_area))  # Using a square shape for simplicity
    height = mask_area // width  # Adjust the height accordingly

    # Randomly place the mask within the image
    x1 = np.random.randint(0, image_width - width)
    y1 = np.random.randint(0, image_height - height)
    x2 = x1 + width
    y2 = y1 + height

    return (x1, y1, x2, y2)


# Function to generate a random mask (bounding box) on a 256x256 image (about 10% of the area)
def generate_random_mask(image_width, image_height, mask_area_percentage=0.1):
    mask_area = int(mask_area_percentage * image_width * image_height)

    # Randomly generate bounding box coordinates for random mask
    width = int(np.sqrt(mask_area))  # Square shape for simplicity
    height = mask_area // width  # Adjust height to match the desired area

    x1 = np.random.randint(0, image_width - width)
    y1 = np.random.randint(0, image_height - height)
    x2 = x1 + width
    y2 = y1 + height

    return (x1, y1, x2, y2)


# Function to calculate IoU (Intersection over Union)
def compute_iou(true_mask, random_mask):
    x1, y1, x2, y2 = true_mask
    rx1, ry1, rx2, ry2 = random_mask

    # Calculate intersection area
    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Area of each mask
    true_area = (x2 - x1) * (y2 - y1)
    random_area = (rx2 - rx1) * (ry2 - ry1)

    # IoU
    union_area = true_area + random_area - inter_area
    iou = inter_area / union_area
    return iou


# Simulate random IoUs for 100 simulations
ious = []
for _ in range(num_simulations):
    true_mask = generate_fixed_ground_truth_mask(image_width, image_height,
                                                 mask_area_percentage=0.1)  # Fixed ground truth
    random_mask = generate_random_mask(image_width, image_height, mask_area_percentage=0.1)  # Random predicted mask
    iou = compute_iou(true_mask, random_mask)
    ious.append(iou)

random_ious = np.array(ious)

directory = "dd/S3/Mask Images Meta"
observed_ious = compute_iou_array(directory)
# IOU between gradcam and ground truth


# Perform a Mann-Whitney U test to compare the observed IoUs against the null distribution
stat, p_value = mannwhitneyu(observed_ious, random_ious, alternative='greater')

# Print results
print(f"Mann-Whitney U test statistic: {stat}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("The observed IoU is significantly better than random.")
else:
    print("There is no significant difference between the observed IoU and random.")

