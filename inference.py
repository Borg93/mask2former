import random

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Grab the trained model and processor from the hub
model = Mask2FormerForUniversalSegmentation.from_pretrained("adirik/maskformer-swin-base-sceneparse-instance").to(device)
processor = Mask2FormerImageProcessor.from_pretrained("adirik/maskformer-swin-base-sceneparse-instance")


# Use random test image
index = random.randint(0, len(test))
image = test[index]["image"].convert("RGB")
target_size = image.size[::-1]
# Preprocess image
inputs = processor(images=image, return_tensors="pt").to(device)
# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    outputs = model(**inputs)


# Post-process results to retrieve instance segmentation maps
result = processor.post_process_instance_segmentation(outputs, threshold=0.5, target_sizes=[target_size])[
    0
]  # we pass a single output therefore we take the first result (single)
instance_seg_mask = result["segmentation"].cpu().detach().numpy()
print(f"Final mask shape: {instance_seg_mask.shape}")
print("Segments Information...")
for info in result["segments_info"]:
    print(f"  {info}")


def visualize_instance_seg_mask(mask):
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    # Create labels
    labels = np.unique(mask)
    label2color = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels
    }
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]
    image = image / 255
    return image


instance_seg_mask_disp = visualize_instance_seg_mask(instance_seg_mask)
plt.figure(figsize=(10, 10))
for plot_index in range(2):
    if plot_index == 0:
        plot_image = image
        title = "Original"
    else:
        plot_image = instance_seg_mask_disp
        title = "Segmentation"

    plt.subplot(1, 2, plot_index + 1)
    plt.imshow(plot_image)
    plt.title(title)
    plt.axis("off")
