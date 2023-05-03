import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download  # huggingface-cli login

# TODO clean this module and rewrite som variables..


def seg_plot(dataset, index, labels):
    image = dataset[index]["pixel_values"]
    image = np.array(image.convert("RGB"))
    annotation = dataset[index]["label"]
    segmentation_map = np.array(annotation)

    number_of_total_labels = len(labels)
    palette = color_palette()[: (number_of_total_labels + 1)]

    color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)  # height, width, 3
    for label, color in enumerate(palette):
        color_segmentation_map[segmentation_map - 1 == label, :] = color

    # Convert to BGR
    ground_truth_color_seg = color_segmentation_map[..., ::-1]
    ground_truth_palette = np.array(palette)[..., ::-1]

    only_unique_ground_truth_color_seg = np.unique(ground_truth_color_seg.reshape(-1, 3), axis=0)

    temp_label_and_color_labels_from_palette = []
    for id, color_from_gt in enumerate(ground_truth_palette):
        for unique_colors in only_unique_ground_truth_color_seg:
            if np.array_equal(unique_colors, color_from_gt):
                temp_label_and_color_labels_from_palette.append([id + 1, color_from_gt])

    patches = [mpatches.Patch(color=color[1] / 255.0) for color in temp_label_and_color_labels_from_palette]
    lengend_label = [labels[lab[0]] for lab in temp_label_and_color_labels_from_palette]

    img_annotated = np.array(image) * 0.3 + ground_truth_color_seg * 0.7
    img_annotated = img_annotated.astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Image_index: {index}")
    fig.subplots_adjust(left=0.01, right=0.84, top=0.95, bottom=0.05, wspace=0.05)
    ax1.imshow(image)
    ax2.imshow(img_annotated)
    ax2.legend(
        patches,
        lengend_label,
        loc="upper left",
        bbox_to_anchor=(1.04, 1),
        borderaxespad=0,
    )

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.waitforbuttonpress(0)
    plt.close(fig)


def color_palette():
    """Color palette that maps each class to RGB values.

    This one is actually taken from ADE20k.
    """
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]


def get_labels(dataset_name, filename):
    id2label = json.load(
        open(
            hf_hub_download(repo_id=dataset_name, filename=filename, repo_type="dataset"),
            "r",
        )
    )
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    return id2label, label2id
