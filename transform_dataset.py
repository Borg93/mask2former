import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, transform):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = np.array(self.dataset[idx]["pixel_values"])
        original_segmentation_map = np.array(self.dataset[idx]["label"])

        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed["image"], transformed["mask"]

        # convert to C, H, W
        image = image.transpose(2, 0, 1)

        return image, segmentation_map, original_image, original_segmentation_map


class ImageTransformation:
    def __init__(self) -> None:
        self.ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        self.ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    def train_and_test_transform(self):
        train_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=1333),
                A.RandomCrop(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=self.ADE_MEAN, std=self.ADE_STD),
            ]
        )

        test_transform = A.Compose(
            [
                A.Resize(width=512, height=512),
                A.Normalize(mean=self.ADE_MEAN, std=self.ADE_STD),
            ]
        )

        return train_transform, test_transform

    def unnormalize_transformed_image(self, image):
        unnormalized_image = (image * np.array(self.ADE_STD)[:, None, None]) + np.array(self.ADE_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        im = Image.fromarray(unnormalized_image)
        im.show()

    @staticmethod
    def visualize_mask(batch, labels, label_name):
        print("Label:", label_name)
        idx = labels.index(label_name)
        visual_mask = (batch["mask_labels"][0][idx].bool().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(visual_mask)
        im.show()


def custom_collate_fn(batch, preprocessor):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]

    return batch


def control_shape_and_image_transform(id2label, image_transformer, train_dataloader):
    batch = next(iter(train_dataloader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v[0].shape)

    pixel_values = batch["pixel_values"][0].numpy()
    print(pixel_values.shape)

    image_transformer.unnormalize_transformed_image(pixel_values)

    labels = [id2label[label] for label in batch["class_labels"][0].tolist()]
    print(labels)

    print(batch["mask_labels"][0].shape)

    image_transformer.visualize_mask(batch, labels, labels[0])

    return batch
