import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerModel,
)

from eval import evaluate_metric, initial_loss_and_eval_metric
from train import trainer_accelerate
from transform_dataset import (
    ImageSegmentationDataset,
    ImageTransformation,
    control_shape_and_image_transform,
    custom_collate_fn,
)
from utils import get_labels, seg_plot


def main(batch_size, num_epochs, log_step, step_to_idx, lr):
    # load data
    local_cache_dir = "/home/gabriel/Desktop/mask2former/.cache"
    dataset_name = "segments/sidewalk-semantic"
    dataset = load_dataset(dataset_name, cache_dir=local_cache_dir)

    dataset = dataset.shuffle(seed=1)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=420)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    filename_label_dataset = "id2label.json"

    id2label, label2id = get_labels(dataset_name, filename=filename_label_dataset)

    # Sample plot
    seg_plot(train_ds, index=3, labels=id2label)

    # Transform data
    image_transformer = ImageTransformation()
    train_transform, test_transform = image_transformer.train_and_test_transform()
    train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
    test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)

    # unnormalize image
    # image, segmentation_map, _, _ = train_dataset[0]
    # image_transformer.unnormalize_transformed_image(image)

    preprocessor = Mask2FormerImageProcessor(
        ignore_index=0,
        reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, preprocessor),
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, preprocessor),
    )

    ## Control shape and image_transformation
    batch = control_shape_and_image_transform(id2label, image_transformer, train_dataloader)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic",
        id2label=id2label,
        ignore_mismatched_sizes=True,
        cache_dir=local_cache_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    initial_loss_and_eval_metric(id2label, preprocessor, val_dataloader, model, batch, step_to_idx)

    model = trainer_accelerate(train_dataloader, val_dataloader, model, device, num_epochs, log_step, lr)

    evaluate_metric(id2label, preprocessor, val_dataloader, model, step_to_idx)

    model.save_pretrained("./.model")


if __name__ == "__main__":
    main(batch_size=2, num_epochs=10, log_step=50, step_to_idx=200, lr=5e-5)
