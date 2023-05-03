import evaluate
import torch
from tqdm.auto import tqdm


def evaluate_metric(id2label, preprocessor, test_dataloader, model, step_to_idx=None):
    eval_metric = "mean_iou"
    metric = evaluate.load(eval_metric)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    for idx, batch in enumerate(tqdm(test_dataloader)):
        if step_to_idx is not None:
            if idx > step_to_idx:
                break

        pixel_values = batch["pixel_values"]

        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values.to(device))

        # get original images
        original_images = batch["original_images"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        # predict segmentation maps
        predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # get ground truth segmentation maps
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]

        metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)

    # NOTE this metric outputs a dict that also includes the mIoU per category as keys
    # so if you're interested, feel free to print them as well
    print(f"{eval_metric}:", metric.compute(num_labels=len(id2label), ignore_index=0)[eval_metric])


def initial_loss_and_eval_metric(model, batch):
    outputs = model(
        pixel_values=batch["pixel_values"],
        mask_labels=batch["mask_labels"],
        class_labels=batch["class_labels"],
    )

    print(outputs.loss)
