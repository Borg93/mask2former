import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def trainer_accelerate(train_dataloader, val_dataloader, model, device, num_epochs, log_step, lr):
    # Tensorboard
    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} | Training")

        model.train()
        train_loss, val_loss = [], []

        for step, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            train_loss.append(loss.item())
            loss.backward()

            if step == log_step:
                print("  Training loss: ", round(sum(train_loss) / len(train_loss), 6))
                train_log_step_loss = round(sum(train_loss) / len(train_loss), 6)
                writer.add_scalar("Avg Train Loss at Step", train_log_step_loss, step)

            # Optimization
            optimizer.step()

        # Average train epoch loss
        train_loss = sum(train_loss) / len(train_loss)

        model.eval()

        print(f"Epoch {epoch} | Validation")
        for step, batch in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                )
                # Get validation loss
                loss = outputs.loss
                val_loss.append(loss.item())
                if step == log_step:
                    print("  Validation loss: ", round(sum(val_loss) / len(val_loss), 6))
                    val_log_step_loss = round(sum(val_loss) / len(val_loss), 6)
                    writer.add_scalar("Avg Val Loss at Step", val_log_step_loss, step)

        # Average validation epoch loss
        val_loss = sum(val_loss) / len(val_loss)

        # Print epoch losses
        print(f"Epoch {epoch} | train_loss: {train_loss} | validation_loss: {val_loss}")

        writer.close()

    return model
