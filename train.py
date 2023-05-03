import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def trainer_accelerate(train_dataloader, val_dataloader, model, num_epochs, log_step, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tensorboard
    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_num_samples = 0
    val_num_samples = 0

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

            batch_size = batch["pixel_values"].size(0)
            train_num_samples += batch_size

            if step % log_step == 0:
                avg_train_loss = round(sum(train_loss) / len(train_loss), 6)
                print("  Training loss: ", avg_train_loss)
                writer.add_scalar("Running Train Loss", avg_train_loss, train_num_samples)

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
                batch_size = batch["pixel_values"].size(0)
                val_num_samples += batch_size

                if step % log_step == 0:
                    avg_val_loss = round(sum(val_loss) / len(val_loss), 6)
                    print("  Validation loss: ", avg_val_loss)
                    writer.add_scalar("Avg Val Loss at Step", avg_val_loss, val_num_samples)

        # Average validation epoch loss
        val_loss = sum(val_loss) / len(val_loss)

        # Print epoch losses
        print(f"Epoch {epoch} | train_loss: {train_loss} | validation_loss: {val_loss}")

        writer.close()

    return model
