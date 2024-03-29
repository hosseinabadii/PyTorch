import json
import os
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch


def create_samples(
    data: torch.utils.data.Dataset,
    num_samples: int,
    random_seed: int | None = None,
):
    if random_seed:
        random.seed(random_seed)
    samples = []
    labels = []
    for sample, label in random.sample(list(data), k=num_samples):
        samples.append(sample)
        labels.append(label)
    return samples, labels


def make_predictions(
    model: torch.nn.Module,
    samples: list[torch.Tensor],
    device: torch.device | str,
) -> list[int]:
    pred_labels = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            sample = sample.to(device)
            pred_logits = model(sample.unsqueeze(dim=0))
            pred_label = torch.argmax(pred_logits, dim=1)
            pred_labels.append(pred_label.cpu().item())

    return pred_labels


def plot_predictions(
    samples: list[torch.Tensor],
    lables: list[int],
    pred_labels: list[int],
    class_names: list[str],
    nrows: int = 3,
    ncols: int = 3,
    figsize: tuple[int, int] = (9, 9),
):
    plt.figure(figsize=figsize)
    for i, sample in enumerate(samples):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(sample.permute(1, 2, 0))
        pred_label = class_names[pred_labels[i]]
        truth_label = class_names[lables[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")
        plt.axis(False)


def plot_loss_curves(results, figsize: tuple[int, int] = (12, 6)):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=figsize)

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str,
    results: dict[str, list] | None = None,
) -> None:
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    if results:
        results_save_path = target_dir_path / (model_save_path.stem + ".json")
        print(f"[INFO] Saving training results to: {results_save_path}")
        with open(results_save_path, "w") as f:
            json.dump(results, f, indent=4)


def load_model_results(results_path: str) -> dict[str, list]:
    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def download_dataset_from_url(
    url: str,
    data_dir: str,
):
    # Setup path to data folder
    data_root = Path("data/")
    image_dir = data_root / data_dir
    zip_file_path = data_root / (data_dir + ".zip")

    # If the image folder doesn't exist, download it and prepare it...
    if image_dir.is_dir():
        print(f"{image_dir} directory exists.")
    else:
        print(f"Did not find {image_dir} directory, creating one...")
        image_dir.mkdir(parents=True, exist_ok=True)

        # Download data
        with open(zip_file_path, "wb") as f:
            print("Downloading data...")
            request = requests.get(url)
            f.write(request.content)
            print(f"Data Downloaded to {image_dir}")

        # Unzip data
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            print(f"Unzipping {zip_file_path}...")
            zip_ref.extractall(image_dir)
            print("Unzipping Done!")

    return image_dir


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames):3} images in '{dirpath}'."
        )
