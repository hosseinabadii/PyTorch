from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ActivationFn = Callable[[torch.Tensor], torch.Tensor]
AccuracyFn = Callable[[torch.Tensor], torch.Tensor]


def train_step(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    activation_fn: ActivationFn,
    accuracy_fn: AccuracyFn,
    device: torch.device,
) -> tuple[float, float]:
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += accuracy_fn(activation_fn(y_logits), y).item()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    activation_fn: ActivationFn,
    accuracy_fn: AccuracyFn,
    device: torch.device,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(activation_fn(y_logits), y).item()

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        return test_loss, test_acc


def evaluation_step(
    model: torch.nn.Module,
    data_loader: DataLoader,
    activation_fn: ActivationFn,
    device: torch.device,
):
    model.to(device)
    model.eval()
    y_preds = []
    with torch.inference_mode():
        for X, _ in tqdm(data_loader, desc="Making predictions"):
            X = X.to(device)
            y_logits = model(X)
            y_pred = activation_fn(y_logits)
            y_preds.append(y_pred.cpu())

        return torch.cat(y_preds)


def train(
    epochs: int,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    activation_fn: ActivationFn,
    accuracy_fn: AccuracyFn,
    device: torch.device,
) -> dict[str, list]:

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            activation_fn=activation_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            activation_fn=activation_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        tqdm.write(
            f"Epoch: {epoch:3} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
