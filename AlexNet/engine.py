import torch
from tqdm import tqdm


def train_fn(model, data_loader, loss_fn, optimizer, device):
    # set model into training mode
    model = model.train()
    loss_sum = 0
    progress_bar = tqdm(data_loader)
    for data in progress_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        model_output = model(images)

        loss = loss_fn(model_output, labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        loss_sum += loss.item()

    return loss_sum / len(data_loader)


def eval_fn(model, data_loader, device):
    # set model into evalution mode

    model = model.eval()
    correct = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            model_output = model(images)

            _, predictions = torch.max(model_output, dim=1)

            correct += torch.sum(predictions == labels).item()

        return correct / len(data_loader)
