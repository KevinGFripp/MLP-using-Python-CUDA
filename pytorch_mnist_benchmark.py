import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def pytorch_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load dataset on CPU ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # ---- Move entire dataset to GPU ----
    X = train_data.data.float().view(-1, 784) / 255.0
    y = train_data.targets

    X = X.to(device)
    y = y.to(device)

    # ---- Model ----
    model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---- Training ----
    batch_size = 2048
    epochs = 10
    N = X.shape[0]

    torch.cuda.synchronize()
    start = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            data = X[idx]
            target = y[idx]

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    end = time.time()

    print(f"Total time: {end - start:.3f} s")