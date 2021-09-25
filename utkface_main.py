from dataloader_factory import DataloaderFactory
import torchvision.models as models
from torch.optim import SGD, Adam
import torch.nn as nn
from utils import get_accuracy
from tqdm import tqdm

tmp = DataloaderFactory.get_dataloader("utkface", img_size=176,
                                                    batch_size=128,
                                                    num_workers=0,
                                                    target=None)

num_classes, num_groups, train_loader, test_loader = tmp

model = models.resnet18(pretrained=True)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 50

model.train()
for epoch in range(epochs):
    running_acc = 0.0
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader)):
        inputs, _, groups, targets, _ = data
        labels = targets

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        running_acc += get_accuracy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(running_acc / i, running_loss / i)



