import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.saved_cifar10_count_dataset import SavedCIFAR10CountDataset
from models.etcnet_model import ETCNet
from trainer import Trainer

image_grid_distribution = (3, 3)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_root = './data/CIFAR10Count'
test_set = SavedCIFAR10CountDataset(data_root, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

model = ETCNet(output_size=1)
model.load_state_dict(torch.load("./trained_models/ETCNet_batch.pt"))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainer = Trainer(model, criterion, optimizer, "test", device=device)
model.eval()

max_test_loss = 0
test_accumulated_loss = 0
total_batches = 0
with torch.no_grad():
    for data in test_loader:
        image_grids, templates, counts = data

        image_grids = image_grids.to(device)
        counts = counts.to(device)

        for i in range(len(templates)):
            current_template = templates[i]
            current_template = current_template.to(device)
            outputs = model(image_grids, current_template)
            for pred, truth in zip(outputs, counts[:, i]):
                print(f"Pred: {pred.item()} Truth: {truth.item()}")
