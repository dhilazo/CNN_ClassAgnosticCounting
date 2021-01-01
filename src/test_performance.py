import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.saved_cifar10_count_dataset import SavedCIFAR10CountDataset
from models.etcnet_model import ETCNet
from models.etscnn_model import ETSCNN
from models.model import DoubleInputNet
from models.resnet import ResNet
from models.siamese_resnet_model import SiameseResNet

image_grid_distribution = (3, 3)
transform = transforms.Compose([transforms.ToTensor()])

data_root = './data/CIFAR10Count'
test_set = SavedCIFAR10CountDataset(data_root, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

#model = ResNet(output_size=1)
# model = DoubleInputNet(output_size=1)
# model = SiameseResNet(output_size=1)
# model = ETCNet(output_size=1)
model = ETSCNN(output_size=1)
print(type(model).__name__, flush=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# model = nn.DataParallel(model)  # If the saved model is a dataparallel already, otherwise do after the if
# model.load_state_dict(torch.load("./trained_models/ResNet_batch.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/DoubleInputCount_ResNet_batch.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/SiameseResNet_Raw_final.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/ETCNet_batch.pt", map_location=device))
model.load_state_dict(torch.load("./trained_models/ETSCNN_batch.pt", map_location=device))

model = model.to(device)
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()

model.eval()

max_test_loss = 0
test_accumulated_loss = 0
total_batches = 0
mse_losses = []
mae_losses = []
accuracies = []

print(len(test_set), flush=True)
with torch.no_grad():
    for batch, data in enumerate(test_loader):
        print(batch, flush=True)
        image_grids, templates, counts = data

        image_grids = image_grids.to(device)
        counts = counts.to(device)

        for i in range(len(templates)):
            correct = 0
            current_template = templates[i]
            current_template = current_template.to(device)
            outputs = model(image_grids, current_template)
            mse_losses.append(mse_criterion(outputs, counts[:, i]).item())
            mae_losses.append(mae_criterion(outputs, counts[:, i]).item())

            correct += sum(np.around(np.reshape(outputs.cpu().numpy(), len(outputs)), decimals=0) == np.reshape(
                counts[:, i].cpu().numpy(), len(outputs)))
            accuracies.append(correct / len(image_grids))

print("MSE Loss:", np.mean(mse_losses), flush=True)
print("MAE Loss:", np.mean(mae_losses), flush=True)
print("Avg Accuracy:", np.mean(accuracies) * 100, "%", flush=True)
