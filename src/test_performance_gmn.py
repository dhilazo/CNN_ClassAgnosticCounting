import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.ilsvrc_dataset import ILSVRC
from datasets.saved_cifar10_count_dataset import SavedCIFAR10CountDataset
from models.adapted_gmn import AdaptedGenericMatchingNetwork
from models.etcnet_model import ETCNet
from models.etscnn_model import ETSCNN
from models.gmn_etcnet import GMNETCNet
from models.gmn_etscnn import GmnETSCNN
from models.model import DoubleInputNet
from models.resnet import ResNet
from models.siamese_gmn import SiameseGenericMatchingNetwork
from models.siamese_resnet_model import SiameseResNet

transform = transforms.Compose([transforms.ToTensor()])

data_root = './data/ILSVRC/ILSVRC2015'
image_shape = (255, 255)
test_set = ILSVRC(data_root, image_shape=image_shape, data_percentage=0.5, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

model_class = AdaptedGenericMatchingNetwork
# model_class = SiameseGenericMatchingNetwork
# model_class = GMNETCNet
# model_class = GMNETCNet
# model_class = GmnETSCNN

model = model_class(output_matching_size=(255 // 4, 255 // 4))
print(type(model).__name__, flush=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = nn.DataParallel(model)  # If the saved model is a dataparallel already, otherwise do after the if
model.load_state_dict(torch.load("./trained_models/AdaptedGMN_laura_batch.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/SiameseGMN_laura_batch.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/GMNETCNet_final_batch.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/GMNETCNet_CIFAR_batch.pt", map_location=device))
# model.load_state_dict(torch.load("./trained_models/GmnETSCNN_batch.pt", map_location=device))

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
failed = []
print(len(test_set), flush=True)
with torch.no_grad():
    for batch, data in enumerate(test_loader):
        print(batch, flush=True)
        images, templates, ground_truth, count, resized_template = data

        images = images.to(device)
        templates = templates.to(device)
        ground_truth = ground_truth.to(device)
        resized_template = resized_template.to(device)

        correct = 0
        if (model_class == AdaptedGenericMatchingNetwork) or (model_class == SiameseGenericMatchingNetwork):
            outputs = model(images, templates)
        else:
            outputs = model(images, templates, resized_template)
        outputs = torch.FloatTensor(
            [SiameseGenericMatchingNetwork.get_count(outputs[i].detach().cpu().numpy()[0]) for i in
             range(len(outputs))])
        mse_losses.append(mse_criterion(outputs, count).item())
        mae_losses.append(mae_criterion(outputs, count).item())

        correct += sum(np.around(np.reshape(outputs.cpu().numpy(), len(outputs)), decimals=0) == np.reshape(
            count.cpu().numpy(), len(outputs)))
        accuracies.append(correct / len(images))

print("MSE Loss:", np.mean(mse_losses), flush=True)
print("MAE Loss:", np.mean(mae_losses), flush=True)
print("Avg Accuracy:", np.mean(accuracies) * 100, "%", flush=True)
