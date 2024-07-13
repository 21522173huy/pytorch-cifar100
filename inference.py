
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.models import ResNet152_Weights, resnet152
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
from dataset import UAVDataset
from mlp_classifier import MLP_Classifier
from moe_classifier import MoE_Classifier
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='A simple command-line argument parser')

# Add arguments
parser.add_argument('--test_folder', type=str, help='Test folder')
parser.add_argument('--label_path', type=str, help='File 952_labels.txt')
parser.add_argument('--mode', type=str, choices=['normal', 'mlp', 'moe'], default='normal', help='Choose classifier layer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

args = parser.parse_args()

# Labels
with open(args.label_path, 'r') as f:
    lines = f.readlines()

label_list = []
for i, line in enumerate(lines):
    if i == 0 : continue
    if line.strip():
        parts = line.strip().split()
        label = int(parts[0])
        character = parts[1]
        latin = parts[-1]
        label_list.append({
            'latin' : latin,
            'character' : character
        })

# Transform

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset, Dataloader
test_data = UAVDataset(data_folder = args.test_folder,
                        label_list = label_list,
                        transform = transform)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Load Model
model = resnet152(weights = None)
if args.mode == 'mlp':
  model.fc = MLP_Classifier(in_features=2048,
                                  out_features=952,
                                  hidden_size = 1024)
elif args.mode == 'moe':
  model.fc = MoE_Classifier(in_features=2048,
                                  out_features=952,
                                  hidden_size = 1024)
else:
  model.fc = nn.Linear(in_features=2048, out_features=952, bias=True)

# Load checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(f'downloaded_weights/{args.mode}_ckg.pth', map_location = torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_correct = 0
total_samples = 0

model.eval()
model.to(device)

for sample in tqdm(test_dataloader):

    image = sample['image'].to(device)
    label = sample['label'].to(device)

    output = model(image)
    if args.mode == 'moe' : prediction = output[0].argmax(dim = -1)
    else : prediction = output.argmax(dim = -1)

    total_correct += (prediction == label).sum().item()
    total_samples += len(label)

accuracy = total_correct / total_samples
print(f'Accuracy: {accuracy:.4f}')
