
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.models import ResNet152_Weights, resnet152
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
from dataset import UAVDataset
from mlp_classifier import MLP_Classifier
from moe_classifier import MoE_Classifier
from trainer import Trainer
from trainer_moe import Trainer_MoE
import argparse
parser = argparse.ArgumentParser(description='A simple command-line argument parser')

# Add arguments
parser.add_argument('--train_folder', type=str, help='Train folder')
parser.add_argument('--val_folder', type=str, help='Val folder')
parser.add_argument('--test_folder', type=str, help='Test folder')
parser.add_argument('--label_path', type=str, help='File 952_labels.txt')
parser.add_argument('--mode', type=str, choices=['normal', 'mlp', 'moe'], default='normal', help='Choose classifier layer')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--continue_epoch', type=int, default=0, help='Use for continuous training')

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

# Load data
train_data = UAVDataset(data_folder = args.train_folder,
                        label_list = label_list,
                        transform = transform)

val_data = UAVDataset(data_folder = args.val_folder,
                        label_list = label_list,
                        transform = transform)

test_data = UAVDataset(data_folder = args.test_folder,
                        label_list = label_list,
                        transform = transform)

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

model = resnet152(weights = None)
if args.mode == 'mlp':
  model.fc = MLP_Classifier(in_features=2048,
                                  out_features=train_data.num_labels(),
                                  hidden_size = 1024)
elif args.mode == 'moe':
  model.fc = MoE_Classifier(in_features=2048,
                                  out_features=train_data.num_labels(),
                                  hidden_size = 1024)
else:
  model.fc = nn.Linear(in_features=2048, out_features=train_data.num_labels(), bias=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.85, 0.95), weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.mode == 'moe':
  trainer = Trainer_MoE(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device)
else :
  trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device)

trainer.fit(epochs=args.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            eval_every=2,
            continue_epoch=args.continue_epoch)
