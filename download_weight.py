
import os
import gdown
import argparse
parser = argparse.ArgumentParser(description='A simple command-line argument parser')

parser.add_argument('--normal_path', type=str, default = "https://drive.google.com/uc?id=1fknxkiPE-JYFYzFNWy7tCd1tOb7Qs2Pc", help='Normal Checkpoint')
parser.add_argument('--mlp_path', type=str, default = "https://drive.google.com/uc?id=1-J9yN-yZf5L4Ilg2MZbdjPzrPf5g4Tc5", help='MLP Checkpoint')
parser.add_argument('--moe_path', type=str, default = "https://drive.google.com/uc?id=1fIxybkqn0-4Ac9-ZsiPMyBsF9hs7BOsT", help='MoE Checkpoint')

args = parser.parse_args()

download_dir = 'downloaded_weights'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

normal_path = os.path.join(download_dir, 'normal_ckg.pth')
mlp_path = os.path.join(download_dir, 'mlp_ckg.pth')
moe_path = os.path.join(download_dir, 'moe_ckg.pth')

gdown.download(args.normal_path, normal_path)
gdown.download(args.mlp_path, mlp_path)
gdown.download(args.moe_path, moe_path)
