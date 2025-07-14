import pika
import uuid
import argparse
import yaml
import sys
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim

import src.Log

from sklearn.metrics import roc_curve, auc
from src.RpcClient import RpcClient
from src.Model import ICUData

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--attack', type=bool, required=False,
                    default=False, help='Set to True to enable attack mode, False otherwise.')
parser.add_argument('--attack_mode', type=str, choices=['Random', 'Min-Max', 'Min-Sum', 'Opt-Fang', 'LIE'],
                    help="Mode of operation when the attack is enabled (e.g., Random or Min-Max, Min-Sum ...).")
parser.add_argument('--attack_round', type=int,
                    help="Client attack at round.")
parser.add_argument('--attack_args', type=float, nargs='+', required=False,
                    help="A list of attack args. Use space to separate values (e.g., 0.1 0.2 0.5).")

args = parser.parse_args()

# Validate condition
if args.attack and not args.attack_mode:
    print("Error: --attack_mode is required when --attack is True.")
    sys.exit()
if args.attack and not args.attack_round:
    print("Error: --attack_round is required when --attack is True.")
    sys.exit()

print(f"Attack: {args.attack}, Mode: {args.attack_mode}")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

credentials = pika.PlainCredentials(username, password)


def train_on_device(model, data_name, epoch, lr, momentum, clip_grad_norm, trainloader):
    if data_name == "ICU":
        return train_ICU(model, epoch, lr, momentum, clip_grad_norm, trainloader)
    elif data_name == "HAR":
        return train_HAR(model, epoch, lr, momentum, clip_grad_norm, trainloader)
    else:
        raise ValueError(f"Data name '{data_name}' is not valid.")


def train_ICU(model, epoch, lr, momentum, clip_grad_norm, trainloader):
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epoch):
        epoch_loss = 0.0
        for vitals, labs, labels in tqdm(trainloader):
            model.train()
            optimizer.zero_grad()

            if vitals.size(0) == 1 or labs.size(0) == 1:
                continue
            vitals = vitals.to(device)
            labs = labs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)

            #optimizer.zero_grad()
            output = model(vitals, labs)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            loss_ = epoch_loss / len(trainloader)

            if torch.isnan(loss).any():
                src.Log.print_with_color("NaN detected in loss, stop training", "yellow")
                return False

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            loss.backward()
            optimizer.step()

        src.Log.print_with_color(f"Loss {loss_} ", "yellow")

    return True


def train_HAR(model, epoch, lr, momentum, clip_grad_norm, trainloader):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for e in range(epoch):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(trainloader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {e + 1}, Loss: {total_loss:.4f}")
    return True


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "message": "Hello from Client!", "attack": args.attack}
    print("Client ID:", client_id)
    client = RpcClient(client_id, address, username, password, train_on_device, device)
    if args.attack:
        client.set_attack_config(args.attack_mode, args.attack_round, args.attack_args)

    client.send_to_server(data)
    client.wait_response()
