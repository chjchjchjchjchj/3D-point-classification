from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN, GCN, GCNResNet, GCNPolynomial
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import ipdb
import sys
import time
import hydra
import wandb
from tqdm import tqdm
import pandas as pd

def train_step(model, epoch, train_loader, config, device, optimizer):
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)
    
    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch}/{config.epochs}"
    )
    for batch_idx in progress_bar:
        # ipdb.set_trace()
        all_data = next(iter(train_loader))
        data, label = all_data[0].to(device), all_data[1].to(device).squeeze()
        if config.model_name == 'pointnet' or config.model_name == 'dgcnn':
            data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        prediction = model(data)
        # loss = F.nll_loss(prediction, label)
        loss = F.cross_entropy(prediction, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(label).sum().item()
    
    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / len(train_loader.dataset)
    
    wandb.log({
        "Train/Loss": epoch_loss,
        "Train/Accuracy": epoch_accuracy
    })

def val_step(model, epoch, val_loader, config, device):
    model.eval()
    
    epoch_loss, correct = 0, 0
    num_val_examples = len(val_loader)

    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"Validation Epoch {epoch}/{config.epochs}"
    )

    for batch_idx in progress_bar:
        # data = next(iter(val_loader)).to(device)
        all_data = next(iter(val_loader))
        data, label = all_data[0].to(device), all_data[1].to(device).squeeze()
        if config.model_name == 'pointnet' or config.model_name == 'dgcnn':
            data = data.permute(0, 2, 1)

        with torch.no_grad():
            prediction = model(data)
        
        # loss = F.nll_loss(prediction, label)
        loss = F.cross_entropy(prediction, label)
        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(label).sum().item()
    
    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / len(val_loader.dataset)
    
    wandb.log({
        "Validation/Loss": epoch_loss,
        "Validation/Accuracy": epoch_accuracy
    })


@hydra.main(config_path="configs", config_name="defaults", version_base="1.1")
def main(args):
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    wandb_project = "point-cloud-classification" #@param {"type": "string"}
    wandb_run_name = args.exp_name

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="baseline-train", config=args)
    # config = wandb.config
    # ipdb.set_trace()
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model_name == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model_name == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model_name == 'gcn':
        model = GCN(args).to(device)
    elif args.model_name == 'gcnresnet':
        model = GCNResNet(args).to(device)
    elif args.model_name == 'gcnpolynomial':
        model = GCNPolynomial(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)

    if not args.eval:
        for epoch in range(1, args.epochs + 1):
            train_step(model=model, epoch=epoch, train_loader=train_loader, config=args, device=device, optimizer=optimizer)
            val_step(model=model, epoch=epoch, val_loader=test_loader, config=args, device=device)
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() },
                "checkpoint.pt"
            )
            artifact_name = wandb.util.make_artifact_name_safe(f"{wandb.run.name}-{wandb.run.id}-checkpoint")
            checkpoint_artifact = wandb.Artifact(artifact_name, type="checkpoint")
            checkpoint_artifact.add_file("checkpoint.pt")
            wandb.log_artifact(checkpoint_artifact, aliases=["latest", f"epoch-{epoch}"])
    else:
        class_correct = np.zeros(40)
        class_total = np.zeros(40)
        model.load_state_dict(torch.load(args.model_path))
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=1, shuffle=False, drop_last=False)
        with torch.no_grad():
            for i, all_data in enumerate(test_loader):
                data, label = all_data[0].to(device), all_data[1].to(device).squeeze()

                prediction = model(data)

                correct = prediction == label

        class_accuracy = class_correct / class_total
        for i in range(40):
            print(f'Class {i}: Accuracy {class_accuracy[i] * 100:.2f}%')

        class_accuracy = pd.DataFrame(class_accuracy)
        class_accuracy.to_csv('out.csv')


if __name__ == "__main__":
    main()