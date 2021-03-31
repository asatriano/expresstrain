# %%
# Adapted from:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os

# Clone ExpressTrain
os.system("rm -rf ./expresstrain")
os.system("git clone --branch development https://github.com/asatriano/expresstrain/")
# Import ExpressTrain :)
import expresstrain as et

# %%
# Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x 
    #logits (or compute logsoftmax, and use a NLLLoss as a 
    # custom loss function in ExpressTrainer)

# %%
def main():
    # Training hyperparameters
    parser=argparse.ArgumentParser(description='PyTorch FashionMNIST Example')
    parser.add_argument('--random-seed', type=int, default=42, metavar='RS',
                        help='input random seed integer (default: 42)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='input batch size to use at training (default: 32)')
    parser.add_argument('--batch-size-multiplier', type=int, default=2, metavar='BSM',
                        help='input batch size multiplier for validation (default: 2)')
    parser.add_argument('--num-workers-dataloader', type=int, default=0, metavar='NM',
                        help='input number of workers for dataloaders (default: 0)')
    parser.add_argument('--learning-rate', type=float, default=1e-2, metavar='LR',
                        help='input training learnign rate (default: 3e-4)')
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='input training epochs (default=10)')
    parser.add_argument('--path-performance', type=str, default=None, metavar='PP',
                        help='input saving path for loss and metrics (default: None)')
    parser.add_argument('--path-perf-model', type=str, default=None, metavar='PPM',
                        help='input saving path for loss, metric, and model params')
    parser.add_argument('--use-fp16', type=bool, default=False, metavar='FP16',
                        help='input whether to use Automatic Mixed Precision (default: True)')
    parser.add_argument('--use-progbar', type=bool, default=False, metavar='PB',
                        help='input whether to use Progress Bar')

    args=parser.parse_args()

# %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    torch.manual_seed(args.random_seed)
     
    # Define your transforms:
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Import your datasets
    dataset1 = datasets.FashionMNIST('./data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.FashionMNIST('./data', train=False,
                        transform=transform)

    # Define Dataloaders:
    train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': True}
    valid_kwargs = {'batch_size': args.batch_size*args.batch_size_multiplier,
                    'shuffle': False}
    workers_kwargs = {'num_workers': args.num_workers_dataloader}

    train_kwargs.update(workers_kwargs)
    valid_kwargs.update(workers_kwargs)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset2, **valid_kwargs)

    # Instance your favourite model and optimizer
    model = Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Define your favourite metric
    def accuracy(preds, targets):
        assert(len(preds)==len(targets))
        correct=torch.sum(preds == targets)
        return correct/len(targets)
    
    metric_used=accuracy

    # Subclass Express Train:
    class CustomExpressTrain(et.ExpressTrain):
        def __init__(self, **kwargs):
            super(CustomExpressTrain, self).__init__()
            self.initialize_all(kwargs)

        def on_train_epoch_begin(self):
            print(f"Message before epoch {self.epoch+1} - Today is a great day :)")
    
    # Instance your Custom Express Train trainer
    trainer_kwargs={'train_loader': train_loader,
                    'valid_loader': valid_loader,
                    'model': model,
                    'num_classes': 10,
                    'device': device,
                    'learning_rate': args.learning_rate,
                    'optimizer': optimizer,
                    'metric_used': metric_used,
                    'use_progbar': args.use_progbar,
                    'path_performance': args.path_performance,
                    'path_performance_and_model': args.path_perf_model}
    if args.use_fp16==True:
        print("Using Automatic Mixed Precision")
        trainer_kwargs.update({'fp16': args.use_fp16})
    
    trainer=CustomExpressTrain(**trainer_kwargs)

    trainer.fit(args.epochs)
                      

if __name__ == '__main__':
    main()
