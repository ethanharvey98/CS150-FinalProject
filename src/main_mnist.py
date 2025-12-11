import os
import time
import pandas as pd
import torch
import torchvision
# Importing our custom module(s)
import likelihoods
import losses
import models
import unet
import utils

# python main_mnist.py
if __name__ == "__main__":

    batch_size = 32
    dataset_dir = "/cluster/tufts/hugheslab/eharve06/MNIST"
    epochs = 1000
    experiments_dir = "/cluster/tufts/hugheslab/eharve06/CS150-FinalProject/experiments/MNIST2"
    lr = 0.01
    model_name = "batch_size=32_epochs=1000_lr=0.01_random_state=42"
    num_workers = 0
    random_state = 42
    
    torch.manual_seed(random_state)
    
    os.makedirs(experiments_dir, exist_ok=True)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=32),
    ])
    train_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=False, transform=transform, download=True)    
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    eps_model = unet.EnhanceUNet(image_channels=1)
    model = models.DenoisingDiffusionProbabilisticModel(eps_model=eps_model, T=5)
    likelihood = likelihoods.GaussianLikelihood()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    likelihood.to(device)
    
    criterion = losses.SimpleLoss(model, likelihood)

    optimizer = torch.optim.SGD([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr, weight_decay=0.0, momentum=0.9, nesterov=True)
            
    for epoch in range(epochs):
        
        train_epoch_start_time = time.time()
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, train_dataloader, num_samples=1)
        train_epoch_end_time = time.time()
        
        test_metrics = utils.evaluate(model, criterion, test_dataloader, num_samples=1)
    
        if epoch == 0:
            columns = ["epoch", *[f"test_{key}" for key in test_metrics.keys()], *[f"train_{key}" for key in train_metrics.keys()], "train_sec/epoch"]
            model_history_df = pd.DataFrame(columns=columns)
            
        row = [epoch, *[value for key, value in test_metrics.items()], *[value for key, value in train_metrics.items()], train_epoch_end_time - train_epoch_start_time]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f"{experiments_dir}/{model_name}.csv")
    
        torch.save(model.state_dict(), f"{experiments_dir}/{model_name}.pth")
        