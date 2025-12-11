import torch

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None, num_samples=1):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.train()

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {}
    
    for x_0, _ in dataloader:
        
        batch_size = len(x_0)
                                
        if device.type == "cuda":
            x_0 = x_0.to(device)

        optimizer.zero_grad()
        # TODO: Add params to forward pass.
        #params = flatten_params(model)

        for _ in range(num_samples):

            logits, labels = model(x_0)
            losses = criterion(logits, labels, None, len(dataloader.dataset))
            losses["loss"].backward()

            for key, value in losses.items():
                metrics[key] = metrics.get(key, 0.0) + (batch_size / dataset_size) * (1 / num_samples) * value.item()

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(1/num_samples)

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()
                
    return metrics

def evaluate(model, criterion, dataloader, num_samples=1):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {}
    
    with torch.no_grad():
        for x_0, _ in dataloader:

            batch_size = len(x_0)

            if device.type == "cuda":
                x_0 = x_0.to(device)

            # TODO: Add params to forward pass.
            #params = flatten_params(model)

            for _ in range(num_samples):

                logits, labels = model(x_0)
                losses = criterion(logits, labels, None, len(dataloader.dataset))

                for key, value in losses.items():
                    metrics[key] = metrics.get(key, 0.0) + (batch_size / dataset_size) * (1 / num_samples) * value.item()
    
    return metrics
