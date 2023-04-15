from torch import Tensor
import scipy
import torch
from torchvision import transforms
import timm
from data_aug.dybat_datasets import DyBATDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(0)

def transform(crop_size):
    """Return a set of data augmentation transformations."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    data_transforms = transforms.Compose([transforms.Resize((crop_size, crop_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    return data_transforms

def torch_compute_confidence_interval(data: Tensor, confidence: float = 0.95) -> Tensor:
    """Computes the confidence interval for a given survey of a data set."""
    n = len(data)
    mean: Tensor = data.mean()
    se: Tensor = data.std(unbiased=True) / (n**0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
    ci = t_p * se
    return mean.item(), ci.item()

if __name__ == "__main__":
    device = 'cuda:0'
    ds = ['cifar10', 'cifar100', 'stl10']
    n_classes = {'cifar10': 10, 'cifar100': 100, 'stl10': 10}
    nets = ['resnet18', 'resnet50', 'efficientnet_b4', 'efficientnetv2_s', 'mobilenetv3_small_100']
    deltas = [0.2, 0.5, 0.8, 1.0]
    crop_size = 128
    batch_sizes = [64, 128, 256, 512]
    exps = []
    train_top_1 = []
    test_top_1 = []

    for network in nets:
        for d in ds:
            dataset = DyBATDataset('./data')
            train_dataset, test_dataset = dataset.get_dataset(d, crop_size)
            train_loader = DataLoader(train_dataset, 512, shuffle=False)
            test_loader = DataLoader(test_dataset, 512, shuffle=False)
            net = timm.create_model(network, pretrained=False, num_classes=n_classes[d]).to(device)
            acc = MulticlassAccuracy(n_classes[d], top_k=1, average='weighted').to(device)

            for delta in deltas:
                for batch_size in batch_sizes:
                    exp = f'delta_{delta}_{network}_{d}_b_{batch_size}'
                    ckpt = f'./checkpoints/{exp}.pt'
                    if not os.path.exists(ckpt): continue 
                    net.load_state_dict(torch.load(ckpt))

                    acc_1 = []
                    with torch.no_grad():
                        for i, data in enumerate(tqdm(train_loader), 0):
                            inputs, labels = data
                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            outputs = net(inputs)
                            batch_acc = acc(outputs, labels)
                            acc_1.append(batch_acc)

                    acc_1 = torch.tensor(acc_1).to(device)
                    mean, std = torch_compute_confidence_interval(acc_1)
                    exps.append(exp)
                    train_top_1.append(f'{mean:.3f}±{std:.3f}')

                    acc_1 = []
                    with torch.no_grad():
                        for i, data in enumerate(tqdm(test_loader), 0):
                            inputs, labels = data
                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            outputs = net(inputs)
                            batch_acc = acc(outputs, labels)
                            acc_1.append(batch_acc)

                    acc_1 = torch.tensor(acc_1).to(device)
                    mean, std = torch_compute_confidence_interval(acc_1)
                    test_top_1.append(f'{mean:.3f}±{std:.3f}')
                
    dict = {'exp': exps, 'train top-1 acc': train_top_1, 'test top-1 acc': test_top_1}
    df = pd.DataFrame(dict)
    df.to_csv('./confidence_metrics_r18.csv')