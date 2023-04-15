from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10('./data', train=True, transform=transform)
loader = DataLoader(dataset, 128, shuffle=False, drop_last=True, pin_memory=False)

nb_samples = 0.
channel_mean = torch.Tensor([0., 0., 0.])
channel_std = torch.Tensor([0., 0., 0.])
for images, x in tqdm(loader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1)*images.size(2), 3)
    for i in range(3):
        channel_mean[i]+=images[:, :,i].mean(1).sum(0)
        channel_std[i]+=images[:, :,i].std(1).sum(0)
    nb_samples += batch_samples

channel_mean /= nb_samples
channel_std /= nb_samples