import timm
import torch
import wandb
import logging
import argparse 
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy

from utils import metrics
from data_aug.dybat_datasets import DyBATDataset

parser = argparse.ArgumentParser(description='turbo')
parser.add_argument('-data', metavar='DIR', default='./data', help='path to dataset')
parser.add_argument('-d', '--dataset_name', default='cifar10',help='dataset name', choices=['cifar10', 'cifar100', 'imagenet', 'stl10', 'oxfordiiitpet', 'celeba', 'stanfordcars', 'mnist', 'flowers102'])
parser.add_argument('-c', '--crop_size', default=128, type=int, help='crop size')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4', 'efficientnetv2_s', 'mobilenetv3_small_100'])
parser.add_argument('-n', '--n_classes', type=int, default=None, help='the number of classes for classification')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-m', '--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--log_interval', default=1, type=int, help='Log every n steps')
parser.add_argument('-delta', default=0.5, type=float, help='rebatching fraction (default: 0.5)')
parser.add_argument('-zeta', default=1, type=int, help='zeta (default: 1)')
parser.add_argument('-device', type=int, default=0, help='device')
args = parser.parse_args()

torch.manual_seed(0)

def normal_train(model, train_loader, test_loader, criterion, optimizer, auroc, acc_1, acc_5):
  for epoch in tqdm(range(args.epochs)):
    for _, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs = inputs.to(args.device)
      labels = labels.to(args.device)

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    tr_loss, tr_top1, tr_top5, tr_auc = metrics(model, train_loader, criterion, auroc, acc_1, acc_5, args.device)
    val_loss, val_top1, val_top5, val_auc = metrics(model, test_loader, criterion, auroc, acc_1, acc_5, args.device)

    wandb.log({'Train Loss': tr_loss, 'Train Top-1 Accuracy': tr_top1, 'Train Top-5 Accuracy': tr_top5, 'Train AUC': tr_auc, 
               'Val Loss': val_loss, 'Val Top-1 Accuracy': val_top1, 'Val Top-5 Accuracy': val_top5, 'Val AUC': val_auc})
    
  ckpt = f'checkpoints/delta_{args.delta}_{args.arch}_{args.dataset_name}_b_{args.batch_size}.pt'
  torch.save(model.state_dict(), ckpt)

def turbo_train(model, train_loader, test_loader, criterion, optimizer, auroc, acc_1, acc_5):
  n_iter = 0
  b = int(len(train_loader))
  delta = int(args.delta * b)
  args.zeta = int((args.epochs - 1) / args.delta)
  args.log_interval = b
  retrain_batches = [(0, 0)] * b

  for _ in tqdm(range(1)):
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs = inputs.to(args.device)
      labels = labels.to(args.device)
      
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      retrain_batches[i] = (i, loss.item())

      if (n_iter + 1) % args.log_interval == 0:
        train_loss, train_top1, train_top5, train_auc = metrics(model, train_loader, criterion, auroc, acc_1, acc_5, args.device)
        val_loss, val_top1, val_top5, val_auc = metrics(model, test_loader, criterion, auroc, acc_1, acc_5, args.device)

        wandb.log({'Train Loss': train_loss, 'Train Top-1 Accuracy': train_top1, 'Train Top-5 Accuracy': train_top5, 'Train AUC': train_auc,
                    'Val Loss': val_loss, 'Val Top-1 Accuracy': val_top1, 'Val Top-5 Accuracy': val_top5, 'Val AUC': val_auc})      

      n_iter += 1
      
  assert(n_iter == b)

  args.zeta *= 50
  flag = False
  log_iter = 1

  for zeta in range(args.zeta):
      if(flag):
          logging.info(f'zeta loop broken at {zeta}')
          break
      j = 0
      retrain_batches = sorted(retrain_batches, key = lambda a: a[1], reverse=True)
      batch_ids = [p[0] for p in retrain_batches]
      actual_retrain = retrain_batches[:int(delta)]
      retrain_ids = sorted([p[0] for p in actual_retrain])

      for i, data in enumerate(tqdm(train_loader), 0):
        if(j < len(retrain_ids) and retrain_ids[j] == i):
          if(n_iter + 1) > (args.epochs * b):
            logging.info(f'break exec at at {n_iter + 1}')
            flag = True
            break
          
          inputs, labels = data
          inputs = inputs.to(args.device)
          labels = labels.to(args.device)
          
          outputs = model(inputs)
          batch_loss = criterion(outputs, labels)
          
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          
          # update retrain_batches list to new losses
          idx = batch_ids.index(retrain_ids[j])
          retrain_batches[idx] = (retrain_ids[j], batch_loss.item())
          
          if(n_iter + 1) % args.log_interval == 0:
            train_loss, train_top1, train_top5, train_auc = metrics(model, train_loader, criterion, auroc, acc_1, acc_5, args.device)
            val_loss, val_top1, val_top5, val_auc = metrics(model, test_loader, criterion, auroc, acc_1, acc_5, args.device)

            wandb.log({'Train Loss': train_loss, 'Train Top-1 Accuracy': train_top1, 'Train Top-5 Accuracy': train_top5, 'Train AUC': train_auc,
                        'Val Loss': val_loss, 'Val Top-1 Accuracy': val_top1, 'Val Top-5 Accuracy': val_top5, 'Val AUC': val_auc})      
            log_iter += 1
              
          j += 1
          n_iter += 1
      
  ckpt = f'checkpoints/delta_{args.delta}_{args.arch}_{args.dataset_name}.pt'
  torch.save(model.state_dict(), ckpt)
    
if __name__ == "__main__":
  name = f'delta_{args.delta}_{args.arch}_{args.dataset_name}'
  wandb.init(project=f'runs_crop_size_{args.crop_size}', name=name)
  # wandb.run.log_code(".")
  args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

  dataset = DyBATDataset(args.data)
  train_dataset, test_dataset = dataset.get_dataset(args.dataset_name, args.crop_size)
  train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, drop_last=True, pin_memory=False)
  test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=True, pin_memory=False)

  if(args.dataset_name == 'cifar10' or args.dataset_name == 'stl10' or args.dataset_name == 'mnist'): args.n_classes = 10
  elif(args.dataset_name == 'cifar100'): args.n_classes = 100
  elif(args.dataset_name == 'imagenet'): args.n_classes = 1000
  elif(args.dataset_name == 'stanfordcars'): args.n_classes = 196
  elif(args.dataset_name == 'oxfordiiitpet'): args.n_classes = 37
  elif(args.dataset_name == 'flowers102'): args.n_classes = 102
  else: args.n_classes = None

  model = timm.create_model(args.arch, pretrained=False, num_classes=args.n_classes).to(args.device)

  criterion = torch.nn.CrossEntropyLoss().to(args.device)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  auroc = MulticlassAUROC(args.n_classes, 'weighted').to(args.device)
  acc_1 = MulticlassAccuracy(args.n_classes, top_k=1, average='weighted').to(args.device)
  acc_5 = MulticlassAccuracy(args.n_classes, top_k=5, average='weighted').to(args.device)
  
  if int(args.delta) == 1: normal_train(model, train_loader, test_loader, criterion, optimizer, auroc, acc_1, acc_5)
  else: turbo_train(model, train_loader, test_loader, criterion, optimizer, auroc, acc_1, acc_5)