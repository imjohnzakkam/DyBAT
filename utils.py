import os
import sys
import shutil
import time

import torch
import yaml

def load_checkpoint(model, filepath, device):
    if(os.path.exists(filepath)):
        ckpt = torch.load(filepath, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        epoch = ckpt['epoch']
        return model, epoch
    else:
        raise FileNotFoundError('ckpt file not found')

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/'.join(filename.split('/')[:-1]) + '/model_best.pth')

def save_config_file(model_checkpoints_folder, filename, args):
    os.makedirs(model_checkpoints_folder, exist_ok=True)
    # os.makedirs(os.path.join(model_checkpoints_folder, 'checkpoints'), exist_ok=True)
    with open(os.path.join(model_checkpoints_folder, filename), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def metrics(model, loader, criterion, auroc, acc_1, acc_5, device):
  with torch.no_grad():
    loss, top_1, top_5, auc = [], [], [], []
    for _, data in enumerate(loader, 0):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)

      batch_loss = criterion(outputs, labels)
      loss.append(batch_loss)

    #   top1, top5 = accuracy(outputs, labels, topk=(1, 5))
      top1, top5 = acc_1(outputs, labels), acc_5(outputs, labels)
      top_1.append(top1 * 100)
      top_5.append(top5 * 100)
      
      batch_auc = auroc(outputs, labels)
      auc.append(batch_auc)

    avg_loss = sum(loss) / len(loss)
    avg_top_1 = sum(top_1) / len(top_1)
    avg_top_5 = sum(top_5) / len(top_5)
    avg_auc = sum(auc) / len(auc)

    return avg_loss, avg_top_1, avg_top_5, avg_auc

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f