import torch
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm
from .metrics import compute_iou, compute_dice
from .visualization import visualize_batch

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=0; self.avg=0; self.sum=0; self.count=0
    def update(self, val, n=1):
        self.val=val; self.sum+=val*n; self.count+=n; self.avg=self.sum/self.count

def criterion(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    inter = (pred * target).sum()
    dice = (2. * inter + 1.) / (pred.sum() + target.sum() + 1.)
    return bce + (1 - dice)

class Trainer:
    def __init__(self, model, loader, optimizer, config):
        self.model = model
        self.loader = loader
        self.optim = optimizer
        self.cfg = config
        self.history = {'loss': [], 'iou': []}
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def train_epoch(self, epoch, viz_fn=None):
        self.model.train()
        meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
        
        pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
        for idx, (img, mask, text) in enumerate(pbar):
            img, mask = img.to(self.cfg.DEVICE), mask.to(self.cfg.DEVICE)
            self.optim.zero_grad()
            pred = self.model(img, text)
            loss = criterion(pred, mask)
            loss.backward()
            self.optim.step()
            
            with torch.no_grad(): iou = compute_iou(pred, mask)
            meters['loss'].update(loss.item(), img.size(0))
            meters['iou'].update(iou, img.size(0))
            pbar.set_postfix({'Loss': f"{meters['loss'].avg:.3f}", 'IoU': f"{meters['iou'].avg:.3f}"})
            
            if viz_fn and idx % 50 == 0: viz_fn(visualize_batch(img, mask, pred, epoch, idx))
            
        self.history['loss'].append(meters['loss'].avg)
        torch.save(self.model.state_dict(), os.path.join(self.cfg.CHECKPOINT_DIR, f"txtctrlf_seg_ep{epoch}.pth"))