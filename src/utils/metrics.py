import torch
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore

def compute_fid(real_images, fake_images, device='cuda'):
    """Tính FID. Input: Tensor [N, 3, H, W] range [0, 1]"""
    fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    fid.update(real_images.to(device), real=True)
    fid.update(fake_images.to(device), real=False)
    return fid.compute().item()

def compute_lpips(real_images, fake_images, device='cuda'):
    """Tính LPIPS. Input range [0, 1]"""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    score = lpips(fake_images.to(device), real_images.to(device))
    return score.item()

def compute_clip_score(fake_images, prompts, device='cuda'):
    """Tính CLIP Score. fake_images [0, 1]"""
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    fake_imgs_uint8 = (fake_images * 255).to(torch.uint8)
    score = clip(fake_imgs_uint8.to(device), prompts)
    return score.item()

def compute_iou(pred_mask, true_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).float()
    intersection = (pred_bin * true_mask).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def compute_dice(pred_mask, true_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).float()
    intersection = (pred_bin * true_mask).sum(dim=(1, 2, 3))
    return ((2. * intersection + 1e-6) / (pred_bin.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) + 1e-6)).mean().item()

def evaluate(pipeline, dataloader, device, max_samples=None):
    
    all_real = []
    all_fake = []
    all_prompts = []
    count = 0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for imgs, _, texts in tqdm(dataloader, desc="Generating Images"):
        imgs = imgs.to(device)
        
        for i in range(len(imgs)):
            real_t = torch.clamp(imgs[i] * std + mean, 0, 1)
            mask_pil = pipeline.get_mask_pil(imgs[i].unsqueeze(0), texts[i])
    
            real_pil = ToPILImage()(real_t.cpu())
            fake_pil = pipeline.inpainter.generate(real_pil, mask_pil, texts[i])
            
            all_real.append(real_t.cpu())
            all_fake.append(ToTensor()(fake_pil).cpu())
            all_prompts.append(texts[i])
        
        count += len(imgs)
        if max_samples and count >= max_samples:
            break

    full_real = torch.stack(all_real)
    full_fake = torch.stack(all_fake)
    
    fid_score = compute_fid(full_real, full_fake, device)
    lpips_score = compute_lpips(full_real, full_fake, device)
    clip_s = compute_clip_score(full_fake, all_prompts, device)
    
    return {
        "fid": fid_score,
        "lpips": lpips_score,
        "clip": clip_s
    }