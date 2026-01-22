import torch
import matplotlib.pyplot as plt

def visualize_batch(images, masks_gt, masks_pred, epoch, batch_idx):
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    batch_size = min(images.shape[0], 3)
    fig, axs = plt.subplots(batch_size, 3, figsize=(10, 3*batch_size))
    for i in range(batch_size):
        t = images[i].clone().cpu()
        for c in range(3): t[c] = t[c] * STD[c] + MEAN[c]
        img = torch.clamp(t, 0, 1).permute(1, 2, 0).numpy()
        ax = axs[i] if batch_size > 1 else axs
        ax[0].imshow(img); ax[0].axis('off')
        ax[1].imshow(masks_gt[i].squeeze().cpu().numpy(), cmap='gray'); ax[1].axis('off')
        ax[2].imshow(masks_pred[i].squeeze().detach().cpu().numpy(), cmap='gray'); ax[2].axis('off')
    plt.tight_layout()
    return fig