import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from .config import Config
from .models.txtctrlf_seg import TxtCtrlFSegmentation
from .models.inpainter import TxtCtrlFInpainter

class TxtCtrlFPipeline:
    def __init__(self, seg_checkpoint=None):
        self.device = Config.DEVICE
        
        self.seg_model = TxtCtrlFSegmentation(self.device).to(self.device).eval()
        ckpt = seg_checkpoint if seg_checkpoint else Config.STAGE1_BEST_PATH
        if os.path.exists(ckpt):
             self.seg_model.load_state_dict(torch.load(ckpt, map_location=self.device))
             print(f"Loaded Seg Model: {ckpt}")
        else:
             print(f"Warning: No checkpoint at {ckpt}")

        self.inpainter = TxtCtrlFInpainter(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_mask_pil(self, img_tensor, prompt):
        with torch.no_grad(): mask = self.seg_model(img_tensor, [prompt])
        mask_np = (mask.squeeze().cpu().numpy() > 0.5).astype('uint8') * 255
        return Image.fromarray(mask_np, mode="L")

    def run_full(self, img_pil, prompt):
        img_pil = img_pil.convert("RGB").resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        mask_pil = self.get_mask_pil(img_t, prompt)
        res_pil = self.inpainter.generate(img_pil, mask_pil, prompt)
        return {"original": img_pil, "mask": mask_pil, "result": res_pil}