import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DFMMDataset(Dataset):
    def __init__(self, data_root, ann_file, image_size=512):
        self.data_root = data_root
        self.image_size = image_size
        with open(ann_file, 'r') as f:
            self.anns = [json.loads(line) for line in f]
        
        # Transform ảnh Input (Normalize ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform Mask (Giữ nguyên 0-1)
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = self.anns[idx]
        
        # --- SỬA CÁC KEY Ở ĐÂY ---
        img_name = item['image']      # Cũ: item['image_name']
        mask_name = item['mask']      # Cũ: item['mask_name']
        text_prompt = item['text']    # Cũ: item['text_prompt']
        
        # Logic tìm ảnh
        img_path = os.path.join(self.data_root, "train_images", img_name)
        if not os.path.exists(img_path):
             img_path = os.path.join(self.data_root, "test_images", img_name)

        mask_path = os.path.join(self.data_root, "mask", mask_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            t_image = self.transform(image)
            t_mask = self.mask_transform(mask)
            t_mask = (t_mask > 0.5).float() # Binarize
            
            return t_image, t_mask, text_prompt
        except Exception as e:
            # print(f"Error loading {img_path}: {e}") # Uncomment để debug nếu cần
            return self.__getitem__((idx + 1) % len(self))