import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPTextModel, CLIPTokenizer

class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim=512):
        super().__init__()
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))

    def forward(self, visual, text):
        b, c, h, w = visual.shape
        vis_flat = visual.view(b, c, -1).permute(0, 2, 1) 
        Q = self.vis_proj(vis_flat)
        K = V = self.text_proj(text)
        attn_out, _ = self.cross_attn(Q, K, V)
        x = self.norm(Q + attn_out)
        out = x + self.ffn(x)
        return out.permute(0, 2, 1).view(b, -1, h, w)

class TxtCtrlFSegmentation(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_enc = nn.Sequential(*list(resnet.children())[:-2])
        for p in self.visual_enc.parameters(): p.requires_grad = False
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.text_enc.parameters(): p.requires_grad = False
        
        self.fusion = CrossAttentionFusion(2048, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, img, text):
        img_f = self.visual_enc(img)
        tok = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad(): txt_f = self.text_enc(**tok).last_hidden_state
        mask = self.decoder(self.fusion(img_f, txt_f))
        if mask.shape[-2:] != img.shape[-2:]:
            mask = F.interpolate(mask, size=img.shape[-2:], mode='bilinear', align_corners=False)
        return mask