import torch
import os

class Config:
    # --- PATHS ---
    DATA_ROOT = "/kaggle/input/dfmm-spotlight" 
    TRAIN_ANN = os.path.join(DATA_ROOT, "mask_ann/train_ann_file.jsonl")
    TEST_ANN = os.path.join(DATA_ROOT, "mask_ann/test_ann_file.jsonl")
    
    CHECKPOINT_DIR = "../checkpoints"
    STAGE1_BEST_PATH = os.path.join(CHECKPOINT_DIR, "txtctrlf_seg_best.pth")
    
    # --- SYSTEM ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    # --- PARAMS ---
    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 30
    
    CONTROLNET_ID = "lllyasviel/sd-controlnet-seg"
    SD_INPAINT_ID = "runwayml/stable-diffusion-inpainting"
    GUIDANCE_SCALE = 7.5
    NUM_INFERENCE_STEPS = 30