import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from ..config import Config

class TxtCtrlFInpainter:
    def __init__(self, device):
        self.device = device
        self.controlnet = ControlNetModel.from_pretrained(Config.CONTROLNET_ID, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            Config.SD_INPAINT_ID, controlnet=self.controlnet, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        if "cuda" in device: self.pipe.enable_model_cpu_offload()

    def generate(self, img_pil, mask_pil, prompt):
        return self.pipe(
            prompt=prompt,
            negative_prompt="low quality, bad anatomy, blurry",
            image=img_pil,
            mask_image=mask_pil,
            control_image=mask_pil,
            height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE,
            num_inference_steps=Config.NUM_INFERENCE_STEPS,
            guidance_scale=Config.GUIDANCE_SCALE
        ).images[0]