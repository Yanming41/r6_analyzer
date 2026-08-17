import warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import torch
import numpy as np
from transformers import LlavaNextVideoProcessor

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)

print("=== Processor config ===")
print(f"  num_frames attr: {getattr(processor, 'num_frames', 'NOT SET')}")
cfg = processor.image_processor
for attr in ["num_frames", "max_num_frames", "temporal_patch_size", "patch_size", "size"]:
    if hasattr(cfg, attr):
        print(f"  image_processor.{attr} = {getattr(cfg, attr)}")

# Test: pass 64 frames, see what shape goes to model
for n_frames in [8, 16, 32, 64]:
    dummy = np.zeros((n_frames, 360, 640, 3), dtype=np.uint8)
    conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "test"}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=dummy, return_tensors="pt")
    pv = inputs.get("pixel_values_videos", inputs.get("pixel_values", None))
    if pv is not None:
        print(f"\n  Input {n_frames} frames -> pixel_values shape: {pv.shape}")
        # shape is usually (1, num_frames, C, H, W) or (1, num_frames*patches, C, h, w)
    else:
        keys = list(inputs.keys())
        print(f"\n  Input {n_frames} frames -> keys: {keys}")
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {v.shape}")
