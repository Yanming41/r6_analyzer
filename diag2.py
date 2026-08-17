"""
Diagnostic: measure actual token usage and test whether the model understands
real R6 gameplay footage.
"""
import warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import torch
import numpy as np
from pathlib import Path
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

print("Loading model...")
processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)
model.eval()
ctx_window = getattr(
    model.config, "max_position_embeddings",
    getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 4096)
)
print(f"Model context window: {ctx_window} tokens\n")

# --- Test 1: token count at different frame counts ---
print("=== Test 1: input token count vs frame count ===")
conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "describe"}]}]
prompt = processor.apply_chat_template(conv, add_generation_prompt=True)

for n in [8, 16, 32, 64]:
    frames = np.zeros((n, 360, 640, 3), dtype=np.uint8)
    inp = processor(text=prompt, videos=frames, return_tensors="pt")
    total = inp["input_ids"].shape[1]
    overflow = "  << OVERFLOW" if total > ctx_window * 0.9 else ""
    print(f"  {n:>2} frames -> {total} tokens  (context: {ctx_window}){overflow}")

# --- Test 2: can the model understand real gameplay? ---
seg_path = Path("output_real/segments/segment_0000.mp4")
if not seg_path.exists():
    seg_path = Path("output/segments/segment_0000.mp4")

if not seg_path.exists():
    print("\nNo segment found, skipping test 2")
else:
    print(f"\n=== Test 2: real segment {seg_path.name}, 8 frames ===")
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(seg_path), ctx=cpu(0))
        total_frames = len(vr)
        idxs = [int(i * total_frames / 8) for i in range(8)]
        frames = vr.get_batch(idxs).asnumpy()
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(seg_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_list = []
        for i in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total_frames / 8))
            ok, f = cap.read()
            if ok:
                frames_list.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.array(frames_list)

    print(f"  Extracted {len(frames)} frames from {total_frames}-frame video")

    SIMPLE = "Describe in detail what you see in this video. What game is this? What is happening? List any game events you notice."

    conv2 = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": SIMPLE}]}]
    prompt2 = processor.apply_chat_template(conv2, add_generation_prompt=True)
    inputs = processor(text=prompt2, videos=frames, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"  Input tokens: {inputs['input_ids'].shape[1]}")

    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=400, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0
    new_ids = out[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    print(f"  Inference time: {elapsed:.1f}s")
    print("\n--- Model response ---")
    print(response)
    print("---")

    # Test 3: same frames but with R6-specific prompt
    print("\n=== Test 3: same frames, R6-specific prompt ===")
    R6_SPECIFIC = """This is a Rainbow Six Siege gameplay video.
Look at the kill feed (top-right corner), health bar, operator HUD, and round timer.
List every game event you can identify with a timestamp [MM:SS]:
- Any kills shown in the kill feed
- Any deaths or eliminations
- Round start or end
- Any operator abilities used"""

    conv3 = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": R6_SPECIFIC}]}]
    prompt3 = processor.apply_chat_template(conv3, add_generation_prompt=True)
    inputs3 = processor(text=prompt3, videos=frames, return_tensors="pt")
    inputs3 = {k: v.to(model.device) for k, v in inputs3.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        out3 = model.generate(
            **inputs3, max_new_tokens=400, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0
    new3 = out3[:, inputs3["input_ids"].shape[1]:]
    response3 = processor.batch_decode(new3, skip_special_tokens=True)[0].strip()

    print(f"  Inference time: {elapsed:.1f}s")
    print("\n--- Model response (R6 prompt) ---")
    print(response3)
    print("---")
