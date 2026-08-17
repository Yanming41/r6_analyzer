import warnings, logging, time, sys
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import torch
import numpy as np
from pathlib import Path

def progress(msg):
    print(msg, flush=True)

progress("=== R6 Model Diagnostic ===\n")

# ── Load model ───────────────────────────────────────────────────────────────
progress("[1/4] Loading processor...")
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
progress("[1/4] Processor loaded.")

progress("[2/4] Loading model weights into GPU...")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)
model.eval()
vram = torch.cuda.memory_allocated() / 1e9
ctx = getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", 4096)
progress(f"[2/4] Model loaded. VRAM: {vram:.1f}GB  Context window: {ctx} tokens\n")

# ── Token count test (no GPU inference) ──────────────────────────────────────
progress("[3/4] Token count test (no inference, fast)...")
conv = [{"role":"user","content":[{"type":"video"},{"type":"text","text":"describe"}]}]
prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
for n in [8, 16, 32, 64]:
    frames = np.zeros((n, 360, 640, 3), dtype=np.uint8)
    inp = processor(text=prompt, videos=frames, return_tensors="pt")
    total = inp["input_ids"].shape[1]
    flag = " *** NEAR LIMIT" if total > ctx * 0.85 else ""
    progress(f"    {n:>2} frames -> {total:>5} tokens / {ctx} context{flag}")
progress("")

# ── Real video inference with 8 frames ───────────────────────────────────────
seg = Path("output_real/segments/segment_0000.mp4")
if not seg.exists():
    progress("No segment found, skipping inference test.")
    sys.exit(0)

progress("[4/4] Extracting 8 frames from real gameplay segment...")
try:
    from decord import VideoReader, cpu
    vr = VideoReader(str(seg), ctx=cpu(0))
    n_total = len(vr)
    idxs = [int(i * n_total / 8) for i in range(8)]
    frames = vr.get_batch(idxs).asnumpy()
except Exception:
    import cv2
    cap = cv2.VideoCapture(str(seg))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_list = []
    for i in range(8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * n_total / 8))
        ok, f = cap.read()
        if ok:
            frames_list.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.array(frames_list)

progress(f"    Got {len(frames)} frames from {n_total}-frame video")

PROMPT = (
    "This video shows a first-person shooter game. "
    "Describe exactly what you see: the game name if you recognize it, "
    "the HUD elements (health, ammo, minimap), any player actions, "
    "kills, deaths, or notable events. Be specific."
)
conv2 = [{"role":"user","content":[{"type":"video"},{"type":"text","text":PROMPT}]}]
prompt2 = processor.apply_chat_template(conv2, add_generation_prompt=True)
inputs = processor(text=prompt2, videos=frames, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
n_tokens = inputs["input_ids"].shape[1]
progress(f"    Input tokens: {n_tokens}  (context: {ctx})")
progress("    Running inference... (this takes ~2-3 minutes on 7B model)")

t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=300, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id
    )
elapsed = time.perf_counter() - t0
response = processor.batch_decode(out[:, n_tokens:], skip_special_tokens=True)[0].strip()

progress(f"    Done in {elapsed:.0f}s\n")
progress("=== Model output ===")
progress(response)
progress("=== End ===")
