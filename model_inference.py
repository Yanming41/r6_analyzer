"""
Model inference: load a LLaVA-Video-compatible model and analyse a video segment
to produce a list of R6 game-event timestamps.

Supported model IDs (standard transformers, no custom packages needed):
  llava-hf/LLaVA-NeXT-Video-7B-hf        ← default, stable HF build
  llava-hf/LLaVA-NeXT-Video-34B-hf       ← larger, higher accuracy
  llava-hf/llava-onevision-qwen2-7b-ov-hf ← LLaVA-OneVision architecture

Note on lmms-lab/LLaVA-Video-7B-Qwen2:
  That model uses a private 'LlavaQwenForCausalLM' architecture absent from
  standard transformers.  To use it you would need:
    pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
  which may conflict with transformers>=5.  The llava-hf models above are
  official HF conversions of the same architecture family.
"""
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

R6_PROMPT = """\
You are a Rainbow Six Siege (R6S) gameplay analyst. Watch this video clip and \
report everything notable that happens on screen.

For every notable moment output ONE line in this exact format:
[MM:SS] - TAG - description (under 15 words)

TAG rules — use the exact tags below when applicable, otherwise invent a short \
descriptive tag:
  KILL       ← MANDATORY: report every single enemy elimination you see
  DEATH      ← player character is eliminated
  MULTI_KILL ← 2+ eliminations in rapid succession
  CLUTCH     ← outnumbered player clutches the round
  ROUND_START / ROUND_END ← round phase transitions
  For anything else use a fitting tag, e.g.:
    DRONE, BREACH, PLANT, DEFUSE, ABILITY, GADGET, RELOAD,
    REPOSITION, HEADSHOT_FEED, DAMAGE, CALLOUT, etc.

General rules:
  • Timestamps MUST be in MM:SS format (e.g. 00:15, 02:43, 09:07)
  • List in chronological order
  • KILL events are mandatory — never skip an elimination
  • Also describe any other gameplay moment worth noting
  • Be specific: name the location, weapon, or operator if visible

If nothing notable happens in the clip, output: NO_EVENTS_DETECTED

Analysis:"""

# ── Regex parser ──────────────────────────────────────────────────────────────

# Matches any TAG the model invents, not just the fixed list
_EVENT_RE = re.compile(
    r"\[?(\d{1,2}:\d{2})\]?\s*[-–]\s*"
    r"([A-Z][A-Z0-9_]{1,20})\s*[-–]\s*(.+)",
    re.IGNORECASE,
)


def _mmss_to_seconds(ts: str) -> float:
    m, s = ts.strip().split(":")
    return int(m) * 60 + int(s)


def _seconds_to_mmss(secs: float) -> str:
    return f"{int(secs // 60):02d}:{int(secs % 60):02d}"


# ── Model loading strategies ──────────────────────────────────────────────────

def _load_nextvideo(model_id: str, load_kw: dict):
    """LlavaNextVideo – llava-hf/LLaVA-NeXT-Video-* models."""
    from transformers import (  # noqa: PLC0415
        LlavaNextVideoForConditionalGeneration,
        LlavaNextVideoProcessor,
    )
    logger.info("Architecture: LlavaNextVideo")
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_id, **load_kw)
    return model, processor, "nextvideo"


def _load_onevision(model_id: str, load_kw: dict):
    """LlavaOnevision – llava-hf/llava-onevision-* models."""
    from transformers import (  # noqa: PLC0415
        LlavaOnevisionForConditionalGeneration,
        LlavaOnevisionProcessor,
    )
    logger.info("Architecture: LlavaOnevision")
    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **load_kw)
    return model, processor, "onevision"


def _load_llava_package(model_id: str, load_kw: dict):
    """
    lmms-lab LLaVA custom package (LlavaQwenForCausalLM).
    Requires:  pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    """
    from llava.model.builder import load_pretrained_model  # noqa: PLC0415

    logger.info("Architecture: LlavaQwen (lmms-lab llava package)")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name="llava_qwen",
        device_map=load_kw.get("device_map", "auto"),
    )
    # Wrap into a pseudo-processor so the rest of the code is uniform
    processor = _LlavaPackageProcessor(tokenizer, image_processor)
    return model, processor, "llava_pkg"


# Thin wrapper so llava-package models share the same inference interface
class _LlavaPackageProcessor:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor


# ── Main class ────────────────────────────────────────────────────────────────

class ModelInference:
    """Wraps LLaVA-Video inference with OOM-aware retry logic."""

    def __init__(
        self,
        model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        fps: float = 1.0,
        max_frames: int = 64,
        quantize: bool = False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. This pipeline requires an NVIDIA GPU.\n"
                "Verify drivers with: nvidia-smi\n"
                "Install PyTorch CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

        self.model_id = model_id
        self.fps = fps
        self.max_frames = max_frames
        self.quantize = quantize
        self.model = None
        self.processor = None
        self._arch: Optional[str] = None

        self._load()

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self):
        dev = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {dev}  |  VRAM: {vram_gb:.1f} GB")
        logger.info(f"Loading model: {self.model_id}")

        # transformers 5.x uses 'dtype'; 4.x used 'torch_dtype'
        load_kw: dict = {
            "dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if self.quantize:
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kw.pop("dtype", None)
                load_kw["quantization_config"] = bnb
                logger.info("4-bit quantization enabled (bitsandbytes)")
            except ImportError:
                logger.warning("bitsandbytes not available – using fp16 instead of 4-bit")

        # Pick strategy order based on model ID
        mid_lower = self.model_id.lower()
        if "lmms-lab" in mid_lower or "llava_qwen" in mid_lower:
            strategies = [_load_llava_package, _load_onevision, _load_nextvideo]
        elif "onevision" in mid_lower:
            strategies = [_load_onevision, _load_nextvideo]
        else:
            # llava-hf/LLaVA-NeXT-Video-* and anything else
            strategies = [_load_nextvideo, _load_onevision]

        last_exc: Optional[Exception] = None
        for strategy in strategies:
            try:
                self.model, self.processor, self._arch = strategy(self.model_id, load_kw)
                break
            except Exception as exc:
                logger.debug(f"{strategy.__name__} failed: {exc}")
                last_exc = exc
        else:
            raise RuntimeError(
                f"All loading strategies failed for {self.model_id}.\n"
                f"Last error: {last_exc}\n\n"
                "Suggested fixes:\n"
                "  - For lmms-lab/LLaVA-Video-7B-Qwen2, install the custom package:\n"
                "      pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git\n"
                "    (may require downgrading transformers to 4.x)\n"
                "  - Or switch to an HF-native model:\n"
                "      --model llava-hf/LLaVA-NeXT-Video-7B-hf"
            )

        if hasattr(self.model, "eval"):
            self.model.eval()

        used_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Model ready  |  arch={self._arch}  |  VRAM used: {used_gb:.1f} GB")

    # ── frame extraction ──────────────────────────────────────────────────────

    def _extract_frames(
        self, video_path: Path, fps: float, max_frames: int
    ) -> Tuple[np.ndarray, float]:
        try:
            return self._frames_decord(video_path, fps, max_frames)
        except Exception as exc:
            logger.debug(f"decord failed ({exc}), trying cv2")
            return self._frames_cv2(video_path, fps, max_frames)

    def _frames_decord(
        self, video_path: Path, fps: float, max_frames: int
    ) -> Tuple[np.ndarray, float]:
        from decord import VideoReader, cpu  # noqa: PLC0415

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = len(vr)
        vfps = float(vr.get_avg_fps())
        duration = total / vfps

        interval = max(1, round(vfps / fps))
        indices = list(range(0, total, interval))
        if len(indices) > max_frames:
            step = max(1, len(indices) // max_frames)
            indices = indices[::step][:max_frames]

        logger.debug(f"decord: {len(indices)} frames, duration={duration:.1f}s")
        return vr.get_batch(indices).asnumpy(), duration

    def _frames_cv2(
        self, video_path: Path, fps: float, max_frames: int
    ) -> Tuple[np.ndarray, float]:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open: {video_path}")

        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / vfps
        interval = max(1, round(vfps / fps))

        frames, idx = [], 0
        while cap.isOpened() and len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()

        logger.debug(f"cv2: {len(frames)} frames, duration={duration:.1f}s")
        return np.array(frames, dtype=np.uint8), duration

    # ── inference ─────────────────────────────────────────────────────────────

    def _make_inputs_nextvideo(self, frames: np.ndarray) -> dict:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": R6_PROMPT},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        # LlavaNextVideoProcessor expects numpy (N, H, W, 3) directly
        inputs = self.processor(
            text=prompt,
            videos=frames,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _make_inputs_onevision(self, frames: np.ndarray) -> dict:
        pil_frames = [Image.fromarray(f) for f in frames]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": R6_PROMPT},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            videos=[pil_frames],
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _make_inputs_llava_pkg(self, frames: np.ndarray) -> dict:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: PLC0415
        from llava.conversation import conv_templates  # noqa: PLC0415
        from llava.mm_utils import tokenizer_image_token  # noqa: PLC0415

        pil_frames = [Image.fromarray(f) for f in frames]
        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor

        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.human, DEFAULT_IMAGE_TOKEN + "\n" + R6_PROMPT)
        conv.append_message(conv.gpt, None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        video_tensor = image_processor.preprocess(
            pil_frames, return_tensors="pt"
        )["pixel_values"].to(self.model.device, dtype=torch.float16)

        return {"input_ids": input_ids, "_video_tensor": video_tensor}

    def _generate(self, inputs: dict) -> str:
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None) or \
                 getattr(self.processor.tokenizer, "eos_token_id", None)

        if self._arch == "llava_pkg":
            video_tensor = inputs.pop("_video_tensor")
            with torch.no_grad():
                out_ids = self.model.generate(
                    inputs["input_ids"],
                    images=[video_tensor],
                    modalities=["video"],
                    do_sample=False,
                    max_new_tokens=512,
                )
            prompt_len = inputs["input_ids"].shape[1]
            tokenizer = self.processor.tokenizer
            return tokenizer.batch_decode(
                out_ids[:, prompt_len:], skip_special_tokens=True
            )[0].strip()
        else:
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(
                new_ids, skip_special_tokens=True
            )[0].strip()

    def _make_inputs(self, frames: np.ndarray) -> dict:
        if self._arch == "nextvideo":
            return self._make_inputs_nextvideo(frames)
        elif self._arch == "onevision":
            return self._make_inputs_onevision(frames)
        elif self._arch == "llava_pkg":
            return self._make_inputs_llava_pkg(frames)
        raise RuntimeError(f"Unknown arch: {self._arch}")

    # ── parsing ───────────────────────────────────────────────────────────────

    def _parse(self, text: str, duration: float) -> List[Dict]:
        if "NO_EVENTS_DETECTED" in text.upper():
            return []
        events = []
        for line in text.splitlines():
            m = _EVENT_RE.search(line)
            if m:
                ts_str, etype, desc = m.groups()
                secs = _mmss_to_seconds(ts_str)
                if secs <= duration + 10:
                    events.append(
                        {
                            "timestamp": secs,
                            "time_str": _seconds_to_mmss(secs),
                            "event_type": etype.upper(),
                            "description": desc.strip(),
                        }
                    )
        return events

    # ── public API ────────────────────────────────────────────────────────────

    def analyze_segment(self, video_path: Path) -> Optional[List[Dict]]:
        """
        Analyse one video segment. Retries with halved fps/max_frames on CUDA OOM.
        Returns list of detected events, or None if all attempts failed.
        """
        fps = self.fps
        max_frames = self.max_frames

        for attempt in range(3):
            try:
                if attempt > 0:
                    logger.info(f"  OOM retry {attempt}: fps={fps:.2f}, max_frames={max_frames}")

                frames, duration = self._extract_frames(video_path, fps, max_frames)
                if frames.size == 0:
                    logger.error("No frames extracted from segment")
                    return None

                logger.debug(
                    f"  {len(frames)} frames  |  duration={duration:.1f}s  |  "
                    f"shape={frames.shape}"
                )

                t0 = time.perf_counter()
                inputs = self._make_inputs(frames)
                response = self._generate(inputs)
                elapsed = time.perf_counter() - t0

                logger.info(f"  Inference: {elapsed:.1f}s")
                logger.debug(f"  Model output:\n{response}")

                events = self._parse(response, duration)
                logger.info(f"  Events found: {len(events)}")
                return events

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fps /= 2.0
                max_frames //= 2
                logger.warning(
                    f"CUDA OOM (attempt {attempt + 1}). "
                    f"Retrying with fps={fps:.2f}, max_frames={max_frames}"
                )
                if max_frames < 4:
                    logger.error("Cannot reduce further. Segment skipped.")
                    return None

            except Exception as exc:
                logger.error(
                    f"Inference error (attempt {attempt + 1}): {exc}", exc_info=True
                )
                return None

        return None
