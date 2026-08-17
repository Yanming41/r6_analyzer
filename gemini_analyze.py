"""
R6 Siege highlight analyzer via Gemini video understanding.

Usage:
    # 第一次：上传并分析
    python gemini_analyze.py --video "D:/videos/gameplay.mp4" --api_key "AIza..."

    # 复用已上传的文件（跳过上传，直接分析）
    python gemini_analyze.py --file_id "files/8sbmwbcu6qor" --video "D:/videos/gameplay.mp4" --api_key "AIza..."

    # 只列出已上传的文件
    python gemini_analyze.py --list_files --api_key "AIza..."
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Patch httpx to accept non-ASCII header values (google-genai 2.6.0 bug workaround)
import httpx._models as _hm
_orig_nhv = _hm._normalize_header_value
def _patched_nhv(value, encoding=None):
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Header value must be str or bytes, not {type(value)}")
    try:
        return value.encode("ascii")
    except UnicodeEncodeError:
        # Print which value is causing the issue (for debugging), then encode as latin-1
        print(f"[DEBUG] Non-ASCII header value: {repr(value[:50])}", flush=True)
        return value.encode("utf-8", errors="replace")
_hm._normalize_header_value = _patched_nhv

PROMPT = """你是一名 Rainbow Six Siege（R6S）职业选手解说兼视频剪辑师。

仔细观看这段完整的 R6 游戏录像，找出所有值得剪辑的精彩时刻。

每个事件输出一行，格式严格如下：
[MM:SS] - 标签 - 描述（15字以内）

标签说明（可自由扩展，不限于此列表）：
  KILL        每次击杀都必须报告，不能漏掉任何一个
  MULTI_KILL  连续击杀2人或以上
  CLUTCH      以少打多赢得回合（如1v2、1v3）
  DEATH       玩家被淘汰
  ROUND_START 回合开始
  ROUND_END   回合结束（标注胜负）
  HIGHLIGHT   其他精彩时刻：妙投手雷、极限救援、绕后包抄等
  ABILITY     运营商技能的关键使用
  BREACH      重要的墙壁/地板破坏

其他要求：
- 时间戳必须是 MM:SS 格式（如 02:15、09:47）
- 按时间升序排列
- KILL 事件不能遗漏
- 对于 CLUTCH / MULTI_KILL 等高光时刻，描述要说清楚背景（几打几、用什么武器）
- 如果有回合数，注明是第几回合

直接输出事件列表，不需要其他说明。"""


def mmss_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def parse_events(text: str) -> list:
    pattern = re.compile(
        r"\[?(\d{1,2}:\d{2})\]?\s*[-–]\s*([A-Z][A-Z0-9_]{1,20})\s*[-–]\s*(.+)",
        re.IGNORECASE,
    )
    events = []
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            ts, tag, desc = m.groups()
            events.append({
                "timestamp": mmss_to_seconds(ts),
                "time_str": ts.strip(),
                "tag": tag.upper(),
                "description": desc.strip(),
            })
    return events


def list_uploaded_files(api_key: str) -> None:
    from google import genai
    client = genai.Client(api_key=api_key)
    print("已上传到 Gemini 的文件：")
    print("-" * 60)
    count = 0
    for f in client.files.list():
        state = getattr(f.state, "name", str(f.state))
        expire = getattr(f, "expiration_time", "unknown")
        print(f"  ID: {f.name}")
        print(f"  名称: {f.display_name}")
        print(f"  状态: {state}  到期: {expire}")
        print()
        count += 1
    if count == 0:
        print("  （没有已上传的文件）")
    print("-" * 60)


def analyze(video_path: str, api_key: str, file_id: str = None) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # ── 1. 获取文件信息（上传或复用）────────────────────────────────────────────
    if file_id:
        print(f"复用已上传文件：{file_id}")
        file_info = client.files.get(name=file_id)
        state = file_info.state.name
        if state == "FAILED":
            print("错误：该文件状态为 FAILED，请重新上传")
            sys.exit(1)
        elif state == "PROCESSING":
            print("文件仍在处理中，等待 …")
            while True:
                file_info = client.files.get(name=file_id)
                state = file_info.state.name
                if state == "ACTIVE":
                    break
                elif state == "FAILED":
                    print("错误：视频处理失败")
                    sys.exit(1)
                print(f"  状态：{state}，等待中 …")
                time.sleep(5)
        print(f"文件就绪：{file_info.display_name}")
        video_name = file_info.display_name or file_id
        uploaded_name = None  # 不清理复用的文件
    else:
        video_path_obj = Path(video_path)
        size_gb = video_path_obj.stat().st_size / 1e9
        print(f"视频：{video_path_obj.name}  ({size_gb:.2f} GB)")
        print("正在上传到 Gemini Files API …（首次约需 1-3 分钟）")

        with open(video_path_obj, "rb") as f:
            uploaded = client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    mime_type="video/mp4",
                    display_name=video_path_obj.name,
                ),
            )

        print(f"上传完成，文件 ID：{uploaded.name}")
        print(f"下次可用 --file_id {uploaded.name} 跳过上传")

        print("Gemini 正在处理视频 …")
        while True:
            file_info = client.files.get(name=uploaded.name)
            state = file_info.state.name
            if state == "ACTIVE":
                print("视频处理完成，开始分析 …")
                break
            elif state == "FAILED":
                print("错误：视频处理失败")
                sys.exit(1)
            else:
                print(f"  状态：{state}，等待中 …")
                time.sleep(5)

        video_name = video_path_obj.name
        uploaded_name = uploaded.name

    # ── 2. 调用 Gemini 分析 ────────────────────────────────────────────────────
    print("正在让 Gemini 识别亮点事件 …")
    t0 = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=file_info.uri, mime_type="video/mp4"),
                    types.Part.from_text(text=PROMPT),
                ],
            )
        ],
    )

    elapsed = time.perf_counter() - t0
    raw_text = response.text
    print(f"分析完成，耗时 {elapsed:.1f}s\n")

    # ── 3. 解析 + 输出 ─────────────────────────────────────────────────────────
    events = parse_events(raw_text)

    print("=" * 64)
    print(f"  R6 亮点分析：{video_name}")
    print(f"  共识别事件：{len(events)} 个")
    print("=" * 64)
    print()

    if events:
        for ev in events:
            print(f"  [{ev['time_str']}]  {ev['tag']:<14}  {ev['description']}")
    else:
        print("  模型原始输出（未能解析为结构化事件）：")
        print()
        for line in raw_text.splitlines()[:60]:
            print(f"  {line}")

    # ── 4. 保存结果 ────────────────────────────────────────────────────────────
    out_dir = Path("output_gemini")
    out_dir.mkdir(exist_ok=True)
    stem = Path(video_name).stem

    json_path = out_dir / f"{stem}_highlights.json"
    json_path.write_text(
        json.dumps({
            "video": video_name,
            "model": "gemini-2.0-flash",
            "analysis_date": datetime.now().isoformat(timespec="seconds"),
            "total_events": len(events),
            "events": events,
            "raw_output": raw_text,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt_path = out_dir / f"{stem}_highlights.txt"
    lines = [
        f"R6 亮点分析 - {video_name}",
        f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"识别事件：{len(events)} 个",
        "",
        "=" * 56,
        "事件时间线",
        "=" * 56,
    ]
    for ev in events:
        lines.append(f"[{ev['time_str']}]  {ev['tag']:<14}  {ev['description']}")
    lines += ["", "=" * 56, "模型原始输出", "=" * 56, raw_text]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"  JSON → {json_path}")
    print(f"  TXT  → {txt_path}")
    print("=" * 64)

    # ── 5. 清理（仅新上传的文件）─────────────────────────────────────────────
    if uploaded_name:
        try:
            client.files.delete(name=uploaded_name)
            print("  已清理 Gemini 端缓存文件")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Gemini R6 亮点分析")
    ap.add_argument("--video", default="", help="视频文件路径（使用 --file_id 时可省略）")
    ap.add_argument("--file_id", default="", help="已上传文件的 ID（如 files/8sbmwbcu6qor），跳过上传")
    ap.add_argument("--list_files", action="store_true", help="列出所有已上传到 Gemini 的文件")
    ap.add_argument("--api_key", default="", help="Gemini API Key（或设置 GEMINI_API_KEY 环境变量）")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("错误：需要提供 Gemini API Key")
        print("  方式1：--api_key AIza...")
        print("  方式2：set GEMINI_API_KEY=AIza...")
        sys.exit(1)

    if args.list_files:
        list_uploaded_files(api_key)
        return

    if not args.file_id and not args.video:
        print("错误：需要 --video 或 --file_id 其中之一")
        sys.exit(1)

    if not args.file_id and not Path(args.video).exists():
        print(f"错误：视频文件不存在：{args.video}")
        sys.exit(1)

    analyze(args.video, api_key, file_id=args.file_id or None)


if __name__ == "__main__":
    main()
