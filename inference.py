import os
import cv2
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────
# 1. IMAGE STITCHING  (patch_0 is always top-left)
# ─────────────────────────────────────────────

def right_score(img1, img2):
    """MSE between right edge of img1 and left edge of img2."""
    return np.mean((img1[:, -1].astype(np.float32) - img2[:, 0].astype(np.float32)) ** 2)

def bottom_score(img1, img2):
    """MSE between bottom edge of img1 and top edge of img2."""
    return np.mean((img1[-1, :].astype(np.float32) - img2[0, :].astype(np.float32)) ** 2)


def load_patches(patches_dir):
    files = [f for f in os.listdir(patches_dir) if f.endswith(".png")]
    files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    images = {f: cv2.imread(os.path.join(patches_dir, f)) for f in files}
    return files, images


def build_grid(files, images):
    """
    Greedy neighbour assignment.
    patch_0.png is guaranteed to be the top-left anchor.
    """
    anchor = "patch_0.png"
    remaining = [f for f in files if f != anchor]

    # For each patch find its best right-neighbour and best bottom-neighbour
    parent_right  = {}
    parent_bottom = {}

    all_files = files  # includes anchor
    for f1 in all_files:
        best_r, best_b = None, None
        best_rs, best_bs = float("inf"), float("inf")

        for f2 in all_files:
            if f1 == f2:
                continue
            rs = right_score(images[f1], images[f2])
            bs = bottom_score(images[f1], images[f2])
            if rs < best_rs:
                best_rs = rs;  best_r = f2
            if bs < best_bs:
                best_bs = bs;  best_b = f2

        parent_right[f1]  = best_r
        parent_bottom[f1] = best_b

    # Walk from anchor
    grid    = {}
    visited = set()

    row_head = anchor
    row_idx  = 0

    while row_head and row_head not in visited:
        col_idx = 0
        cur     = row_head

        while cur and cur not in visited:
            grid[cur] = (row_idx, col_idx)
            visited.add(cur)
            cur      = parent_right[cur]
            col_idx += 1

        row_head  = parent_bottom[row_head]
        row_idx  += 1

    # Place anything not yet visited
    for f in files:
        if f not in grid:
            grid[f] = (row_idx, 0)
            row_idx += 1

    return grid


def stitch_map(files, images, grid):
    """Assemble the full map image from the grid."""
    max_row = max(r for r, c in grid.values())
    max_col = max(c for r, c in grid.values())

    sample    = next(iter(images.values()))
    ph, pw    = sample.shape[:2]

    canvas = np.zeros(((max_row + 1) * ph, (max_col + 1) * pw, 3), dtype=np.uint8)

    for fname, (row, col) in grid.items():
        canvas[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw] = images[fname]

    return canvas


# ─────────────────────────────────────────────
# 2. VLM QUESTION ANSWERING  (offline, no internet)
# ─────────────────────────────────────────────

def load_vlm():
    """
    Load Qwen2-VL-7B-Instruct (downloaded by setup.bash into ./model_weights/).
    Falls back to answering 5 (skip) if model not available.
    """
    model_path = os.path.join(os.path.dirname(__file__), "model_weights", "Qwen2-VL-7B-Instruct")

    if not os.path.isdir(model_path):
        print(f"[WARN] Model not found at {model_path}. All answers will be 5 (unanswered).")
        return None, None, None

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        print("Loading VLM …")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print("VLM loaded.")
        return model, processor, "qwen2vl"

    except Exception as e:
        print(f"[WARN] Failed to load VLM: {e}. All answers will be 5.")
        return None, None, None


def ask_vlm(model, processor, model_type, map_pil, question, options):
    """
    Send the stitched map + question + options to the VLM.
    Returns an integer 1–4, or 5 if unsure.
    """
    import torch

    opt_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    prompt = (
        f"You are analysing a satellite/map image. "
        f"Answer the following multiple-choice question by replying ONLY with the number "
        f"1, 2, 3, or 4 that corresponds to the correct option. "
        f"If you are not confident, reply with 5.\n\n"
        f"Question: {question}\n"
        f"Options:\n{opt_text}\n\n"
        f"Answer (just the number):"
    )

    # Resize map for VLM input (keep it manageable)
    max_side = 1024
    w, h = map_pil.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        map_pil = map_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": map_pil},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8)

        generated = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        match = re.search(r'[1-5]', generated)
        if match:
            return int(match.group())
    except Exception as e:
        print(f"[WARN] VLM inference error: {e}")

    return 5   # safe fallback — no penalty


# ─────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────

def main(test_dir):
    patches_dir = os.path.join(test_dir, "patches")
    test_csv    = os.path.join(test_dir, "test.csv")

    # ── Stitch ──────────────────────────────
    print("Loading patches …")
    files, images = load_patches(patches_dir)
    print(f"  {len(files)} patches found.")

    print("Building grid …")
    grid = build_grid(files, images)

    print("Stitching map …")
    stitched_bgr = stitch_map(files, images, grid)
    stitched_rgb = cv2.cvtColor(stitched_bgr, cv2.COLOR_BGR2RGB)
    map_pil      = Image.fromarray(stitched_rgb)
    print(f"  Stitched map size: {map_pil.size}")

    # ── Load questions ───────────────────────
    test_df = pd.read_csv(test_csv)
    print(f"  {len(test_df)} questions to answer.")

    # ── Load VLM ────────────────────────────
    model, processor, model_type = load_vlm()

    # ── Answer questions ─────────────────────
    records = []
    for _, row in test_df.iterrows():
        qid      = row["id"]
        question = row["question"]
        options  = [row["option_1"], row["option_2"], row["option_3"], row["option_4"]]

        if model is not None:
            answer = ask_vlm(model, processor, model_type, map_pil, question, options)
        else:
            answer = 5   # no model → safe skip

        print(f"  {qid}: {answer}")
        records.append({"id": qid, "question_num": qid, "option": answer})

    # ── Save submission (in CWD, not test_dir) ──
    out_path = "submission.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {out_path} with {len(records)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True,
                        help="Absolute path to test directory containing test.csv and patches/")
    args = parser.parse_args()
    main(args.test_dir)
