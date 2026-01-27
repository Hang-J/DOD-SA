import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation_script"))

import evaluation_script as es  # noqa: E402

KAIST = es.KAIST
KAISTPedEval = es.KAISTPedEval

TIME_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


def extract_hour(file_name: str):
    match = TIME_RE.search(file_name)
    if not match:
        return None
    return int(match.group(4))


def is_night(hour: int, night_start: int, night_end: int) -> bool:
    if night_start <= night_end:
        return night_start <= hour < night_end
    return hour >= night_start or hour < night_end


def add_cvc_fields(kaist_gt: KAIST) -> None:
    for ann in kaist_gt.anns.values():
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        if "height" not in ann:
            ann["height"] = float(bbox[3])
        if "occlusion" not in ann:
            ann["occlusion"] = 0
        if "ignore" not in ann:
            ann["ignore"] = 0


def ignore_border_gts(kaist_gt: KAIST) -> None:
    img_sizes = {
        img["id"]: (img.get("width"), img.get("height"))
        for img in kaist_gt.dataset.get("images", [])
    }
    ignored = 0
    missing = 0
    for ann in kaist_gt.anns.values():
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        img_size = img_sizes.get(ann.get("image_id"))
        if not img_size or img_size[0] is None or img_size[1] is None:
            missing += 1
            continue
        img_w, img_h = img_size
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            ann["ignore"] = 1
            ignored += 1
    if missing:
        print(f"[warn] {missing} annotations missing image size; border ignore skipped")
    if ignored:
        print(f"[info] marked {ignored} GT boxes as ignore for touching image border")


def get_gt_size(kaist_gt: KAIST):
    widths = {img["width"] for img in kaist_gt.dataset.get("images", [])}
    heights = {img["height"] for img in kaist_gt.dataset.get("images", [])}
    width = max(widths) if widths else None
    height = max(heights) if heights else None
    if len(widths) > 1 or len(heights) > 1:
        print(f"[warn] multiple image sizes detected: widths={sorted(widths)}, heights={sorted(heights)}")
    return width, height


def load_name_to_id(gt_path: str) -> dict:
    with open(gt_path, "r") as f:
        gt = json.load(f)
    return {img["file_name"]: img["id"] for img in gt["images"]}


def load_ids(list_path: str, name_to_id: dict, strict: bool) -> list:
    ids = []
    missing = []
    with open(list_path, "r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            name = os.path.basename(name)
            if name in name_to_id:
                ids.append(name_to_id[name])
            else:
                missing.append(name)
    if missing:
        msg = f"[warn] {list_path}: {len(missing)} names not found in GT"
        if strict:
            raise ValueError(msg)
        print(msg)
    return sorted(set(ids))


def split_day_night_ids(kaist_gt: KAIST, night_start: int, night_end: int, strict: bool):
    day_ids = []
    night_ids = []
    unknown = []
    for img in kaist_gt.dataset.get("images", []):
        hour = extract_hour(img.get("file_name", ""))
        if hour is None:
            unknown.append(img["id"])
            continue
        if is_night(hour, night_start, night_end):
            night_ids.append(img["id"])
        else:
            day_ids.append(img["id"])
    if unknown:
        msg = f"[warn] {len(unknown)} images missing time info; excluded from day/night split"
        if strict:
            raise ValueError(msg)
        print(msg)
    return sorted(set(day_ids)), sorted(set(night_ids))


def scale_txt_results(src: Path, dst: Path, x_scale: float, y_scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as f, dst.open("w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                print(f"[warn] skip bad line: {line}")
                continue
            image_id, x, y, w, h, score = parts
            x = float(x) * x_scale
            y = float(y) * y_scale
            w = float(w) * x_scale
            h = float(h) * y_scale
            g.write(f"{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score}\n")


def scale_json_results(src: Path, dst: Path, x_scale: float, y_scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        data = data["annotations"]
    if not isinstance(data, list):
        raise ValueError("result json must be a list or contain an 'annotations' list")
    scaled = []
    bad = 0
    for det in data:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            bad += 1
            continue
        det = det.copy()
        det["bbox"] = [
            float(bbox[0]) * x_scale,
            float(bbox[1]) * y_scale,
            float(bbox[2]) * x_scale,
            float(bbox[3]) * y_scale,
        ]
        scaled.append(det)
    if bad:
        print(f"[warn] skipped {bad} detections without valid bbox")
    with dst.open("w") as f:
        json.dump(scaled, f)


def eval_ids(kaist_gt: KAIST,
             kaist_dt: KAIST,
             img_ids: list,
             eval_width: int,
             eval_height: int,
             min_height: int,
             method: str) -> float:
    if not img_ids:
        print("[warn] empty img_ids, skip evaluation")
        return None

    evaluator = KAISTPedEval(kaist_gt, kaist_dt, "bbox", method)
    evaluator.params.catIds = [1]
    evaluator.params.imgIds = img_ids
    evaluator.params.HtRng = [[min_height, 1e5 ** 2]]
    evaluator.params.OccRng = [[0, 1, 2]]
    evaluator.params.SetupLbl = ["Reasonable"]
    # CVC-14 reasonable: ignore GTs truncated by image boundaries (5px margin).
    # Use the official CVC-14 boundary range: [5, 5, 635, 466].
    evaluator.params.bndRng = [5, 5, 635, 466]

    evaluator.evaluate(0)
    evaluator.accumulate()
    return evaluator.summarize(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CVC-14 with KAIST-style reasonable protocol.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--day-list", default=None)
    ap.add_argument("--night-list", default=None)
    ap.add_argument("--strict-lists", action="store_true", help="error on list names missing from GT")
    ap.add_argument("--night-start", type=int, default=20, help="hour to start night split (default: 20)")
    ap.add_argument("--night-end", type=int, default=6, help="hour to end night split (default: 6)")
    ap.add_argument("--input-width", type=int, default=None)
    ap.add_argument("--input-height", type=int, default=None)
    ap.add_argument("--eval-width", type=int, default=None)
    ap.add_argument("--eval-height", type=int, default=None)
    ap.add_argument("--no-scale", action="store_true", help="do not scale result file")
    ap.add_argument("--scaled-result", default=None, help="output path for scaled results")
    ap.add_argument("--min-height", type=int, default=55)
    args = ap.parse_args()

    kaist_gt = KAIST(args.gt)
    add_cvc_fields(kaist_gt)
    ignore_border_gts(kaist_gt)

    gt_width, gt_height = get_gt_size(kaist_gt)
    eval_width = args.eval_width or gt_width
    eval_height = args.eval_height or gt_height
    if eval_width is None or eval_height is None:
        raise ValueError("unable to infer eval size; please set --eval-width/--eval-height")

    if args.day_list or args.night_list:
        if not (args.day_list and args.night_list):
            raise ValueError("please provide both --day-list and --night-list")
        name_to_id = load_name_to_id(args.gt)
        day_ids = load_ids(args.day_list, name_to_id, args.strict_lists)
        night_ids = load_ids(args.night_list, name_to_id, args.strict_lists)
    else:
        day_ids, night_ids = split_day_night_ids(kaist_gt, args.night_start, args.night_end, args.strict_lists)

    all_ids = sorted(kaist_gt.getImgIds())

    result_path = Path(args.result)
    input_width = args.input_width or eval_width
    input_height = args.input_height or eval_height
    use_scaled = not args.no_scale and (
        input_width != eval_width or input_height != eval_height
    )
    if use_scaled:
        x_scale = eval_width / float(input_width)
        y_scale = eval_height / float(input_height)
        if args.scaled_result:
            scaled_path = Path(args.scaled_result)
        else:
            suffix = f"_scaled_{eval_width}x{eval_height}"
            scaled_path = result_path.with_name(result_path.stem + suffix + result_path.suffix)
        if result_path.suffix.lower() == ".txt":
            scale_txt_results(result_path, scaled_path, x_scale, y_scale)
        else:
            scale_json_results(result_path, scaled_path, x_scale, y_scale)
        result_path = scaled_path
        print(f"[info] scaled results -> {result_path}")

    kaist_dt = kaist_gt.loadRes(str(result_path))
    method = result_path.stem.split("_")[0]

    mr_all = eval_ids(kaist_gt, kaist_dt, all_ids, eval_width, eval_height, args.min_height, method)
    mr_day = eval_ids(kaist_gt, kaist_dt, day_ids, eval_width, eval_height, args.min_height, method)
    mr_night = eval_ids(kaist_gt, kaist_dt, night_ids, eval_width, eval_height, args.min_height, method)

    if mr_all is not None:
        print(f"MR_all:  {mr_all * 100:.2f}")
    else:
        print("MR_all:  N/A")

    if mr_day is not None:
        print(f"MR_day:  {mr_day * 100:.2f}")
    else:
        print("MR_day:  N/A")

    if mr_night is not None:
        print(f"MR_night:{mr_night * 100:.2f}")
    else:
        print("MR_night:N/A")


if __name__ == "__main__":
    main()
