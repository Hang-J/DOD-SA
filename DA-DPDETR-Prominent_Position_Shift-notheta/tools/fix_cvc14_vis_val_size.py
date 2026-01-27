#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def _clip_bbox_xywh(bbox, width, height):
    x, y, bw, bh = [float(v) for v in bbox]
    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + bw, float(width)))
    y2 = max(0.0, min(y + bh, float(height)))
    new_w = max(0.0, x2 - x1)
    new_h = max(0.0, y2 - y1)
    return [x1, y1, new_w, new_h], new_w * new_h


def _safe_symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.symlink_to(src)


def _safe_hardlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.link(src, dst)


def _safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def _crop_top_left_to(im, target_w, target_h):
    w, h = im.size
    if w < target_w or h < target_h:
        raise ValueError(
            f'Image too small to crop: got {w}x{h}, need {target_w}x{target_h}.'
        )
    if w == target_w and h == target_h:
        return im
    return im.crop((0, 0, target_w, target_h))


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Fix CVC-14 val VIS images with inconsistent sizes by cropping (top-left) '
            'to a target size and updating the corresponding COCO json (width/height + bbox clip).'
        ))
    parser.add_argument(
        '--dataset-dir',
        default='/data1/jinhang/Datasets/CVC-14',
        help='CVC-14 dataset root directory.')
    parser.add_argument(
        '--vis-image-dir',
        default='val_alllabeled/vis/images',
        help='VIS image directory relative to dataset-dir.')
    parser.add_argument(
        '--anno',
        default='val_vis_coco_paired_mbnet.json',
        help='COCO json (VIS) relative to dataset-dir.')
    parser.add_argument(
        '--target-width',
        type=int,
        default=640,
        help='Target width after cropping.')
    parser.add_argument(
        '--target-height',
        type=int,
        default=471,
        help='Target height after cropping.')
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Modify images/json in place (overwrite).')
    parser.add_argument(
        '--backup-json',
        action='store_true',
        help='When using --inplace, save a .bak copy of the original json.')
    parser.add_argument(
        '--output-vis-image-dir',
        default=None,
        help='Output VIS image dir (relative to dataset-dir) when not using --inplace.')
    parser.add_argument(
        '--output-anno',
        default=None,
        help='Output json path (relative to dataset-dir) when not using --inplace.')
    parser.add_argument(
        '--link-mode',
        choices=['symlink', 'hardlink', 'copy'],
        default='symlink',
        help=(
            'When not using --inplace, how to populate unchanged images in output dir. '
            'Changed images are always written as new cropped files.'
        ))
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report how many images/annotations would be changed.')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    vis_dir = dataset_dir / args.vis_image_dir
    anno_path = dataset_dir / args.anno
    target_w = int(args.target_width)
    target_h = int(args.target_height)

    if not vis_dir.is_dir():
        raise FileNotFoundError(f'VIS image dir not found: {vis_dir}')
    if not anno_path.is_file():
        raise FileNotFoundError(f'Annotation json not found: {anno_path}')

    if args.inplace:
        out_vis_dir = vis_dir
        out_anno_path = anno_path
    else:
        if args.output_vis_image_dir is None:
            out_vis_dir = vis_dir.parent / f'{vis_dir.name}_{target_w}x{target_h}'
        else:
            out_vis_dir = dataset_dir / args.output_vis_image_dir

        if args.output_anno is None:
            out_anno_path = anno_path.with_name(
                f'{anno_path.stem}_{target_w}x{target_h}{anno_path.suffix}')
        else:
            out_anno_path = dataset_dir / args.output_anno

    with anno_path.open('r', encoding='utf-8') as f:
        coco = json.load(f)

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    image_id_to_image = {img['id']: img for img in images if 'id' in img}
    image_id_to_filename = {
        img_id: image.get('file_name')
        for img_id, image in image_id_to_image.items()
    }

    changed_image_ids = set()
    changed_files = []
    for img_id, file_name in image_id_to_filename.items():
        if not file_name:
            continue
        src_path = vis_dir / file_name
        if not src_path.is_file():
            raise FileNotFoundError(f'Missing image referenced by json: {src_path}')
        with Image.open(src_path) as im:
            w, h = im.size
        if (w, h) != (target_w, target_h):
            changed_image_ids.add(img_id)
            changed_files.append((file_name, (w, h)))

    to_clip_ann_ids = set()
    for ann in annotations:
        if ann.get('image_id') in changed_image_ids:
            to_clip_ann_ids.add(ann.get('id'))

    print(f'[summary] images_total={len(images)} anns_total={len(annotations)}')
    print(f'[summary] images_need_crop={len(changed_image_ids)} anns_need_clip={len(to_clip_ann_ids)}')
    if changed_files:
        print('[summary] size_breakdown (first 5):')
        for name, (w, h) in changed_files[:5]:
            print(f'  - {name}: {w}x{h} -> {target_w}x{target_h}')

    if args.dry_run:
        return

    if args.inplace and args.backup_json:
        backup_path = anno_path.with_suffix(anno_path.suffix + '.bak')
        if not backup_path.exists():
            shutil.copy2(anno_path, backup_path)
            print(f'[backup] json saved to {backup_path}')

    if not args.inplace:
        out_vis_dir.mkdir(parents=True, exist_ok=True)
        print(f'[output] vis_dir={out_vis_dir}')
        print(f'[output] anno={out_anno_path}')

    # 1) Fix images (crop changed ones; replicate unchanged ones for non-inplace mode).
    if not args.inplace:
        if args.link_mode == 'symlink':
            link_fn = _safe_symlink
        elif args.link_mode == 'hardlink':
            link_fn = _safe_hardlink
        else:
            link_fn = _safe_copy

        for img_id, file_name in image_id_to_filename.items():
            if not file_name:
                continue
            src_path = vis_dir / file_name
            dst_path = out_vis_dir / file_name
            if img_id not in changed_image_ids:
                link_fn(src_path, dst_path)
                continue
            with Image.open(src_path) as im:
                cropped = _crop_top_left_to(im, target_w, target_h)
                # Keep a known image extension for PIL format inference.
                tmp_path = dst_path.with_suffix('.tmp' + dst_path.suffix)
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                if tmp_path.exists():
                    tmp_path.unlink()
                cropped.save(tmp_path)
                tmp_path.replace(dst_path)
    else:
        for img_id, file_name in image_id_to_filename.items():
            if img_id not in changed_image_ids or not file_name:
                continue
            src_path = vis_dir / file_name
            with Image.open(src_path) as im:
                cropped = _crop_top_left_to(im, target_w, target_h)
                # Keep a known image extension for PIL format inference.
                tmp_path = src_path.with_suffix('.tmp' + src_path.suffix)
                if tmp_path.exists():
                    tmp_path.unlink()
                cropped.save(tmp_path)
            tmp_path.replace(src_path)

    # 2) Fix json: image sizes + bbox clip for changed images.
    for img_id in changed_image_ids:
        image = image_id_to_image.get(img_id)
        if not image:
            continue
        image['width'] = int(target_w)
        image['height'] = int(target_h)

    new_annotations = []
    dropped = 0
    for ann in annotations:
        if ann.get('image_id') not in changed_image_ids:
            new_annotations.append(ann)
            continue
        bbox = ann.get('bbox')
        if not bbox or len(bbox) != 4:
            new_annotations.append(ann)
            continue
        clipped_bbox, area = _clip_bbox_xywh(bbox, target_w, target_h)
        if area <= 0:
            dropped += 1
            continue
        ann['bbox'] = clipped_bbox
        ann['area'] = float(area)
        new_annotations.append(ann)

    coco['annotations'] = new_annotations
    if dropped:
        print(f'[summary] dropped_invalid_anns={dropped}')

    out_anno_path.parent.mkdir(parents=True, exist_ok=True)
    with out_anno_path.open('w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f'[done] wrote {out_anno_path}')


if __name__ == '__main__':
    main()
