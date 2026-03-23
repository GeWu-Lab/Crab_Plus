import torch
import torch.distributed as dist
import os
import json
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from tqdm import tqdm
from pathlib import Path
import logging

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'sam2_inference_rank_{rank}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_coordinates(predict_text):
    try:
        if '</s>' in predict_text:
            predict_text = predict_text.split('</s>', 1)[0]
        if '<answer>false</answer>' in predict_text:
            return None, None, False

        if '<answer>true</answer>' not in predict_text:
            return None, None, False

        coord_pattern = r'\[(\d+),(\d+)(?:,(\d+),(\d+))?\]'
        matches = re.findall(coord_pattern, predict_text)

        if not matches:
            return None, None, False

        bbox = None
        points = []

        for match in matches:
            if match[2] and match[3]:
                if bbox is None:
                    bbox = [int(match[0]), int(match[1]), int(match[2]), int(match[3])]
            else:
                points.append([int(match[0]), int(match[1])])

        if len(points) > 3:
            points = points[-3:]

        if bbox is None and len(points) == 0:
            return None, None, False

        return bbox, points, True

    except Exception as e:
        return None, None, False


def validate_coordinates(bbox, points, img_size=224):
    if bbox is None and len(points) == 0:
        return False

    if bbox is not None:
        if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= img_size or bbox[3] >= img_size or
            bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            return False

    for point in points:
        if point[0] < 0 or point[1] < 0 or point[0] >= img_size or point[1] >= img_size:
            return False

    return True


def load_and_resize_image(image_path, target_size=224):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return np.array(image)
    except Exception as e:
        return None


def load_mask(mask_path, target_size=224):
    try:
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        return mask
    except Exception as e:
        return None


def calculate_metrics(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        jaccard = 1.0 if intersection == 0 else 0.0
    else:
        jaccard = intersection / union

    beta_squared = 0.3

    true_positive = intersection
    predicted_positive = pred_mask.sum()
    actual_positive = gt_mask.sum()

    if predicted_positive == 0:
        precision = 1.0 if true_positive == 0 else 0.0
    else:
        precision = true_positive / predicted_positive

    if actual_positive == 0:
        recall = 1.0 if true_positive == 0 else 0.0
    else:
        recall = true_positive / actual_positive

    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)

    return jaccard, f_score


def calculate_s_metric(pred_mask):
    pred_mask = pred_mask.astype(bool)

    mask_area = np.sum(pred_mask)

    background_area = np.sum(~pred_mask)

    if background_area == 0:
        s_metric = np.inf
    else:
        ratio = mask_area / background_area
        s_metric = np.sqrt(ratio)

    return s_metric


def main():
    parser = argparse.ArgumentParser(description='SAM2 Multi-GPU Inference and Evaluation')
    parser.add_argument('--jsonl_path', type=str, required=True,
                        help='Path to JSONL file')
    parser.add_argument('--checkpoint', type=str,
                        default="./sam2/checkpoints/sam2.1_hiera_large.pt",
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--model_cfg', type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help='Path to model config')
    parser.add_argument('--output_dir', type=str, default="./sam2_results",
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--split',type=str,default="test_s",help="ref-avs type")

    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        backend = 'hccl' if hasattr(torch, 'npu') and torch.npu.is_available() else 'nccl'
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(rank)
    logger.info(f"Rank {rank}/{world_size}, Local rank {local_rank}, Device: {device}")

    use_test_n_eval = (args.split == "test_n")
    if use_test_n_eval and rank == 0:
        logger.info("Using test_n evaluation mode, calculating S metric")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(exist_ok=True)
    if world_size > 1:
        dist.barrier()

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=device))

    with open(args.jsonl_path, 'r') as f:
        all_data = [
            json.loads(line)
            for line in f
            if json.loads(line).get('split') == args.split
        ]

    samples_per_process = len(all_data) // world_size
    start_idx = rank * samples_per_process
    if rank == world_size - 1:
        end_idx = len(all_data)
    else:
        end_idx = start_idx + samples_per_process

    local_data = all_data[start_idx:end_idx]
    valid_samples = []
    invalid_samples = []
    all_metrics = []
    processed_count = 0

    for i in tqdm(range(0, len(local_data), args.batch_size),
                  desc=f"Rank {rank}", disable=(rank != 0)):
        batch_data = local_data[i:i+args.batch_size]
        batch_images = []
        batch_boxes = []
        batch_points = []
        batch_labels = []
        batch_info = []

        for idx, item in enumerate(batch_data):
            global_idx = start_idx + i + idx

            try:
                bbox, points, success = extract_coordinates(item['predict'])

                image_path = item['image_path']
                image = load_and_resize_image(image_path)
                if image is None:
                    invalid_samples.append({
                        'line': global_idx,
                        'image_path': image_path,
                        'reason': 'image_load_failed'
                    })
                    continue

                gt_mask = None
                mask_path = None
                if not use_test_n_eval:
                    fid = item['fid']
                    mask_path = image_path.replace('/media/', f"/gt_mask/").replace('.jpg', '.png').replace('/frames/', f'/fid_{fid}/0000')
                    gt_mask = load_mask(mask_path)
                    if gt_mask is None:
                        invalid_samples.append({
                            'line': global_idx,
                            'image_path': image_path,
                            'reason': 'mask_load_failed'
                        })
                        continue

                if success:
                    if not validate_coordinates(bbox, points):
                        invalid_samples.append({
                            'line': global_idx,
                            'image_path': image_path,
                            'reason': 'invalid_coordinates'
                        })
                        continue

                    batch_images.append(image)
                    batch_boxes.append(np.array(bbox) if bbox is not None else None)
                    batch_points.append(np.array(points) if points else None)
                    batch_labels.append(np.array([1] * len(points)) if points else None)
                    batch_info.append({
                        'global_idx': global_idx,
                        'image_path': image_path,
                        'mask_path': mask_path if gt_mask is not None else None,
                        'gt_mask': gt_mask,
                        'bbox': bbox,
                        'points': points,
                    })
                else:
                    if '<answer>false</answer>' in item['predict']:
                        pred_mask = np.zeros((224, 224), dtype=np.uint8)
                        score_value = 0.0

                        if use_test_n_eval:
                            s_metric = calculate_s_metric(pred_mask)
                            all_metrics.append({
                                'global_idx': global_idx,
                                's_metric': s_metric,
                                'score': score_value
                            })

                            sample_info = {
                                'global_idx': global_idx,
                                'image_path': image_path,
                                's_metric': s_metric,
                                'score': score_value
                            }
                        else:
                            jaccard, f_score = calculate_metrics(pred_mask, gt_mask)

                            all_metrics.append({
                                'global_idx': global_idx,
                                'jaccard': jaccard,
                                'f_score': f_score,
                                'score': score_value
                            })

                            sample_info = {
                                'global_idx': global_idx,
                                'image_path': image_path,
                                'jaccard': jaccard,
                                'f_score': f_score,
                                'score': score_value
                            }

                        valid_samples.append(sample_info)
                        processed_count += 1
                    else:
                        invalid_samples.append({
                            'line': global_idx,
                            'image_path': image_path,
                            'reason': 'coordinate_extraction_failed'
                        })

            except Exception as e:
                logger.error(f"Error processing sample {global_idx}: {e}")
                invalid_samples.append({
                    'line': global_idx,
                    'image_path': item.get('image_path', 'unknown'),
                    'reason': f'processing_error: {str(e)}'
                })

        if batch_images:
            try:
                logger.info(f"Running inference on batch of {len(batch_images)} images")

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor.set_image_batch(batch_images)
                    masks_batch, scores_batch, logits_batch = predictor.predict_batch(
                        point_coords_batch=batch_points if any(p is not None for p in batch_points) else None,
                        point_labels_batch=batch_labels if any(l is not None for l in batch_labels) else None,
                        box_batch=batch_boxes if any(b is not None for b in batch_boxes) else None,
                        multimask_output=False,
                    )

                logger.info(f"Inference complete, got {len(masks_batch)} results")
                for j, (masks, scores, info) in enumerate(zip(masks_batch, scores_batch, batch_info)):
                    try:
                        if isinstance(masks[0], torch.Tensor):
                            pred_mask = masks[0].cpu().numpy().astype(np.uint8)
                        else:
                            pred_mask = masks[0].astype(np.uint8)

                        if isinstance(scores[0], torch.Tensor):
                            score_value = float(scores[0].cpu().numpy())
                        else:
                            score_value = float(scores[0])

                        if use_test_n_eval:
                            s_metric = calculate_s_metric(pred_mask)
                            all_metrics.append({
                                'global_idx': info['global_idx'],
                                's_metric': s_metric,
                                'score': score_value
                            })

                            sample_info = {
                                'global_idx': info['global_idx'],
                                'image_path': info['image_path'],
                                's_metric': s_metric,
                                'score': score_value
                            }
                        else:
                            gt_mask = info['gt_mask']
                            jaccard, f_score = calculate_metrics(pred_mask, gt_mask)

                            all_metrics.append({
                                'global_idx': info['global_idx'],
                                'jaccard': jaccard,
                                'f_score': f_score,
                                'score': score_value
                            })

                            sample_info = {
                                'global_idx': info['global_idx'],
                                'image_path': info['image_path'],
                                'jaccard': jaccard,
                                'f_score': f_score,
                                'score': score_value
                            }

                        valid_samples.append(sample_info)

                        processed_count += 1

                    except Exception as e:
                        logger.error(f"Error processing result {j}: {e}")
                        invalid_samples.append({
                            'line': info['global_idx'],
                            'image_path': info['image_path'],
                            'reason': f'result_processing_error: {str(e)}'
                        })

            except Exception as e:
                logger.error(f"Error during batch inference: {e}")
                for info in batch_info:
                    invalid_samples.append({
                        'line': info['global_idx'],
                        'image_path': info['image_path'],
                        'reason': f'inference_failed: {str(e)}'
                    })

    logger.info(f"Rank {rank} processed {processed_count} valid samples, {len(invalid_samples)} invalid samples")
    results = {
        'rank': rank,
        'split': args.split,
        'use_test_n_eval': use_test_n_eval,
        'valid_samples': valid_samples,
        'invalid_samples': invalid_samples,
        'metrics': all_metrics,
        'processed_count': processed_count
    }

    results_path = output_dir / f"results_rank_{rank}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info("Aggregating results from all ranks...")

        all_valid_samples = []
        all_invalid_samples = []
        all_metrics = []
        total_processed = 0

        for r in range(world_size):
            results_path = output_dir / f"results_rank_{r}.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    all_valid_samples.extend(results['valid_samples'])
                    all_invalid_samples.extend(results['invalid_samples'])
                    all_metrics.extend(results['metrics'])
                    total_processed += results['processed_count']

        if all_invalid_samples:
            with open(output_dir / "invalid_samples.txt", 'w') as f:
                for sample in all_invalid_samples:
                    f.write(f"Line {sample['line']}: {sample['image_path']} - {sample['reason']}\n")

        if all_metrics:
            logger.info(f"\n=== Evaluation Results ===")
            logger.info(f"Split: {args.split}")
            logger.info(f"Total processed samples: {total_processed}")
            logger.info(f"Invalid samples: {len(all_invalid_samples)}")

            if use_test_n_eval:
                s_metrics = [m['s_metric'] for m in all_metrics if not np.isinf(m['s_metric'])]
                scores = [m['score'] for m in all_metrics]

                mean_s_metric = np.mean(s_metrics) if s_metrics else 0.0
                mean_score = np.mean(scores)

                logger.info(f"Average S metric: {mean_s_metric:.4f}")
                logger.info(f"Average SAM Score: {mean_score:.4f}")
                logger.info(f"Number of samples with infinite S metric: {len([m for m in all_metrics if np.isinf(m['s_metric'])])}")

                final_results = {
                    'split': args.split,
                    'evaluation_type': 'null',
                    'mean_s_metric': mean_s_metric,
                    'mean_sam_score': mean_score,
                    'total_processed': total_processed,
                    'invalid_samples': len(all_invalid_samples),
                    'infinite_s_metric_count': len([m for m in all_metrics if np.isinf(m['s_metric'])]),
                    'detailed_metrics': all_metrics,
                    'invalid_samples_list': all_invalid_samples
                }
            else:
                evaluation_type = 'seen' if args.split == 'test_s' else 'unseen' if args.split == 'test_u' else 'traditional'
                jaccard_scores = [m['jaccard'] for m in all_metrics]
                f_scores = [m['f_score'] for m in all_metrics]
                scores = [m['score'] for m in all_metrics]

                mean_jaccard = np.mean(jaccard_scores)
                mean_f_score = np.mean(f_scores)
                mean_score = np.mean(scores)

                logger.info(f"Average Jaccard index (IoU): {mean_jaccard:.4f}")
                logger.info(f"Average F-Score: {mean_f_score:.4f}")
                logger.info(f"Average SAM Score: {mean_score:.4f}")

                final_results = {
                    'split': args.split,
                    'evaluation_type': evaluation_type,
                    'mean_jaccard': mean_jaccard,
                    'mean_f_score': mean_f_score,
                    'mean_sam_score': mean_score,
                    'total_processed': total_processed,
                    'invalid_samples': len(all_invalid_samples),
                    'detailed_metrics': all_metrics,
                    'invalid_samples_list': all_invalid_samples
                }

            with open(output_dir / "evaluation_results.json", 'w') as f:
                json.dump(final_results, f, indent=2)

        for r in range(world_size):
            results_path = output_dir / f"results_rank_{r}.json"
            if results_path.exists():
                results_path.unlink()

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()