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
    print('no npu!')

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
        coord_pattern = r'\[(\d+),(\d+)(?:,(\d+),(\d+))?\]'
        matches = re.findall(coord_pattern, predict_text)
        
        if not matches:
            return None, None, False
        
        bbox_list = []
        points = []
        
        for match in matches:
            if match[2] and match[3]:
                bbox_list.append([int(match[0]), int(match[1]), int(match[2]), int(match[3])])
            else:
                points.append([int(match[0]), int(match[1])])
        bbox = bbox_list[0] if bbox_list else None
        
        return bbox, points, True
        
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None, None, False


def validate_coordinates(bbox, points, img_size=224):
    if bbox:
        if (bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= img_size or bbox[3] >= img_size or
            bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            return False
    if not points or len(points) < 1:
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
        print(f"Error loading image {image_path}: {e}")
        return None


def load_mask(mask_path, target_size=224):
    try:
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        return mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None


def create_comparison_image(pred_mask, gt_mask, original_image_path, bbox, points, target_size=224):
    try:
        original_img = load_and_resize_image(original_image_path, target_size)
        if original_img is None:
            original_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        pred_mask_vis = np.stack([pred_mask * 255] * 3, axis=-1).astype(np.uint8)
        gt_mask_vis = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)

        overlay = original_img.copy()
        overlay[pred_mask == 1] = (overlay[pred_mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        overlay[gt_mask == 1] = (overlay[gt_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        comparison = np.hstack([
            original_img,
            pred_mask_vis,
            gt_mask_vis,
            overlay
        ])
        
        comparison_img = Image.fromarray(comparison)
        draw = ImageDraw.Draw(comparison_img)

        for i, panel_offset_x in enumerate([0, 3 * target_size]):
            if bbox:
                box_coords = [
                    bbox[0] + panel_offset_x, bbox[1],
                    bbox[2] + panel_offset_x, bbox[3]
                ]
                draw.rectangle(box_coords, outline="blue", width=2)
            if points:
                colors = ["yellow", "red", "green"]
                for idx, point in enumerate(points):
                    x, y = point
                    color = colors[idx % len(colors)]
                    point_coords = [
                        x - 3 + panel_offset_x, y - 3,
                        x + 3 + panel_offset_x, y + 3
                    ]
                    draw.ellipse(point_coords, fill=color, outline="black")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        labels = ["Original + Prompts", "Predicted", "Ground Truth", "Overlay"]
        for i, label in enumerate(labels):
            x_pos = i * target_size + 5
            draw.text((x_pos + 1, 5 + 1), label, fill="black", font=font)
            draw.text((x_pos, 5), label, fill="white", font=font)
            
        return comparison_img

    except Exception as e:
        print(f"Error creating comparison image: {e}")
        pred_mask_vis = np.stack([pred_mask * 255] * 3, axis=-1).astype(np.uint8)
        gt_mask_vis = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)
        simple_comparison = np.hstack([pred_mask_vis, gt_mask_vis])
        return Image.fromarray(simple_comparison)


def create_pred_image(pred_mask, original_image_path, bbox, points, target_size=224):
    try:
        original_img = load_and_resize_image(original_image_path, target_size)
        if original_img is None:
            original_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        overlay = original_img.copy()
        overlay[pred_mask == 1] = (overlay[pred_mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        overlay_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_img)

        if bbox:
            box_coords = [bbox[0], bbox[1], bbox[2], bbox[3]]
            draw.rectangle(box_coords, outline="blue", width=2)
        if points:
            colors = ["yellow", "red", "green"]
            for idx, point in enumerate(points):
                x, y = point
                color = colors[idx % len(colors)]
                point_coords = [x - 3, y - 3, x + 3, y + 3]
                draw.ellipse(point_coords, fill=color, outline="black")

        # No text labels

        return overlay_img

    except Exception as e:
        print(f"Error creating pred image: {e}")
        pred_mask_vis = np.stack([pred_mask * 255] * 3, axis=-1).astype(np.uint8)
        return Image.fromarray(pred_mask_vis)


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


def main():
    parser = argparse.ArgumentParser(description='SAM2 Multi-GPU Inference and Evaluation')
    parser.add_argument('--jsonl_path', type=str, required=True,
                        help='Path to JSONL file')
    parser.add_argument('--checkpoint', type=str, 
                        default="/dockerdata/sam2.1_hiera_large.pt",
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--model_cfg', type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml", 
                        help='Path to model config')
    parser.add_argument('--output_dir', type=str, default="./sam2_results",
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='hccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger = setup_logging(rank)
    logger.info(f"Rank {rank}/{world_size}, Local rank {local_rank}, Device: {device}")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(exist_ok=True)
        (output_dir / "predicted_masks").mkdir(exist_ok=True)

    if world_size > 1:
        dist.barrier()

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=device))

    with open(args.jsonl_path, 'r') as f:
        all_data = [json.loads(line) for line in f]

    samples_per_process = len(all_data) // world_size
    start_idx = rank * samples_per_process
    if rank == world_size - 1:
        end_idx = len(all_data)
    else:
        end_idx = start_idx + samples_per_process
    
    local_data = all_data[start_idx:end_idx]

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
                
                if not success:
                    invalid_samples.append({
                        'line': global_idx,
                        'image_path': item['image_path'],
                        'reason': 'coordinate_extraction_failed'
                    })
                    continue

                use_bbox = bbox
                use_points = None

                if bbox and points:
                    if len(points) >= 3:
                        use_points = points[-3:]
                    elif len(points) >= 2:
                        use_points = points[-2:]
                    elif len(points) >= 1:
                        use_points = points[-1:]
                elif points and len(points) >= 3:
                    use_bbox = None
                    use_points = points[-3:]

                if use_points is None:
                    invalid_samples.append({
                        'line': global_idx,
                        'image_path': item['image_path'],
                        'reason': 'insufficient_coordinates'
                    })
                    continue

                image_path = item['image_path']
                image = load_and_resize_image(image_path)
                
                if image is None:
                    invalid_samples.append({
                        'line': global_idx,
                        'image_path': image_path,
                        'reason': 'image_load_failed'
                    })
                    continue

                if not validate_coordinates(use_bbox, use_points):
                    invalid_samples.append({
                        'line': global_idx,
                        'image_path': image_path,
                        'reason': 'invalid_coordinates'
                    })
                    continue

                mask_path = image_path.replace('/frames/', '/labels_rgb/').replace('.jpg', '.png')

                gt_mask = load_mask(mask_path)
                if gt_mask is None:
                    invalid_samples.append({
                        'line': global_idx,
                        'image_path': image_path,
                        'reason': 'mask_load_failed'
                    })
                    continue

                batch_images.append(image)
                batch_boxes.append(np.array(use_bbox) if use_bbox else None)
                batch_points.append(np.array(use_points))
                batch_labels.append(np.array([1] * len(use_points)))
                batch_info.append({
                    'global_idx': global_idx,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'gt_mask': gt_mask,
                    'bbox': use_bbox,
                    'points': use_points,
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
                        point_coords_batch=batch_points,
                        point_labels_batch=batch_labels,
                        box_batch=batch_boxes,
                        multimask_output=False,
                    )
                
                logger.info(f"Inference completed, got {len(masks_batch)} results")
                for j, (masks, scores, info) in enumerate(zip(masks_batch, scores_batch, batch_info)):
                    try:
                        if isinstance(masks[0], torch.Tensor):
                            pred_mask = masks[0].cpu().numpy().astype(np.uint8)
                        else:
                            pred_mask = masks[0].astype(np.uint8)
                        
                        gt_mask = info['gt_mask']
                        if isinstance(scores[0], torch.Tensor):
                            score_value = float(scores[0].cpu().numpy())
                        else:
                            score_value = float(scores[0])

                        jaccard, f_score = calculate_metrics(pred_mask, gt_mask)
                        
                        metric = {
                            'global_idx': info['global_idx'],
                            'jaccard': jaccard,
                            'f_score': f_score,
                            'score': score_value
                        }

                        if jaccard < 0.5:
                            comparison_img = create_comparison_image(
                                pred_mask,
                                gt_mask,
                                info['image_path'],
                                info['bbox'],
                                info['points']
                            )
                            comparison_filename = f"comparison_{info['global_idx']}.png"
                            comparison_path = output_dir / "predicted_masks" / comparison_filename
                            comparison_img.save(comparison_path)
                            metric['comparison_path'] = str(comparison_path)

                            pred_img = create_pred_image(
                                pred_mask,
                                info['image_path'],
                                info['bbox'],
                                info['points']
                            )
                            pred_filename = f"pred_{info['global_idx']}.png"
                            pred_path = output_dir / "predicted_masks" / pred_filename
                            pred_img.save(pred_path)

                        all_metrics.append(metric)
                        
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
        'invalid_samples': invalid_samples,
        'metrics': all_metrics,
        'processed_count': processed_count
    }
    
    if world_size > 1:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

    if rank == 0:
        logger.info("Aggregating results from all ranks...")
        
        if world_size > 1:
            aggregate_list = all_results
        else:
            aggregate_list = [results]

        all_invalid_samples = []
        all_metrics = []
        total_processed = 0

        for res in aggregate_list:
            all_invalid_samples.extend(res['invalid_samples'])
            all_metrics.extend(res['metrics'])
            total_processed += res['processed_count']

        if all_invalid_samples:
            with open(output_dir / "invalid_samples.txt", 'w') as f:
                for sample in all_invalid_samples:
                    f.write(f"Line {sample['line']}: {sample['image_path']} - {sample['reason']}\n")

        if all_metrics:
            jaccard_scores = [m['jaccard'] for m in all_metrics]
            f_scores = [m['f_score'] for m in all_metrics]
            scores = [m['score'] for m in all_metrics]
            
            mean_jaccard = np.mean(jaccard_scores)
            mean_f_score = np.mean(f_scores)
            mean_score = np.mean(scores)
            
            logger.info(f"\n=== Evaluation Results ===")
            logger.info(f"Total processed samples: {total_processed}")
            logger.info(f"Invalid samples: {len(all_invalid_samples)}")
            logger.info(f"Average Jaccard Index (IoU): {mean_jaccard:.4f}")
            logger.info(f"Average F-Score: {mean_f_score:.4f}")
            logger.info(f"Average SAM Score: {mean_score:.4f}")

            final_results = {
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

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()