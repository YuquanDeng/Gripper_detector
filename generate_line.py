#!/usr/bin/env python
import argparse
import logging
import os
import tarfile
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import cv2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from torchvision import transforms
import detectron2.data.transforms as T
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Additional imports for tar handling, folder handling, and JSON output
import glob
import re

class BatchPredictor(nn.Module):
    """
    The batch version of detectron2 DefaultPredictor.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()  # Clone config so the original is not modified.
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def forward(self, imgs):
        with torch.no_grad():
            inputs = []
            for original_image in imgs:
                if self.input_format == "RGB":
                    # Convert from RGB to BGR if needed.
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                entry = {"image": image, "height": height, "width": width}
                inputs.append(entry)
            predictions = self.model(inputs)
            return predictions

def process_image(image_path, predictor):
    """
    Process one image from disk and return a dict with the relative image path,
    the center point of the detection (if any), and the detection score.
    """
    image_cv2 = cv2.imread(image_path)
    if image_cv2 is None:
        logging.error(f"Unable to read image: {image_path}")
        return None

    # Run inference
    det_pred = predictor([image_cv2])
    instances = det_pred[0]['instances']

    # If there are no detections, record empty values.
    if len(instances) == 0:
        return {
            "image": image_path,
            "point": None,
            "score": None
        }

    # Get the predicted bounding boxes as a tensor (shape: [num_instances, 4])
    boxes = instances.pred_boxes.tensor
    # Compute the center points of each bounding box
    center_points = (boxes[:, :2] + boxes[:, 2:]) / 2

    # For simplicity, choose the detection with the highest score.
    scores = instances.scores
    best_idx = torch.argmax(scores).item()
    best_center = center_points[best_idx]
    best_score = scores[best_idx].item()

    # Convert center point to integers
    center_point = [int(best_center[0].item()), int(best_center[1].item())]

    return {
        "image": image_path,
        "point": center_point,
        "score": best_score
    }

def process_tar(tar_path, predictor, batch_size=2048):
    """
    Process all PNG images from the tar archive without extracting everything to disk.
    Returns a list of dictionaries with image path (relative within tar), center point, and detection score.
    """
    results = []
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        batch_imgs = []
        batch_paths = []
        for member in tqdm(members, desc="Processing images in tar"):
            if not member.isfile() or not member.name.lower().endswith('.png'):
                continue
            file_obj = tar.extractfile(member)
            if file_obj is None:
                logging.error(f"Unable to extract {member.name}")
                continue
            file_bytes = file_obj.read()
            arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                logging.error(f"Unable to decode image: {member.name}")
                continue
            batch_imgs.append(image)
            batch_paths.append(member.name)
            if len(batch_imgs) == batch_size:
                batch_predictions = predictor(batch_imgs)
                for prediction, rel_path in zip(batch_predictions, batch_paths):
                    instances = prediction['instances']
                    if len(instances) == 0:
                        results.append({
                            "image": rel_path,
                            "point": None,
                            "score": None
                        })
                    else:
                        boxes = instances.pred_boxes.tensor
                        center_points = (boxes[:, :2] + boxes[:, 2:]) / 2
                        scores = instances.scores
                        best_idx = torch.argmax(scores).item()
                        best_center = center_points[best_idx]
                        best_score = scores[best_idx].item()
                        center_point = [int(best_center[0].item()), int(best_center[1].item())]
                        results.append({
                            "image": rel_path,
                            "point": center_point,
                            "score": best_score
                        })
                batch_imgs = []
                batch_paths = []
        # Process any remaining images in the final batch
        if batch_imgs:
            batch_predictions = predictor(batch_imgs)
            for prediction, rel_path in zip(batch_predictions, batch_paths):
                instances = prediction['instances']
                if len(instances) == 0:
                    results.append({
                        "image": rel_path,
                        "point": None,
                        "score": None
                    })
                else:
                    boxes = instances.pred_boxes.tensor
                    center_points = (boxes[:, :2] + boxes[:, 2:]) / 2
                    scores = instances.scores
                    best_idx = torch.argmax(scores).item()
                    best_center = center_points[best_idx]
                    best_score = scores[best_idx].item()
                    center_point = [int(best_center[0].item()), int(best_center[1].item())]
                    results.append({
                        "image": rel_path,
                        "point": center_point,
                        "score": best_score
                    })
    return results

def process_folder(folder_path, predictor, batch_size=2048):
    """
    Process all PNG images in the given folder and its subfolders using batched inference.
    Returns a list of dictionaries with image path, center point, and detection score.
    """
    results = []
    # Recursively search for PNG images in the folder and subfolders.
    image_paths = glob.glob(os.path.join(folder_path, '**/*.png'), recursive=True)
    # Sort the image_paths based on the numerical value in the filename
    image_paths = sorted(
        image_paths,
        key=lambda x: int(re.search(r'(\d+)\.png$', x).group(1))
    )
    print(f"Found {len(image_paths)} images in folder: {folder_path}")

    if len(image_paths) == 0:
        logging.error(f"No PNG images found in folder: {folder_path}")
        return results

    batch_imgs = []
    batch_paths = []
    for image_path in tqdm(image_paths, desc="Processing images in folder"):
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Unable to read image: {image_path}")
            continue
        batch_imgs.append(image)
        batch_paths.append(image_path)
        if len(batch_imgs) == batch_size:
            batch_predictions = predictor(batch_imgs)
            for prediction, rel_path in zip(batch_predictions, batch_paths):
                instances = prediction['instances']
                if len(instances) == 0:
                    results.append({
                        "image": rel_path,
                        "point": None,
                        "score": None
                    })
                else:
                    boxes = instances.pred_boxes.tensor
                    center_points = (boxes[:, :2] + boxes[:, 2:]) / 2
                    scores = instances.scores
                    best_idx = torch.argmax(scores).item()
                    best_center = center_points[best_idx]
                    best_score = scores[best_idx].item()
                    center_point = [int(best_center[0].item()), int(best_center[1].item())]
                    results.append({
                        "image": rel_path,
                        "point": center_point,
                        "score": best_score
                    })
            batch_imgs = []
            batch_paths = []
    # Process any remaining images in the final batch
    if batch_imgs:
        batch_predictions = predictor(batch_imgs)
        for prediction, rel_path in zip(batch_predictions, batch_paths):
            instances = prediction['instances']
            if len(instances) == 0:
                results.append({
                    "image": rel_path,
                    "point": None,
                    "score": None
                })
            else:
                boxes = instances.pred_boxes.tensor
                center_points = (boxes[:, :2] + boxes[:, 2:]) / 2
                scores = instances.scores
                best_idx = torch.argmax(scores).item()
                best_center = center_points[best_idx]
                best_score = scores[best_idx].item()
                center_point = [int(best_center[0].item()), int(best_center[1].item())]
                results.append({
                    "image": rel_path,
                    "point": center_point,
                    "score": best_score
                })
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference for end-effector detector with support for tar archives, folders, or single images'
    )
    parser.add_argument('--config_file', type=str, default='config/det_config.yaml')
    parser.add_argument('--ckpt', type=str, default='ckpt/model_final.pth')
    parser.add_argument('--device', type=str, default='cuda')
    # Provide either a single image, a tar archive, or a folder.
    parser.add_argument('--input_img', type=str, default=None,
                        help="Path to a single image (e.g., /path/to/0000.png)")
    parser.add_argument('--input_tar', type=str, default=None,
                        help="Path to a tar archive containing images (e.g., bridge.tar)")
    parser.add_argument('--input_folder', type=str, default=None,
                        help="Path to a folder containing images (images can be in subfolders)")
    parser.add_argument('--save_path', type=str, default='/gscratch/krishna/jason328/ft-real/data/points',
                        help="Directory where the JSON file will be saved")
    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help="Confidence threshold for detections")
    # Batch size for inference on tar archives and folders
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for inference on tar archives and folders")
    args = parser.parse_args()

    # Set up the configuration.
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.ckpt  # set the model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thresh  # set the testing threshold
    cfg.MODEL.DEVICE = args.device  # set the device
    predictor = BatchPredictor(cfg).to(args.device)

    # Create output directory if it doesn't exist.
    # os.makedirs(args.save_path, exist_ok=True)
    
    # Determine output JSON filename based on provided input.
    if args.input_tar:
        raise NotImplementedError("Tar file processing is not implemented yet.")
        tar_basename = os.path.basename(args.input_tar)
        tar_basename = os.path.splitext(tar_basename)[0]
        output_json = os.path.join(args.save_path, f"{tar_basename}.json")
        all_results = process_tar(args.input_tar, predictor, batch_size=args.batch_size)
        
    elif args.input_folder:
        output_json = args.input_folder.replace("rgb", "line") + "_point.json"
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        # folder_basename = os.path.basename(os.path.normpath(args.input_folder))
        # output_json = os.path.join(args.save_path, f"{folder_basename}.json")
        all_results = process_folder(args.input_folder, predictor, batch_size=args.batch_size)
    elif args.input_img:
        raise NotImplementedError("Single image processing is not implemented yet.")
        img_basename = os.path.basename(args.input_img)
        img_basename = os.path.splitext(img_basename)[0]
        output_json = os.path.join(args.save_path, f"{img_basename}.json")
        result = process_image(args.input_img, predictor)
        all_results = []
        if result is not None:
            all_results.append(result)
    else:
        raise ValueError("You must provide either --input_img, --input_tar, or --input_folder.")

    # Write the results to a JSON file.
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {output_json}")
    
    ######################################################################################
    # Filter no-ops from the JSON file
    ######################################################################################
    
    print("Filtering no-ops from the JSON file...")
    # Load the JSON data (assuming it's a list of dictionaries)
    with open(output_json, 'r') as f:
        data = json.load(f)

    last_valid_point = None
    last_valid_score = None

    # Iterate over each entry in the JSON
    for entry in data:
        # Check if the "point" is null (None in Python)
        if entry["point"] is None:
            # Replace with the last valid point and score
            entry["point"] = last_valid_point
            entry["score"] = last_valid_score
        else:
            # Update the last valid point and score if this entry is valid
            last_valid_point = entry["point"]
            last_valid_score = entry["score"]

    # Save the updated JSON back to file
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
    print("Updated JSON file saved as:", output_json)
    
    ######################################################################################
    # Connect the points to form lines
    ######################################################################################
    print("Connecting points to form lines...")
    with open(output_json, 'r') as f:
        data = json.load(f)
    
    # Group samples by episode.
    # Assuming the image path format is like:
    #   /gscratch/krishna/jason328/openvla/collection/02-26-25/001/003.png
    # where the folder before the filename is the episode id.
    episodes = {}
    for sample in data:
        # Split the path and extract the episode folder (penultimate element)
        parts = sample['image'].split('/')
        # episode = parts[-1][:parts[-1].find("step_")-1]
        # Also extract the step number from the filename (e.g., "003" from "003.png")
        # step = int(parts[-1].split('.')[0]) # TODO
        # sample['episode'] = episode # TODO
        # sample['step'] = step
        step = int(sample['image'].split('/')[-1].split('.')[0])
        sample['episode'] = sample['image']
        sample['step'] = step
        episode = parts[-2]
        episodes.setdefault(episode, []).append(sample)

    # Sort each episode's samples by step number
    for episode in episodes:
        episodes[episode].sort(key=lambda x: x['step'])

    # Prepare output list
    output = []

    # Process each sample
    for episode, samples in episodes.items():
        for i, sample in enumerate(samples):
            # Get the trajectory from the current sample to the last sample of the episode
            traj = samples[i:]
            n = len(traj)
            print("i:", i, "n:", n)
            # import pdb;pdb.set_trace()
            if n < 5:
                # Use all available points
                line = [s['point'] for s in traj]
                # Pad with the last point until the line has exactly 5 points
                while len(line) < 5:
                    line.append(line[-1])
            else:
                # Otherwise, sample 5 points evenly along the trajectory:
                # starting at the current point and ending at the final point
                indices = np.linspace(0, n - 1, num=5, dtype=int)
                line = [traj[idx]['point'] for idx in indices]

            # Append the result for the current sample
            output.append({
                'image': sample['image'],
                'line': line
            })

    # save_json_path = json_path[:-5] + "_line.json"
    save_json_path = output_json.replace("_point.json", "_line.json")

    # Optionally, save the output to a new JSON file
    with open(save_json_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Connected lines saved to {save_json_path}")
    
    # # For debugging, print the output
    # for item in output:
    #     print(item)
