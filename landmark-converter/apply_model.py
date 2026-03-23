"""Apply trained landmark converter models to new data.

This script loads a trained keypoint converter model and applies it to images
in a COCO JSON file, generating visualizations of the predicted keypoints.

Example:
    $ python apply_model.py --model_path model.pth --coco_json data.json --output_dir results/
"""

import argparse
import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import defaultdict

from src.models import KeypointConverterMLPWithAttention
from utils.data_utils import normalize_keypoints_bbox, denormalize_keypoints_bbox


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Loads a model and its corresponding training arguments from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth).
        device (torch.device): The device to load the model onto.

    Returns:
        tuple: A tuple containing:
            - torch.nn.Module: The loaded and initialized model in evaluation mode.
            - argparse.Namespace: The training arguments stored in the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    train_args = checkpoint.get('args')
    if train_args is None:
        raise ValueError("Checkpoint does not contain 'args'. Cannot instantiate model.")

    # Determine the number of keypoints the model was trained to predict.
    num_keypoints_to_predict = train_args.num_target_kpt - train_args.target_kpt_slice_idx

    print("Instantiating KeypointConverterMLPWithAttention model...")
    model = KeypointConverterMLPWithAttention(
        num_source_kpts=train_args.num_source_kpt,
        num_target_kpts=num_keypoints_to_predict,
        embed_dim=train_args.embed_dim,
        num_heads=train_args.num_heads,
        mlp_hidden_dim=train_args.mlp_hidden_dim
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    
    return model, train_args

def predict_keypoints(model, source_kpts_xy, bbox, device):
    """
    Generates keypoint predictions for a single sample.

    Auto-detects if the input keypoints are in a non-upright orientation
    (e.g., from an upside-down camera) and canonicalizes to upright before
    prediction. The converter model was trained on upright faces only and
    is NOT rotation-invariant — it learns position-specific mappings in
    bbox-normalized [0,1] space.

    Args:
        model (torch.nn.Module): The trained model.
        source_kpts_xy (np.ndarray): A numpy array of source keypoints, shape (num_kpts, 2).
        bbox (np.ndarray): The bounding box [x, y, w, h] for normalization.
        device (torch.device): The device to perform inference on.

    Returns:
        np.ndarray: A numpy array of predicted keypoints in the original image scale.
    """
    source_kpts_tensor = torch.from_numpy(source_kpts_xy).unsqueeze(0).to(device).float()
    bbox_tensor = torch.from_numpy(bbox).unsqueeze(0).to(device).float()

    with torch.no_grad():
        norm_kpts = normalize_keypoints_bbox(source_kpts_tensor, bbox_tensor)

        # Auto-detect orientation from keypoint layout.
        # In upright faces: eyes (kpts 36-47) have smaller Y than mouth (kpts 48-67),
        # and L-eye (36-41) has smaller X than R-eye (42-47).
        # If both are inverted, the image is 180-degree rotated.
        norm_np = norm_kpts.squeeze(0).cpu().numpy()  # (K, 2)
        n_kpts = norm_np.shape[0]

        needs_rot180 = False
        if n_kpts == 68:  # Only auto-detect for 68-kpt source
            eyes_y = norm_np[36:48, 1].mean()
            mouth_y = norm_np[48:68, 1].mean()
            leye_x = norm_np[36:42, 0].mean()
            reye_x = norm_np[42:48, 0].mean()
            y_inverted = eyes_y > mouth_y  # eyes below mouth
            x_inverted = leye_x > reye_x   # L-eye right of R-eye
            needs_rot180 = y_inverted and x_inverted

        if needs_rot180:
            # Rotate 180 in normalized [0,1] space: flip both axes
            norm_kpts[..., 0] = 1.0 - norm_kpts[..., 0]
            norm_kpts[..., 1] = 1.0 - norm_kpts[..., 1]

        input_kpts_flat = norm_kpts.contiguous().view(1, -1)
        predicted_kpts_norm_structured = model(input_kpts_flat)

        if needs_rot180:
            # Undo rotation on output
            predicted_kpts_norm_structured[..., 0] = 1.0 - predicted_kpts_norm_structured[..., 0]
            predicted_kpts_norm_structured[..., 1] = 1.0 - predicted_kpts_norm_structured[..., 1]

        denormalized_preds = denormalize_keypoints_bbox(
            predicted_kpts_norm_structured, bbox_tensor
        )

    return denormalized_preds.squeeze(0).cpu().numpy()

def plot_keypoints_on_ax(ax, keypoints, instance_color='r', show_indices=True):
    """Helper function to plot keypoints on a matplotlib axis."""
    ax.scatter(keypoints[:, 0], keypoints[:, 1], color=[instance_color], s=15, edgecolors='black', linewidths=0.5)
    if show_indices:
        for i, (x, y) in enumerate(keypoints):
            ax.text(x + 2, y + 2, str(i), color='white', fontsize=6,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

def generate_visualization(image_path, source_kpts_list, predicted_kpts_list):
    """
    Creates a matplotlib figure with a side-by-side visualization.

    Args:
        image_path (str): Path to the image file.
        source_kpts_list (list of np.ndarray): List of source keypoints arrays.
        predicted_kpts_list (list of np.ndarray): List of predicted keypoints arrays.

    Returns:
        matplotlib.figure.Figure or None: The generated figure, or None on error.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image at {image_path}. Skipping visualization.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Warning: Error loading image {image_path}: {e}. Skipping visualization.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.suptitle(os.path.basename(image_path), fontsize=14)

    num_instances = len(source_kpts_list)
    
    # Updated colormap usage to resolve MatplotlibDeprecationWarning
    cmap = plt.colormaps.get('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, num_instances)]

    # Setup axes
    axes[0].imshow(img)
    axes[0].set_title(f'Source Keypoints ({num_instances} instances)', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(img)
    axes[1].set_title(f'Predicted Keypoints ({num_instances} instances)', fontsize=10)
    axes[1].axis('off')

    # Plot all instances
    for i in range(num_instances):
        plot_keypoints_on_ax(axes[0], source_kpts_list[i][:, :2], instance_color=colors[i])
        plot_keypoints_on_ax(axes[1], predicted_kpts_list[i], instance_color=colors[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def main(args):
    """
    Main function to drive the inference and visualization pipeline.
    """
    # Setup device
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None and torch.cuda.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model, train_args = load_model_from_checkpoint(args.model_path, device)

    # Load COCO Data
    print(f"Loading COCO annotations from: {args.coco_json}")
    with open(args.coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping from image ID to its file path for quick lookup
    # Assumes 'file_name' in the COCO JSON is a full, absolute path
    image_map = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Limit the number of images to process if specified
    image_ids_to_process = list(annotations_by_image.keys())
    if args.num_images is not None and args.num_images > 0:
        if args.num_images < len(image_ids_to_process):
            # Randomly sample image IDs
            image_ids_to_process = random.sample(image_ids_to_process, args.num_images)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing {len(image_ids_to_process)} images. Visualizations will be saved to {args.output_dir}")
    
    # Prepare for COCO output if requested
    output_coco_data = None
    converted_annotations = []
    if args.output_coco_json:
        import copy
        output_coco_data = copy.deepcopy(coco_data)
        # Update categories if present - update number of keypoints
        if "categories" in output_coco_data and len(output_coco_data["categories"]) > 0:
            for cat in output_coco_data["categories"]:
                if "keypoints" in cat:
                    # Determine number of target keypoints from model
                    num_target_kpts = model.num_target_kpts
                    # Update keypoint names
                    if len(cat["keypoints"]) >= num_target_kpts:
                        cat["keypoints"] = cat["keypoints"][:num_target_kpts]
                    else:
                        cat["keypoints"] = [f"kpt_{i}" for i in range(num_target_kpts)]
                    cat["num_keypoints"] = num_target_kpts
    
    # Process each image
    for image_id in tqdm(image_ids_to_process, desc="Processing Images"):
        if image_id not in image_map:
            print(f"Warning: Skipping image_id {image_id} because it's not found in image map.")
            continue
        
        image_filename = image_map[image_id]
        image_path = os.path.join(args.image_dir, image_filename)
        image_annotations = annotations_by_image[image_id]

        all_source_kpts = []
        all_predicted_kpts = []

        # Generate predictions for all annotations of the current image
        for ann in image_annotations:
            if args.source_kpt_field not in ann:
                print(f"Warning: Source keypoint field '{args.source_kpt_field}' not in annotation {ann['id']}. Skipping this annotation.")
                continue

            # Extract source keypoints and bounding box
            source_kpts = np.array(ann[args.source_kpt_field]).reshape(-1, 3)
            bbox = np.array(ann['bbox'])
            
            # Run prediction
            predicted_kpts = predict_keypoints(model, source_kpts[:, :2], bbox, device)
            
            all_source_kpts.append(source_kpts)
            all_predicted_kpts.append(predicted_kpts)
            
            # Store for COCO output if requested
            if args.output_coco_json:
                import copy
                new_ann = copy.deepcopy(ann)
                # Format predicted keypoints for COCO (x, y, visibility)
                predicted_kpts_with_vis = np.column_stack([
                    predicted_kpts[:, 0],  # x
                    predicted_kpts[:, 1],  # y
                    np.ones(len(predicted_kpts)) * 2  # visibility = 2 (visible)
                ])
                # Flatten to COCO format
                new_ann["keypoints"] = predicted_kpts_with_vis.flatten().tolist()
                new_ann["num_keypoints"] = int(len(predicted_kpts))
                # Store original keypoints for reference
                new_ann["original_keypoints"] = ann[args.source_kpt_field]
                converted_annotations.append(new_ann)

        # If any valid predictions were made for this image, generate and save visualizations
        if all_predicted_kpts:
            fig = generate_visualization(image_path, all_source_kpts, all_predicted_kpts)
            
            if fig is None:
                continue
            
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save individual plots based on flags
            if args.save_png:
                png_path = os.path.join(args.output_dir, f"{base_filename}_prediction.png")
                fig.savefig(png_path)
            elif args.save_pdf:
                pdf_path = os.path.join(args.output_dir, f"{base_filename}_prediction.pdf")
                fig.savefig(pdf_path)

            plt.close(fig) # Close figure after all saving operations
            
    # Save COCO JSON if requested
    if args.output_coco_json and output_coco_data:
        output_coco_data["annotations"] = converted_annotations
        # Add metadata about conversion
        if "info" not in output_coco_data:
            output_coco_data["info"] = {}
        output_coco_data["info"]["conversion_info"] = {
            "model_checkpoint": os.path.basename(args.model_path),
            "source_keypoints": train_args.num_source_kpt,
            "target_keypoints": model.num_target_kpts,
            "total_annotations": len(coco_data["annotations"]),
            "converted_annotations": len(converted_annotations)
        }
        
        output_json_path = os.path.join(args.output_dir, args.output_coco_json)
        with open(output_json_path, 'w') as f:
            json.dump(output_coco_data, f, indent=2)
        print(f"\n✓ COCO JSON saved to: {output_json_path}")
        print(f"  Converted {len(converted_annotations)} annotations")
    
    print(f"\nProcessing complete. Output saved in {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Apply a trained keypoint converter model to a COCO JSON file and generate a PDF visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- File Paths ---
    parser.add_argument("--model_path", type=str, 
                        required=True,
                        help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--coco_json", type=str, 
                        required=True,
                        help="Path to the input COCO JSON file with source keypoints.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory where the images are stored.")
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs/inference/",
                        help="Directory to save the output visualization files.")
    
    # --- Configuration ---
    parser.add_argument("--source_kpt_field", type=str, default="keypoints",
                        help="The key in the COCO annotation for the source keypoints array.")
    parser.add_argument("--gpu_id", type=int, default=None, 
                        help="Specific GPU ID to use. If not set, uses default CUDA device if available, otherwise CPU.")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Maximum number of images to process. If not set, all images will be processed.")
    
    # --- Plotting Arguments ---
    parser.add_argument("--save_png", action='store_true',
                        help="Save visualization as a PNG file instead of the default PDF.")
    parser.add_argument('--save_pdf', action='store_true', default=True,
                        help='Save visualization as a PDF file (default behavior).')
    parser.add_argument('--no_pdf', dest='save_pdf', action='store_false',
                        help='Do not save visualization as a PDF file.')
    
    # --- COCO Output Arguments ---
    parser.add_argument("--output_coco_json", type=str, default=None,
                        help="Filename for output COCO JSON with converted keypoints (saved in output_dir).")
    
    args = parser.parse_args()
    main(args) 