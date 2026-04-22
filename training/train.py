# train.py
#
# PURPOSE: This is the Python training script that each worker executes.
#
# HOW IT FITS IN THE SYSTEM:
# ──────────────────────────
#   1. Go worker receives a task from master (shard path, batch size, learning rate)
#   2. Go worker launches this script as a subprocess:
#        python train.py --data=dataset/train --batch_size=2 --lr=0.01 --weights=weights.json
#   3. This script:
#        a) Loads the ImageFolder dataset from data_path
#        b) Loads current model weights (weights.json → state_dict)
#        c) Runs one training batch (forward pass → loss → backward pass)
#        d) Outputs gradients as JSON to stdout
#   4. Go worker reads the JSON gradients and sends them to master via gRPC
#
# THE MODEL:
# ──────────
# ResNet18 — a real CNN pretrained architecture (weights loaded from master).
# Input: 224x224 RGB images (ImageFolder format)
# Output: class logits → CrossEntropyLoss
#
# Weight loading: weights.json (flat list) → reconstructed state_dict → model
# Gradient extraction: state_dict key order → flat list → JSON stdout

import argparse
import json
import sys
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


# ──────────────────────────────────────────────────────────
# TRAINING FUNCTION — called by the Go worker
# ──────────────────────────────────────────────────────────

def train_on_shard(data_path, batch_size, learning_rate, weights_path=None):
    """
    Train ResNet18 on one ImageFolder shard and return gradients.

    Args:
        data_path:     Path to ImageFolder-format dataset directory
                       (subdirs = class names, each subdir has images)
        batch_size:    Number of images per training batch
        learning_rate: Not applied here — master applies it during aggregation
        weights_path:  Path to weights.json (flat list from master)

    Returns:
        dict with:
            'gradients'      — flat list of floats (same order as state_dict)
            'loss'           — scalar float
            'num_parameters' — total trainable param count
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PYTHON] Using device: {device}", file=sys.stderr)

    # ─── Load Dataset (ImageFolder) ───────────────────────────────────────────
    # Expected structure:
    #   data_path/
    #     class_a/  img1.jpg  img2.jpg ...
    #     class_b/  img1.jpg  img2.jpg ...
    if not os.path.exists(data_path):
        return {"error": f"Dataset path not found: {data_path}", "gradients": [], "loss": 0.0}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),          # ResNet18 expects 224x224
        transforms.ToTensor(),                  # [0,255] uint8 → [0.0,1.0] float
        transforms.Normalize([0.5]*3, [0.5]*3) # Normalize to [-1, 1]
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   # Keep 0 for subprocess safety (Go launches this)
    )
    print(f"[PYTHON] Dataset: {len(dataset)} images, {len(dataset.classes)} classes: {dataset.classes}", file=sys.stderr)

    # ─── Load Model (ResNet18) ────────────────────────────────────────────────
    # MUST match the architecture used in Colab / master
    # fc layer replaced to match number of classes in this dataset
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)
    model.train()

    # ─── Load weights.json → reconstruct state_dict ───────────────────────────
    # CRITICAL: iteration order of state_dict keys MUST be identical
    # on every worker and on the master. Python 3.7+ dicts preserve
    # insertion order; PyTorch state_dict() is stable across the same
    # model definition. Do NOT sort or reorder keys here.
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            flat = json.load(f)  # flat list of floats from master

        state_dict = model.state_dict()
        pointer = 0

        for key, target_tensor in state_dict.items():   # ← same order as master
            numel = target_tensor.numel()
            vals  = flat[pointer:pointer + numel]

            if len(vals) < numel:
                return {
                    "error": f"weights.json too short at key '{key}': "
                             f"need {numel}, got {len(vals)}",
                    "gradients": [], "loss": 0.0
                }

            tensor = torch.tensor(vals, dtype=target_tensor.dtype) \
                          .view(target_tensor.shape)
            state_dict[key] = tensor
            pointer += numel

        if pointer != len(flat):
            print(f"[PYTHON] WARNING: weights.json has {len(flat)-pointer} extra values", file=sys.stderr)

        model.load_state_dict(state_dict)
        print(f"[PYTHON] Loaded weights.json ({pointer} values consumed)", file=sys.stderr)
    else:
        print("[PYTHON] No weights.json found — using random init", file=sys.stderr)

    # ─── Training — ONE batch only ────────────────────────────────────────────
    # Workers compute gradients for one batch; master aggregates and updates.
    criterion = nn.CrossEntropyLoss()
    loss = None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)          # Forward pass
        loss    = criterion(outputs, labels)

        model.zero_grad()                # Clear any stale gradients
        loss.backward()                  # Backprop — populates param.grad
        break                            # One batch per worker call

    if loss is None:
        return {"error": "DataLoader was empty — no batches found", "gradients": [], "loss": 0.0}

    # ─── Extract gradients (ORDER = state_dict keys) ───────────────────────────
    # Master must flatten weights in the SAME order when building weights.json.
    gradients = []
    params_by_name = dict(model.named_parameters())

    for key, tensor in model.state_dict().items():
        param = params_by_name.get(key)
        if param is not None and param.grad is not None:
            gradients.extend(param.grad.detach().cpu().view(-1).tolist())
        else:
            # For buffers or missing gradients, send zeros to keep alignment.
            print(f"[PYTHON] WARNING: no grad for '{key}', sending zeros", file=sys.stderr)
            gradients.extend([0.0] * tensor.numel())

    print(f"[PYTHON] Gradient vector size: {len(gradients)}", file=sys.stderr)
    print(f"[PYTHON] Loss: {loss.item():.6f}", file=sys.stderr)

    return {
        "gradients":      gradients,
        "loss":           round(loss.item(), 6),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


# ──────────────────────────────────────────────────────────
# MAIN — entry point when called from Go worker
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed CNN Training Worker Script")
    parser.add_argument("--data",       required=True,          help="Path to dataset folder (ImageFolder format)")
    parser.add_argument("--batch_size", type=int,  default=4,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weights",    default="",             help="Path to weights JSON file")
    args = parser.parse_args()

    # Status to stderr — Go reads stdout for JSON only
    print(f"[PYTHON] Data path:  {args.data}",       file=sys.stderr)
    print(f"[PYTHON] Batch size: {args.batch_size}", file=sys.stderr)
    print(f"[PYTHON] LR:         {args.lr}",         file=sys.stderr)
    print(f"[PYTHON] Weights:    {args.weights or 'none'}", file=sys.stderr)

    result = train_on_shard(args.data, args.batch_size, args.lr, args.weights)

    if "error" in result:
        print(f"[PYTHON] ERROR: {result['error']}", file=sys.stderr)

    # JSON to stdout — this is what Go reads
    json.dump(result, sys.stdout)