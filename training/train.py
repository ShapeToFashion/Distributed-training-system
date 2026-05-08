# train.py
#
# PURPOSE: Python training step run by each Go worker.
#
# DATASET:
#   ImageFolder format at --data path:
#     data_path/
#       class_a/ img1.jpg img2.jpg ...
#       class_b/ img1.jpg img2.jpg ...
#
# MODEL:
#   ResNet18 (torchvision) with final fc sized to dataset classes.
#
# IO CONTRACT:
#   - Reads weights from --weights JSON (flat float list) if present.
#   - Outputs JSON to stdout: {"gradients": [...], "loss": <float>, "num_parameters": <int>, "error": "..."}

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


def train_on_shard(data_path, batch_size, learning_rate, weights_path=None, partition_index=0, partition_count=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PYTHON] Using device: {device}", file=sys.stderr)

    if not os.path.exists(data_path):
        return {"error": f"Dataset path not found: {data_path}", "gradients": [], "loss": 0.0, "num_parameters": 0}

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    base_dataset = datasets.ImageFolder(data_path, transform=transform)
    if partition_count <= 0:
        partition_count = 1
    if partition_index < 0:
        partition_index = 0
    partition_index = partition_index % partition_count

    partitioned_indices = [i for i in range(len(base_dataset)) if (i % partition_count) == partition_index]
    if not partitioned_indices:
        return {
            "error": f"No samples in partition {partition_index}/{partition_count}",
            "gradients": [],
            "loss": 0.0,
            "num_parameters": 0,
        }

    dataset = torch.utils.data.Subset(base_dataset, partitioned_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(
        f"[PYTHON] Dataset: {len(dataset)} images in partition {partition_index}/{partition_count}, "
        f"{len(base_dataset.classes)} classes: {base_dataset.classes}",
        file=sys.stderr,
    )

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(base_dataset.classes))
    model = model.to(device)
    model.train()

    # Load flat weights into state_dict in state_dict() iteration order.
    # If the provided weights are missing/invalid/too short, fall back to random init.
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            flat = json.load(f)

        state_dict = model.state_dict()
        pointer = 0
        for key, target_tensor in state_dict.items():
            numel = target_tensor.numel()
            vals = flat[pointer : pointer + numel]
            if len(vals) < numel:
                print(
                    f"[PYTHON] WARNING: weights too short at key '{key}': need {numel}, got {len(vals)}. "
                    "Ignoring weights and using random init.",
                    file=sys.stderr,
                )
                pointer = 0
                state_dict = None
                break
            tensor = torch.tensor(vals, dtype=target_tensor.dtype).view(target_tensor.shape)
            state_dict[key] = tensor
            pointer += numel
        if state_dict is not None:
            if pointer != len(flat):
                print(f"[PYTHON] WARNING: weights.json has {len(flat) - pointer} extra values", file=sys.stderr)
            model.load_state_dict(state_dict)
            print(f"[PYTHON] Loaded weights.json ({pointer} values consumed)", file=sys.stderr)
    else:
        print("[PYTHON] No weights.json found — using random init", file=sys.stderr)

    criterion = nn.CrossEntropyLoss()
    loss = None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()
        break

    if loss is None:
        return {"error": "DataLoader was empty — no batches found", "gradients": [], "loss": 0.0, "num_parameters": 0}

    gradients = []
    params_by_name = dict(model.named_parameters())
    for key, tensor in model.state_dict().items():
        param = params_by_name.get(key)
        if param is not None and param.grad is not None:
            gradients.extend(param.grad.detach().cpu().view(-1).tolist())
        else:
            gradients.extend([0.0] * tensor.numel())

    return {
        "gradients": gradients,
        "loss": round(loss.item(), 6),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed CNN Training Worker Script")
    parser.add_argument("--data", required=True, help="Path to dataset folder (ImageFolder format)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (master applies it)")
    parser.add_argument("--weights", default="", help="Path to weights JSON file")
    parser.add_argument("--partition_index", type=int, default=0, help="Worker partition index")
    parser.add_argument("--partition_count", type=int, default=1, help="Total worker partitions")
    args = parser.parse_args()

    print(f"[PYTHON] Data path:  {args.data}", file=sys.stderr)
    print(f"[PYTHON] Batch size: {args.batch_size}", file=sys.stderr)
    print(f"[PYTHON] LR:         {args.lr}", file=sys.stderr)
    print(f"[PYTHON] Weights:    {args.weights or 'none'}", file=sys.stderr)
    print(f"[PYTHON] Partition:  {args.partition_index}/{args.partition_count}", file=sys.stderr)

    result = train_on_shard(
        args.data,
        args.batch_size,
        args.lr,
        args.weights,
        args.partition_index,
        args.partition_count,
    )
    if "error" in result:
        print(f"[PYTHON] ERROR: {result['error']}", file=sys.stderr)
    json.dump(result, sys.stdout)

