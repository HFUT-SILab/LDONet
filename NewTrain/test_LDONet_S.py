"""
Test LDONet-S direct-trained model and extract features for metrics evaluation
"""

import os
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.dataset import MyDataset
from models.LDONet_S import LDONet_S


def test_model(model, data_loader, device):
    """Test model and extract normalized fused features."""
    model.eval()
    correct = 0
    total = 0
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            fused_features, _ = model.processing(data)
            logits = model.arcface(fused_features, None)
            features = F.normalize(fused_features, p=2, dim=1, eps=1e-8)

            preds = logits.data.max(dim=1, keepdim=True)[1]
            correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()
            total += data.size(0)

            all_features.append(features.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = (100.0 * correct) / total
    all_features = np.concatenate(all_features, axis=0)

    return accuracy, all_features, np.array(all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LDONet-S Direct Model")

    parser.add_argument("--dataset", type=str, default="Blue", choices=["Blue", "Green", "NIR", "Red", "HFUT", "PolyU", "Tongji"])
    parser.add_argument("--num_classes", type=int, default=250)
    parser.add_argument("--test_set_file", type=str, required=True)


    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LDONet-S checkpoint")

    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for extracted features")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda")

    print("=" * 60)
    print("Testing LDONet-S Direct Model")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of classes: {args.num_classes}")
    print("=" * 60)

    print(f"\nLoading test dataset: {args.test_set_file}")
    testset = MyDataset(txt=args.test_set_file, transforms=None, train=False, imside=128, outchannels=1)
    data_loader_test = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )

    print("\nLoading model...")
    model = LDONet_S(label_num=args.num_classes)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully (strict mode)")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("Warning: Size mismatch detected, using non-strict loading")
                model.load_state_dict(checkpoint, strict=False)
                print("Model loaded successfully (non-strict mode)")
            else:
                raise
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params / 1e6:.2f}M")

    print(f"\nTesting model on {len(testset)} samples...")
    accuracy, features, labels = test_model(model, data_loader_test, device)

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}%")
    print(f"Total samples: {len(testset)}")
    print(f"Features shape: {features.shape}")
    print(f"{'=' * 60}")

    default_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        args.dataset,
        "LDONet_S",
        "rst_test",
    )
    result_dir = args.output_dir if args.output_dir else default_output_dir
    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "features.npy"), features)
    np.save(os.path.join(result_dir, "labels.npy"), labels)

    with open(os.path.join(result_dir, "test_accuracy.txt"), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}%\\n")
        f.write(f"Total Samples: {len(testset)}\\n")

    print(f"\nFeatures saved to: {result_dir}/features.npy")
    print(f"Labels saved to: {result_dir}/labels.npy")
    print(f"Features shape: {features.shape}")
