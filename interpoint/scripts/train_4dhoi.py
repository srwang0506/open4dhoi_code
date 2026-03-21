import argparse
import os
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]


from tqdm import tqdm
import wandb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset import collate_fn as ivd_collate_fn
from data.dataset_4dhoi import Custom4DHOIAlignedDataset
from data.transforms import get_train_transforms, get_val_transforms
from models import build_model
from models.losses import ChamferLoss


def freeze_all_but_transformer(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "transformer"):
        for p in model.transformer.parameters():
            p.requires_grad = True
    else:
        raise RuntimeError("Model has no transformer module to train.")
    if hasattr(model, "prediction_head"):
        for p in model.prediction_head.parameters():
            p.requires_grad = True
    else:
        raise RuntimeError("Model has no prediction_head module to train.")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    total_human = 0.0
    total_obj = 0.0
    total_neighbor = 0.0
    total_repulsion = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        rgb = batch["rgb_image"].to(device)
        obj_pts = batch["object_points"].to(device)
        human_labels = batch["human_labels"].to(device)
        object_coords = batch["object_coords"].to(device)

        outputs = model(rgb, obj_pts, return_aux=False)
        predictions = {
            "human_logits": outputs["human_logits"],
            "object_coords": outputs["object_coords"],
        }
        targets = {
            "human_labels": human_labels,
            "object_coords": object_coords,
            "object_mask": (human_labels > 0).float(),
            "object_points": obj_pts,
        }

        losses = model.compute_loss(predictions, targets, compute_aux=False)
        loss = losses["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        total_human += losses["human_loss"].item()
        total_obj += losses["object_loss"].item()
        total_neighbor += losses.get("object_neighbor_loss", torch.tensor(0.0)).item()
        total_repulsion += losses.get("object_repulsion_loss", torch.tensor(0.0)).item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "human": f"{losses['human_loss'].item():.4f}",
            "obj": f"{losses['object_loss'].item():.4f}",
        })

    n = max(1, len(loader))
    return {
        "train/loss": total / n,
        "train/human_loss": total_human / n,
        "train/object_loss": total_obj / n,
        "train/object_neighbor_loss": total_neighbor / n,
        "train/object_repulsion_loss": total_repulsion / n,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    total_human = 0.0
    total_obj = 0.0
    total_neighbor = 0.0
    total_repulsion = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        rgb = batch["rgb_image"].to(device)
        obj_pts = batch["object_points"].to(device)
        human_labels = batch["human_labels"].to(device)
        object_coords = batch["object_coords"].to(device)

        outputs = model(rgb, obj_pts, return_aux=False)
        predictions = {
            "human_logits": outputs["human_logits"],
            "object_coords": outputs["object_coords"],
        }
        targets = {
            "human_labels": human_labels,
            "object_coords": object_coords,
            "object_mask": (human_labels > 0).float(),
            "object_points": obj_pts,
        }

        losses = model.compute_loss(predictions, targets, compute_aux=False)
        loss = losses["total_loss"]
        total += loss.item()
        total_human += losses["human_loss"].item()
        total_obj += losses["object_loss"].item()
        total_neighbor += losses.get("object_neighbor_loss", torch.tensor(0.0)).item()
        total_repulsion += losses.get("object_repulsion_loss", torch.tensor(0.0)).item()

    n = max(1, len(loader))
    return {
        "val/loss": total / n,
        "val/human_loss": total_human / n,
        "val/object_loss": total_obj / n,
        "val/object_neighbor_loss": total_neighbor / n,
        "val/object_repulsion_loss": total_repulsion / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload_records", type=str, default="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/4dhoi_autorecon/upload_records.json")
    parser.add_argument("--data_root", type=str, default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/4dhoi_autorecon')
    parser.add_argument("--test_split_ratio", type=float, default=0.2)
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--project", type=str, default="ivd-4dhoi")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints/transformer_only")
    parser.add_argument("--use_lightweight_vlm", type=str, default=False)
    parser.add_argument("--train_vlm", type=str, default=False)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load (model_state_dict)")
    parser.add_argument("--use_knn_repulsion", action="store_true",
                        help="Enable KNN neighbor loss + repulsion (disables object regression loss)")
    parser.add_argument("--neighbor_k", type=int, default=20)
    parser.add_argument("--object_neighbor_loss", type=float, default=0)
    parser.add_argument("--object_repulsion_loss", type=float, default=0.1)
    parser.add_argument("--repulsion_sigma", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ.setdefault("WANDB_MODE", "offline")
    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    train_transform = get_train_transforms(image_size=224, render_size=256)
    val_transform = get_val_transforms(image_size=224, render_size=256)

    train_dataset = Custom4DHOIAlignedDataset(
        upload_records_path=args.upload_records,
        data_root=args.data_root,
        split="train",
        transform=train_transform,
    )
    val_dataset = Custom4DHOIAlignedDataset(
        upload_records_path=args.upload_records,
        data_root=args.data_root,
        split="test",
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ivd_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ivd_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    if train_loader.collate_fn is not ivd_collate_fn:
        raise RuntimeError(
            f"Unexpected collate_fn: {train_loader.collate_fn}. "
            "Expected data.dataset.collate_fn"
        )

    model = build_model({
        "d_tr": 256,
        "num_body_points": 87,
        "num_object_queries": 87,
        "use_lightweight_vlm": False,
        "device": str(device)
    }).to(device)
    model.loss_fn.object_loss = ChamferLoss(bidirectional=True)
    if args.use_knn_repulsion:
        model.loss_fn.lambda_object_repulsion = args.object_repulsion_loss
        model.loss_fn.repulsion_sigma = args.repulsion_sigma
        if args.object_neighbor_loss > 0:
            print("Warning: object_neighbor_loss is not implemented in current IVDLoss; only repulsion is active.")
    else:
        model.loss_fn.lambda_object_repulsion = 0.0

    if args.checkpoint:
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
    if args.train_vlm:
        # Unfreeze input embeddings so newly added special tokens can learn
        emb = model.vlm.model.get_input_embeddings()
        emb.weight.requires_grad = True
    else:
        freeze_all_but_transformer(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = validate(model, val_loader, device)
        scheduler.step()

        print(f"[Epoch {epoch}] "
              f"train_loss={train_stats['train/loss']:.4f} "
              f"val_loss={val_stats['val/loss']:.4f}")

        wandb.log({
            "epoch": epoch,
            **train_stats,
            **val_stats
        })

        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)


if __name__ == "__main__":
    main()
