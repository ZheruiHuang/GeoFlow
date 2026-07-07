import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import argparse
from datetime import datetime
from pprint import pprint
import numpy as np
import torch

from dataset import ODDataset, load_train_valid_test_areas
from utils.metrics import cal_od_metrics, average_listed_metrics
from utils.utils import set_seed, build_dataloader, lst_to_device, pack_list_to_device
from model.prediction import PredODGModel, PredictiveTrainer
from model.generation import FMODGModel, FlowMatchingTrainer


@torch.no_grad()
def validate(trainer, dataloader, device, mode, sample_steps=None):
    trainer.model.eval()

    batch_metrics = []
    for batch in dataloader:
        feat_lst = lst_to_device(batch["feat_list"], device)
        centroid_lst = lst_to_device(batch["centroid_list"], device)
        adj_pack_lst = pack_list_to_device(batch["adj_pack_list"], device)
        od_list = batch["od_list"]
        mask_list = batch["mask_list"]
        od_scaler_list = batch["od_scaler_list"]

        if mode == "generation":
            assert sample_steps is not None and sample_steps > 0
            pred_list, _ = trainer.sample(feat_lst, centroid_lst, adj_pack_lst, n_step=sample_steps)
        else:
            pred_list = trainer.predict(feat_lst, centroid_lst, adj_pack_lst)

        cur_metrics = []
        if mask_list is None:
            mask_iter = [None] * len(pred_list)
        else:
            mask_iter = mask_list
        for pred, gt, scaler, mask in zip(pred_list, od_list, od_scaler_list, mask_iter):
            pred_np = pred.detach().cpu().numpy()
            gt_np = gt.numpy()
            pred_np = np.clip(pred_np, 0, None) * scaler
            gt_np = np.clip(gt_np, 0, None) * scaler
            if mask is not None:
                mask_np = mask.detach().cpu().numpy()
                pos_mask = (gt_np > 0) & (mask_np > 0)
                m = cal_od_metrics(pred_np.copy(), gt_np.copy(), mask=pos_mask)
            else:
                m = cal_od_metrics(pred_np.copy(), gt_np.copy(), mask=None)
            cur_metrics.append(m)
        batch_metrics.extend(cur_metrics)

    return average_listed_metrics(batch_metrics)


def make_config(mode, exp_dir, device):
    seed = 0
    if mode == "prediction":
        return {
            "mode": "prediction",
            "device": device,
            "seed": seed,
            "epochs": 999,
            "lr": 1e-3,
            "batch_size": 64,
            "grad_accum_steps": 1,
            "hid_dim": 32,
            "encode_mlp_layers": 1,
            "encode_gattn_layers": 1,
            "attn_layers": 1,
            "axial_attn_layers": 1,
            "attn_heads": 4,
            "dropout": 0.0,
            "patience": 51,
            "model_save": os.path.join(exp_dir, "best_ckpt.pt"),
            "test_result_file": os.path.join(exp_dir, "test_results.json"),
            "main_metric": "NegCPC",
        }
    elif mode == "generation":
        return {
            "mode": "generation",
            "device": device,
            "seed": seed,
            "epochs": 999,
            "lr": 1e-4,
            "batch_size": 64,
            "grad_accum_steps": 1,
            "sample_steps": 25,
            "hid_dim": 32,
            "time_dim": 32,
            "encode_mlp_layers": 1,
            "encode_gattn_layers": 1,
            "attn_layers": 1,
            "axial_attn_layers": 1,
            "attn_heads": 4,
            "dropout": 0.0,
            "patience": 51,
            "model_save": os.path.join(exp_dir, "best_ckpt.pt"),
            "test_result_file": os.path.join(exp_dir, "test_results.json"),
            "main_metric": "NegCPC",
        }


def main():
    parser = argparse.ArgumentParser()
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--prediction", action="store_true")
    task_group.add_argument("--generation", action="store_true")
    parser.add_argument("--dataset-root", default=".",
                        help="Directory containing data/ and dataset/ split files")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override model/random seed")
    args = parser.parse_args()
    mode = "generation" if args.generation else "prediction"

    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dataset_root = os.path.normpath(args.dataset_root)
    config_output_root = config_dataset_root
    data_root = os.path.abspath(args.dataset_root)
    output_root = data_root
    exp_root = "exps/generation" if mode == "generation" else "exps/prediction"
    seed_value = args.seed if args.seed is not None else 0
    exp_name = f"{exp_time}_seed{seed_value}"
    exp_dir = os.path.join(output_root, exp_root, exp_name)
    config_exp_dir = os.path.join(config_output_root, exp_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    config = make_config(mode, config_exp_dir, device=args.device)
    config["dataset_root"] = config_dataset_root
    config["output_root"] = config_output_root
    if args.lr is not None: config["lr"] = args.lr
    if args.batch_size is not None: config["batch_size"] = args.batch_size
    if args.epochs is not None: config["epochs"] = args.epochs
    if args.seed is not None: config["seed"] = args.seed
    pprint(config)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    set_seed(config["seed"])
    device = torch.device(config["device"])

    train_areas, valid_areas, test_areas = load_train_valid_test_areas(root_dir=data_root)
    train_dataset = ODDataset(root_dir=data_root, filter=train_areas)
    valid_dataset = ODDataset(root_dir=data_root, filter=valid_areas)
    test_dataset = ODDataset(root_dir=data_root, filter=test_areas)

    sample_item = train_dataset[0]
    feat_dim = sample_item["feats"].shape[-1]
    config["feat_dim"] = feat_dim
    print(f"{feat_dim = }")

    train_loader = build_dataloader(train_dataset, config["batch_size"], shuffle=True)
    valid_loader = build_dataloader(valid_dataset, config["batch_size"], shuffle=False)
    test_loader = build_dataloader(test_dataset, config["batch_size"], shuffle=False)

    if mode == "generation":
        model = FMODGModel(
            feat_dim=feat_dim,
            hid_dim=config["hid_dim"],
            time_dim=config["time_dim"],
            encode_mlp_layers=config["encode_mlp_layers"],
            encode_gattn_layers=config["encode_gattn_layers"],
            attn_heads=config["attn_heads"],
            attn_layers=config["attn_layers"],
            axial_attn_layers=config["axial_attn_layers"],
            dropout=config["dropout"],
        )
        trainer = FlowMatchingTrainer(
            model, lr=config["lr"], device=device,
            grad_accum_steps=config["grad_accum_steps"]
        )
    else:
        model = PredODGModel(
            feat_dim=feat_dim,
            hid_dim=config["hid_dim"],
            encode_mlp_layers=config["encode_mlp_layers"],
            encode_gattn_layers=config["encode_gattn_layers"],
            attn_heads=config["attn_heads"],
            attn_layers=config["attn_layers"],
            axial_attn_layers=config["axial_attn_layers"],
            dropout=config["dropout"],
        )
        trainer = PredictiveTrainer(
            model, lr=config["lr"], device=device,
            grad_accum_steps=config["grad_accum_steps"]
        )

    best_metric = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        trainer.model.train()
        epoch_loss = 0.0
        step = 0
        for batch in train_loader:
            feat_lst = lst_to_device(batch["feat_list"], device)
            centroid_lst = lst_to_device(batch["centroid_list"], device)
            adj_pack_lst = pack_list_to_device(batch["adj_pack_list"], device)
            od_lst = lst_to_device(batch["od_list"], device)
            mask_lst = batch["mask_list"]
            if mask_lst is not None:
                mask_lst = lst_to_device(mask_lst, device)

            if mode == "prediction":
                loss = trainer.train_step(feat_lst, centroid_lst, adj_pack_lst, od_lst, mask_lst)
            else:
                loss = trainer.train_step(feat_lst, centroid_lst, adj_pack_lst, od_lst, mask_lst)

            epoch_loss += loss
            step += 1
        epoch_loss /= step
        print(f"[{mode.upper()}][Epoch {epoch}] Train Avg Loss: {epoch_loss:.4f}")

        val_metrics = validate(
            trainer, valid_loader, device, mode,
            sample_steps=config["sample_steps"] if mode == "generation" else None
        )
        torch.cuda.empty_cache()
        print(f"[{mode.upper()}][Epoch {epoch}] Valid Metrics:")
        pprint(val_metrics)

        main_key = config["main_metric"]
        current_metric = val_metrics[main_key]
        improve = (best_metric is None) or (current_metric < best_metric)
        if improve:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            torch.save(trainer.model.state_dict(), config["model_save"])
            print(f"\t-> best performance achieved. {main_key} = {current_metric:.4f}")
        else:
            patience_counter += 1
            print(f"\t -> early stop patience {patience_counter} / {config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early stopping")
                break

    if best_epoch > 0:
        trainer.model.load_state_dict(torch.load(config["model_save"], map_location=device))
        print(f"load best model (at epoch {best_epoch}) for testing")

    test_metrics = validate(
        trainer, test_loader, device, mode,
        sample_steps=config["sample_steps"] if mode == "generation" else None
    )
    print(f"[{mode.upper()}][Test] Metrics:")
    pprint(test_metrics)
    with open(config["test_result_file"], "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()
