import os
import re
import time
import math
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoConfig, AutoTokenizer, Qwen3Model
import gc

from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
    visual,
)
from utils.augmentations import masked_data
from utils.metrics import metric

warnings.filterwarnings("ignore")


def print_selected_gradients(model):
    last_n = 3
    encoder_layer_grads = defaultdict(list)
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if 'head' in name:
                print(f"[HEAD] {name:50s} | grad mean: {param.grad.abs().mean():.4e}, std: {param.grad.std():.4e}")
            if 'layers.' in name:
                try:
                    layer_id = int(name.split("layers.")[1].split(".")[0])
                    encoder_layer_grads[layer_id].append(param.grad.view(-1))
                except:
                    continue
    if encoder_layer_grads:
        max_layer_id = max(encoder_layer_grads.keys())
        for layer_id in range(max(max_layer_id - last_n + 1, 0), max_layer_id + 1):
            if layer_id in encoder_layer_grads:
                grads = torch.cat(encoder_layer_grads[layer_id])
                print(f"[ENCODER LAYER {layer_id:2d}] grad mean: {grads.abs().mean():.4e}, std: {grads.std():.4e}")

def plot_loss_curve(train_losses, val_losses, step_interval, save_path, current_step):
    steps = list(range(step_interval, current_step + 1, step_interval))
    plt.figure()
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.xlabel("Steps"); plt.ylabel("Loss")
    plt.title(f"Loss Curve at Step {current_step}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"loss_step_{current_step}.png"))
    plt.close()

def plot_lr_curve(lr_list, step_interval, save_path, current_step, epoch):
    steps = [i * step_interval for i in range(len(lr_list))]
    plt.figure()
    plt.plot(steps, lr_list, label='Learning Rate')
    plt.xlabel('Step'); plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Curve - Epoch {epoch}')
    plt.legend(); plt.grid(True); plt.tight_layout()
    save_file = os.path.join(save_path, f"lr_curve_epoch{epoch}.png")
    plt.savefig(save_file)
    plt.close()

def verify_module_weights_match(module_a, module_b, rtol=1e-5, atol=1e-8, verbose=True):
    mismatches, matched, total = [], 0, 0
    state_a = dict(module_a.named_parameters())
    state_b = dict(module_b.named_parameters())

    for name in state_a:
        if name not in state_b:
            mismatches.append((name, "missing in B"))
            continue
        param_a = state_a[name].detach().cpu()
        param_b = state_b[name].detach().cpu()
        total += 1
        if torch.allclose(param_a, param_b, rtol=rtol, atol=atol):
            matched += 1
        else:
            mismatches.append((name, "value mismatch"))

    if verbose:
        print(f"\n🧪 Checked {total} parameters: {matched} matched, {len(mismatches)} mismatched.")
        for name, reason in mismatches:
            print(f"  ✗ {name}: {reason}")
        if matched == total:
            print("✅ All parameters match exactly!")
        else:
            print("⚠️ Some parameters are mismatched.")
    return len(mismatches) == 0

def verify_first_two_layers_match(encoder: torch.nn.Module,
                                  bid_encoder: torch.nn.Module,
                                  rtol: float = 0.0,
                                  atol: float = 0.0,
                                  check_unshared: bool = True,
                                  verbose: bool = True) -> bool:
    enc = encoder.module if hasattr(encoder, "module") else encoder
    bid = bid_encoder.module if hasattr(bid_encoder, "module") else bid_encoder

    assert hasattr(enc, "layers") and hasattr(bid, "layers"), "Both encoders must have `.layers`."
    assert len(enc.layers) >= 2 and len(bid.layers) >= 2

    ok, diffs = True, []
    for i in range(2):
        enc_layer = enc.layers[i]
        bid_layer = bid.layers[i]
        enc_sd = enc_layer.state_dict()
        bid_sd = bid_layer.state_dict()

        enc_keys, bid_keys = set(enc_sd.keys()), set(bid_sd.keys())
        missing_in_bid = sorted(list(enc_keys - bid_keys))
        missing_in_enc = sorted(list(bid_keys - enc_keys))
        if missing_in_bid or missing_in_enc:
            ok = False
            diffs.append((i, "keys_mismatch", {"missing_in_bid": missing_in_bid, "missing_in_enc": missing_in_enc}))
        for k in sorted(list(enc_keys & bid_keys)):
            t1, t2 = enc_sd[k], bid_sd[k]
            if t1.shape != t2.shape or t1.dtype != t2.dtype:
                ok = False
                diffs.append((i, k, f"shape/dtype mismatch: {t1.shape},{t1.dtype} vs {t2.shape},{t2.dtype}"))
                continue
            if check_unshared and hasattr(t1, "data") and hasattr(t2, "data"):
                if t1.data_ptr() == t2.data_ptr():
                    ok = False
                    diffs.append((i, k, "unexpected shared storage (same data_ptr)"))
            if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
                ok = False
                max_diff = (t1 - t2).abs().max().item()
                diffs.append((i, k, f"values differ (max_abs_diff={max_diff:.3e})"))

    if verbose:
        if ok:
            print("[verify_first_two_layers_match] PASS: encoder.layers[0:2] == bid_encoder.layers[0:2]")
        else:
            print("[verify_first_two_layers_match] FAIL:")
            for item in diffs:
                layer_idx, what, detail = item
                print(f"  - layer {layer_idx}: {what} -> {detail}")
    return ok


class Exp_My_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_My_Model, self).__init__(args)
        self.writer = SummaryWriter("./outputs/logs") if self.is_main() else None
        self.args = args
        self.patch_len = args.patch_len

    def is_main(self):
        return getattr(self.args, "rank", 0) == 0

    def _maybe_tqdm(self, loader, desc):
        return tqdm(loader, desc=desc) if self.is_main() else loader

    def reduce_mean(self, scalar, device):
        if getattr(self.args, "world_size", 1) == 1 or not dist.is_initialized():
            return scalar
        t = torch.tensor([scalar], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return t.item()

    def _build_model(self):
        if self.args.downstream_task == "forecast":
            model = self.model_dict[self.args.model].Model(self.args).float()
        elif self.args.downstream_task == "classification":
            model = self.model_dict[self.args.model].ClsModel(self.args).float()
        else:
            raise ValueError(f"Unsupported downstream_task: {self.args.downstream_task}")

        if self.args.load_checkpoints:
            import re
            print(f"Loading pre-trained checkpoint from: {self.args.load_checkpoints}")
            ckpt = torch.load(self.args.load_checkpoints, map_location='cpu')
            pretrained_state = ckpt["model_state_dict"]

            model = transfer_weights(
                weights_path=self.args.load_checkpoints,
                model=model,
                exclude_head=True,
                device='cpu'
            )

        enc = getattr(model, "encoder", None)
        if enc is not None and hasattr(enc, "get_input_embeddings"):
            enc_emb = enc.get_input_embeddings()
            if enc_emb is not None:
                for p in enc_emb.parameters():
                    p.requires_grad = False

        bid = getattr(model, "bid_encoder", None)
        if bid is not None and hasattr(bid, "get_input_embeddings"):
            bid_emb = bid.get_input_embeddings()
            if bid_emb is not None:
                for p in bid_emb.parameters():
                    p.requires_grad = False

        print("number of trainable params",
            sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    def _get_data(self, flag):
        llm_path = self.args.llm_path
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True, use_fast=True)

        dataset, loader = data_provider(self.args, flag, llm_model=self.model.encoder, tokenizer=tokenizer)
        with torch.no_grad():
            dataset.get_all_embeddings()
    
        del tokenizer

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if getattr(self.args, "world_size", 1) > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.args.rank,
                shuffle=(flag == "train"),
                drop_last=(flag == "train"),
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=loader.batch_size,
                sampler=sampler,
                num_workers=getattr(loader, "num_workers", self.args.num_workers),
                pin_memory=True,
                drop_last=(flag == "train"),
            )
        return dataset, loader

    def _select_optimizer(self):
        params = (p for p in self.model.parameters() if p.requires_grad)
        return optim.AdamW(params, lr=self.args.learning_rate, weight_decay=0.01)

    def _select_criterion(self):
        if (self.args.task_name == "finetune" and self.args.downstream_task == "classification"):
            criterion = nn.CrossEntropyLoss()
            if self.is_main(): print("Using CrossEntropyLoss")
        else:
            criterion = nn.MSELoss()
            if self.is_main(): print("Using MSELoss")
        return criterion

    def pretrain(self):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if self.is_main():
            os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optim, gamma=self.args.lr_decay)
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        accumulation_steps = max(1, self.args.accumulation_steps)

        for epoch in range(self.args.train_epochs):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_losses = []
            loader = self._maybe_tqdm(train_loader, f"Training Epoch {epoch+1}/{self.args.train_epochs}")

            if self.is_main():
                print("Current learning rate: {:.7f}".format(scheduler.get_last_lr()[0]))

            model_optim.zero_grad(set_to_none=True)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                if i == 0:
                    print(f"[Rank {self.args.rank}] "
                        f"local batch_size={batch_x.size(0)}, "
                        f"global batch_size={batch_x.size(0) * self.args.world_size}")
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    pred_x = self.model(batch_x)
                    loss = criterion(pred_x, batch_x)

                loss_to_backward = loss / accumulation_steps
                scaler.scale(loss_to_backward).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(model_optim)
                    scaler.update()
                    model_optim.zero_grad(set_to_none=True)

                epoch_losses.append(loss.detach().item())

            if (i + 1) % accumulation_steps != 0:
                scaler.step(model_optim); scaler.update()
                model_optim.zero_grad(set_to_none=True)

            scheduler.step()

            vali_loss = self.valid_one_epoch(vali_loader)
            vali_loss = self.reduce_mean(vali_loss, self.device)
            train_loss = float(np.mean(epoch_losses))
            train_loss = self.reduce_mean(train_loss, self.device)

            if self.is_main():
                print(f"Epoch: {epoch+1}/{self.args.train_epochs} | Train Loss: {train_loss:.6f}, Vali Loss: {vali_loss:.6f}")
                if self.writer:
                    self.writer.add_scalars("/pretrain_loss", {"train_loss": train_loss, "vali_loss": vali_loss}, epoch)

                state = {
                    "epoch": epoch,
                    "model_state_dict": (self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()),
                }
                torch.save(state, os.path.join(path, f"ckpt{epoch+1}.pth"))

    def valid_one_epoch(self, vali_loader):
        self.model.eval()
        criterion = self._select_criterion()
        losses = []
        loader = self._maybe_tqdm(vali_loader, "Validation")
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.args.use_amp):
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                pred_x = self.model(batch_x)
                loss = criterion(pred_x, batch_x)
                losses.append(loss.item())
        return float(np.mean(losses)) if len(losses) else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if self.is_main():
            os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=self.is_main())
        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()

        total_steps = len(train_loader) * self.args.train_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be less than total_steps")

        def warmup_cosine(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=model_optim, lr_lambda=warmup_cosine)

        step_losses, step_val_losses, lr_list = [], [], []
        global_step = 0
        validstep = len(train_loader)

        accumulation_steps = max(1, self.args.accumulation_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)

        best_val_loss = float('inf')

        for epoch in range(self.args.train_epochs):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            iter_count = 0
            epoch_loss_vals = []
            loader = self._maybe_tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

            if self.is_main():
                print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]["lr"]))

            self.model.train()
            start_time = time.time()
            model_optim.zero_grad(set_to_none=True)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(loader):
                iter_count += 1
                global_step += 1

                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                text_emb = train_data.get_text_embeddings(index).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    loss = self.model(batch_x, batch_y, text_emb=text_emb)

                loss_to_backward = loss / accumulation_steps
                scaler.scale(loss_to_backward).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(model_optim)
                    scaler.update()
                    model_optim.zero_grad(set_to_none=True)

                epoch_loss_vals.append(loss.detach().item())
                if global_step % validstep == 0:
                    avg_train_loss_local = float(np.mean(epoch_loss_vals))
                    avg_train_loss = self.reduce_mean(avg_train_loss_local, self.device)

                    self.model.eval()
                    with torch.no_grad():
                        val_loss_local = self.valid(vali_data, vali_loader, model_criteria)
                    self.model.train()

                    val_loss = self.reduce_mean(val_loss_local, self.device)

                    if self.is_main():
                        step_losses.append(avg_train_loss)
                        step_val_losses.append(val_loss)
                        lr_list.append(model_optim.param_groups[0]["lr"])

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(
                                (self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()),
                                os.path.join(path, "checkpoint.pth")
                            )

                scheduler.step()

            if (i + 1) % accumulation_steps != 0:
                scaler.step(model_optim); scaler.update()
                model_optim.zero_grad(set_to_none=True)

            if self.is_main():
                plot_loss_curve(step_losses, step_val_losses, step_interval=validstep, save_path=path, current_step=global_step)
                plot_lr_curve(lr_list, step_interval=validstep, save_path=path, current_step=global_step, epoch=epoch + 1)

            train_loss_epoch_local = float(np.mean(epoch_loss_vals))
            train_loss_epoch = self.reduce_mean(train_loss_epoch_local, self.device)
            vali_loss_local = self.valid(vali_data, vali_loader, model_criteria)
            vali_loss = self.reduce_mean(vali_loss_local, self.device)
            test_loss_local = self.valid(vali_data, test_loader, model_criteria)
            test_loss = self.reduce_mean(test_loss_local, self.device)

            end_time = time.time()
            if self.is_main():
                print(
                    f"Epoch: {epoch+1}, Steps: {len(train_loader)}, Time: {end_time - start_time:.2f}s | "
                    f"Train Loss: {train_loss_epoch:.7f} "
                    f"Val  Loss: {vali_loss:.7f} "
                    f"Test Loss: {test_loss:.7f}"
                )
                log_path = os.path.join(path, "log.txt")
                with open(log_path, "a") as log_file:
                    log_file.write(
                        f"Epoch: {epoch+1}, Steps: {len(train_loader)}, Time: {end_time - start_time:.2f}s | "
                        f"Train Loss: {train_loss_epoch:.7f} Val Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}\n"
                    )

            if self.is_main():
                early_stopping(vali_loss, self.model.module if hasattr(self.model, "module") else self.model, path=path)
                stop_now = early_stopping.early_stop
            else:
                stop_now = False
            if getattr(self.args, "world_size", 1) > 1 and dist.is_initialized():
                stop_tensor = torch.tensor([1 if stop_now else 0], device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())

            if stop_now:
                if self.is_main():
                    print("Early stopping")
                break

        if self.is_main():
            best_model_path = os.path.join(path, "checkpoint.pth")
            if os.path.exists(best_model_path):
                state = torch.load(best_model_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" not in state:
                    self.model.load_state_dict(state if not hasattr(self.model, "module") else {k: v for k, v in state.items()}, strict=False)
                else:
                    self.model.load_state_dict(state["state_dict"])
        if getattr(self.args, "world_size", 1) > 1 and dist.is_initialized():
            dist.barrier()

        self.lr = scheduler.get_last_lr()[0]
        return self.model

    def valid(self, vali_data, vali_loader, model_criteria):
        self.model.eval()
        losses = []
        loader = self._maybe_tqdm(vali_loader, "Validation")

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.args.use_amp):
            for batch_x, batch_y, batch_x_mark, batch_y_mark, index in loader:
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                text_emb = vali_data.get_text_embeddings(index).to(self.device)
                loss = self.model(batch_x, batch_y, text_emb=text_emb)
                losses.append(loss.item())
        loss_mean_local = float(np.mean(losses)) if len(losses) else 0.0
        loss_mean = self.reduce_mean(loss_mean_local, self.device)
        self.model.train()
        return loss_mean

    def test(self):
        test_data, test_loader = self._get_data(flag="test")
        preds, trues = [], []
        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if self.is_main():
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        loader = self._maybe_tqdm(test_loader, "Testing")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                text_emb = test_data.get_text_embeddings(index).to(self.device)

                if hasattr(self.model, "module"):
                    self.model.module.pred_len = self.args.pred_len
                else:
                    self.model.pred_len = self.args.pred_len
                pred_x = self.model(batch_x, text_emb=text_emb)

                f_dim = -1 if self.args.features == "MS" else 0
                pred_x = pred_x[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = pred_x.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                if self.is_main() and i % 10 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    file_name = os.path.join(folder_path, f"{self.args.pred_len}_{i}_pred_vs_true.pdf")
                    visual(gt, pd, file_name)

        if self.is_main():
            preds = np.concatenate(preds, axis=0) if len(preds) else np.array([])
            trues = np.concatenate(trues, axis=0) if len(trues) else np.array([])
            if preds.size and trues.size:
                mae, mse, _, _, _ = metric(preds, trues)
                print("{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(self.args.input_len, self.args.pred_len, mse, mae))
                args = self.args
                setting = "{}_{}_{}_{}_il{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_ps{}_text{}".format(
                    args.task_name, args.model, args.data, args.features, args.input_len, args.label_len,
                    args.pred_len, args.d_model, args.d_ff, args.n_heads, args.e_layers, args.d_layers, args.factor,
                    args.dropout, args.head_dropout, args.train_epochs, args.batch_size, args.learning_rate, args.patch_len,
                    args.text
                )
                with open(os.path.join(folder_path, "score.txt"), "a") as f:
                    f.write("{0}->{1}, {2:.3f}, {3:.3f},{4} \n".format(self.args.input_len, self.args.pred_len, mse, mae, setting))
        if getattr(self.args, "world_size", 1) > 1 and dist.is_initialized():
            dist.barrier()
