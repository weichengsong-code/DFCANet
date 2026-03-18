import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import cv2
import math

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label

from models.DFCANet import DFCANet




def denormalize_img_tensor(img: torch.Tensor, config: dict):
    """
    img: (3,H,W) tensor, 通常是 Normalize 后的
    优先从 config 里找 mean/std；找不到就按 [0,1] clamp 处理
    return: uint8 RGB numpy (H,W,3)
    """
    x = img.detach().float().cpu()

    mean = None
    std = None
    # 常见字段名（你们项目可能不同）
    for mk, sk in [
        ("image_mean", "image_std"),
        ("mean", "std"),
        ("pixel_mean", "pixel_std"),
    ]:
        if mk in config and sk in config:
            mean, std = config[mk], config[sk]
            break

    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        x = x * std + mean

    x = x.clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()  # HWC, RGB
    return x


def overlay_heatmap_on_rgb(rgb_uint8: np.ndarray, heatmap_2d: np.ndarray,
                           alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET, blur_ksize: int = 21):
    """
    rgb_uint8: (H,W,3) RGB uint8
    heatmap_2d: (h,w) 或 (H,W) float
    return: overlay_rgb_uint8, heatmap_color_rgb_uint8
    """
    H, W = rgb_uint8.shape[:2]
    hm = heatmap_2d.astype(np.float32)

    hm_min, hm_max = float(hm.min()), float(hm.max())
    if hm_max - hm_min < 1e-12:
        hm = np.zeros_like(hm, dtype=np.float32)
    else:
        hm = (hm - hm_min) / (hm_max - hm_min)

    hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)

    if blur_ksize and blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        hm = cv2.GaussianBlur(hm, (blur_ksize, blur_ksize), 0)

    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    hm_color_bgr = cv2.applyColorMap(hm_u8, colormap)          # BGR
    hm_color_rgb = cv2.cvtColor(hm_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = (rgb_uint8.astype(np.float32) * (1 - alpha) + hm_color_rgb.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay, hm_color_rgb


class ViTGradCAM:
    """
    在 ViT 最后一层 block 输出上做 Grad-CAM：
      - activations: (B, N, C), N含CLS + patches
      - 对 target logit 反传，取 patch tokens 的梯度/特征生成 cam
    """
    def __init__(self, vit_model, target_module=None):
        self.vit = vit_model
        self.activations = None

        # 默认 hook 到最后一个 block 的输出
        if target_module is None:
            if hasattr(self.vit, "blocks") and len(self.vit.blocks) > 0:
                target_module = self.vit.blocks[-1]
            else:
                raise ValueError("找不到 vit.blocks，无法做 ViTGradCAM。请检查 model.visual_encoder 结构。")

        def forward_hook(module, inp, out):
            # out: (B, N, C)
            self.activations = out
            self.activations.retain_grad()

        self.hook = target_module.register_forward_hook(forward_hook)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __call__(self, forward_fn, target_score: torch.Tensor):
        """
        forward_fn: 一个“已经完成 forward 并返回 logits 的函数”，这里只需要调用 backward 所以 forward 已经做过也行
        target_score: 标量 tensor，比如 logits[0,1]
        return: cam (num_patches,) -> reshape 成 (S,S)
        """
        # 反传
        target_score.backward(retain_graph=True)

        act = self.activations  # (B,N,C)
        if act is None or act.grad is None:
            raise RuntimeError("没有捕获到 activations/grad，确认 forward 在 hook 生效后执行，并且启用了梯度。")

        # 取第一个样本
        act0 = act[0]           # (N,C)
        grad0 = act.grad[0]     # (N,C)

        # 去掉 CLS token
        act_p = act0[1:, :]     # (P,C)
        grad_p = grad0[1:, :]   # (P,C)

        # channel 权重：对 tokens 平均
        weights = grad_p.mean(dim=0)  # (C,)
        cam = (act_p * weights).sum(dim=1)  # (P,)
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()

        # reshape 成方阵
        P = cam_np.shape[0]
        S = int(math.sqrt(P))
        if S * S != P:
            # 非方阵则退化：尽量按接近方阵 reshape
            S = int(round(math.sqrt(P)))
            S = max(1, S)
            # pad 到 S*S
            pad = S * S - P
            if pad > 0:
                cam_np = np.pad(cam_np, (0, pad), mode="constant", constant_values=0)
            cam_np = cam_np[: S * S]
        cam_np = cam_np.reshape(S, S)

        # 归一化
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 1e-12:
            cam_np = cam_np / cam_np.max()
        return cam_np


def visualize_gradcam_samples(args, model, data_loader, tokenizer, device, config, out_dir, max_vis=50):
    """
    保存 Grad-CAM 叠加图到 out_dir：
      overlay_*.jpg / heatmap_*.jpg / raw_*.jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # 只对主进程保存，避免分布式重复写
    if hasattr(utils, "is_main_process") and (not utils.is_main_process()):
        return

    # 准备 ViTGradCAM：假设视觉编码器在 model.visual_encoder
    if not hasattr(model, "visual_encoder"):
        print("[GradCAM] model 没有 visual_encoder，无法可视化（请检查 DFCANet 实现）。")
        return

    cam_extractor = ViTGradCAM(model.visual_encoder)

    saved = 0
    for step, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(data_loader):
        if saved >= max_vis:
            break

        # 逐样本处理更稳（避免 batch 内不同 text/label 结构复杂）
        bs = image.size(0)
        for b in range(bs):
            if saved >= max_vis:
                break

            img_b = image[b:b+1].to(device, non_blocking=True)
            label_b = [label[b]]  # 你的 forward 里 label 是 list[str]
            text_b = [text[b]]
            fib_b = fake_image_box[b:b+1].to(device) if torch.is_tensor(fake_image_box) else fake_image_box
            # fake_word_pos 保持在 CPU，避免 numpy 报错
            fwp_b = fake_word_pos[b:b+1].cpu() if torch.is_tensor(fake_word_pos) else fake_word_pos


            text_input = tokenizer(
                text_b, max_length=128, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_token_type_ids=False
            )
            text_input, fake_token_pos, _ = text_input_adjust(text_input, fwp_b, device)

            # 关键：启用梯度
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits_real_fake, logits_multicls, output_coord, logits_tok = model(
                    img_b, label_b, text_input, fib_b, fake_token_pos, is_train=False
                )

                # 以“fake 概率对应的 logit”为目标（class=1）
                # 你如果想看 real，就换成 [0,0]
                target_score = logits_real_fake[0, 1]

                cam_2d = cam_extractor(forward_fn=None, target_score=target_score)

            # 把输入图还原为 RGB uint8
            rgb = denormalize_img_tensor(img_b[0], config)

            overlay, hm_color = overlay_heatmap_on_rgb(rgb, cam_2d, alpha=0.5, blur_ksize=21)

            # 命名
            tag = str(label_b[0])
            fname = f"{saved:05d}_{tag}"

            cv2.imwrite(os.path.join(out_dir, f"raw_{fname}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f"heatmap_{fname}.jpg"), cv2.cvtColor(hm_color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f"overlay_{fname}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            saved += 1

    cam_extractor.remove()
    print(f"[GradCAM] saved {saved} visualizations to: {out_dir}")






def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as DFCANet is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)
        
        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)
        
        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch

  

@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
    
    TP_all_multicls = np.zeros(4, dtype = int)
    TN_all_multicls = np.zeros(4, dtype = int)
    FP_all_multicls = np.zeros(4, dtype = int)
    FN_all_multicls = np.zeros(4, dtype = int)
    F1_multicls = np.zeros(4)

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        for cls_idx in range(logits_multicls.shape[1]):
            cls_pred = logits_multicls[:, cls_idx]
            cls_pred[cls_pred>=0]=1
            cls_pred[cls_pred<0]=0
            
            TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
            TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
            FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
            FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()
            
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
                    
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)

        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()
                 
    ##================= real/fake cls ========================## 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)
    # ##================= token cls========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)
    ##================= multi-label cls ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
            
    for cls_idx in range(logits_multicls.shape[1]):
        Precision_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx])
        Recall_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx])
        F1_multicls[cls_idx] = 2*Precision_multicls*Recall_multicls / (Precision_multicls + Recall_multicls)            

    return AUC_cls, ACC_cls, EER_cls, \
        MAP.item(), OP, OR, OF1, CP, CR, CF1, F1_multicls, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num, 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)
    
    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    #### Model #### 
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    if args.log:
        print(f"Creating MAMMER")
    model = DFCANet(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    
    model = model.to(device)   

    checkpoint_dir = '/bvg/code/MultiModal-DeepFake/results/Z/checkpoint_best.pth'
    checkpoint = torch.load(checkpoint_dir, map_location='cpu') 
    state_dict = checkpoint['model']                       

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
                   
    # model.load_state_dict(state_dict)  
    if args.log:
        print('load checkpoint from %s'%checkpoint_dir)  
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)  

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)
    
    if args.distributed:  
        samplers = create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None]

    val_loader = create_loader([val_dataset],
                                samplers,
                                batch_size=[config['batch_size_val']], 
                                num_workers=[4], 
                                is_trains=[False], 
                                collate_fns=[None])[0]

    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.log:
        print("Start evaluation")

    AUC_cls, ACC_cls, EER_cls, \
    MAP, OP, OR, OF1, CP, CR, CF1, F1_multicls, \
    IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
    ACC_tok, Precision_tok, Recall_tok, F1_tok  = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)
    
    
    # ====== Grad-CAM Heatmap Visualization ======
    if getattr(args, "vis_heatmap", False):
        vis_dir = os.path.join(log_dir, "gradcam_vis")
        visualize_gradcam_samples(
            args=args,
            model=model_without_ddp,
            data_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            config=config,
            out_dir=vis_dir,
            max_vis=getattr(args, "vis_num", 50),
        )

    
    #============ evaluation info ============#
    val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                    "ACC_cls": "{:.4f}".format(ACC_cls*100),
                    "EER_cls": "{:.4f}".format(EER_cls*100),
                    "MAP": "{:.4f}".format(MAP*100),
                    "OP": "{:.4f}".format(OP*100),
                    "OR": "{:.4f}".format(OR*100),
                    "OF1": "{:.4f}".format(OF1*100),
                    "CP": "{:.4f}".format(CP*100),
                    "CR": "{:.4f}".format(CR*100),
                    "CF1": "{:.4f}".format(CF1*100),
                    "F1_FS": "{:.4f}".format(F1_multicls[0]*100),
                    "F1_FA": "{:.4f}".format(F1_multicls[1]*100),
                    "F1_TS": "{:.4f}".format(F1_multicls[2]*100),
                    "F1_TA": "{:.4f}".format(F1_multicls[3]*100),
                    "IOU_score": "{:.4f}".format(IOU_score*100),
                    "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                    "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                    "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                    "ACC_tok": "{:.4f}".format(ACC_tok*100),
                    "Precision_tok": "{:.4f}".format(Precision_tok*100),
                    "Recall_tok": "{:.4f}".format(Recall_tok*100),
                    "F1_tok": "{:.4f}".format(F1_tok*100),
    }
    
    if utils.is_main_process(): 
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': args.test_epoch,
                    }             
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='/bvg/code/MultiModal-DeepFake/results')
    parser.add_argument('--text_encoder', default='/bvg/bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)
    parser.add_argument('--vis_heatmap', action='store_true', help='save Grad-CAM heatmaps')
    parser.add_argument('--vis_num', type=int, default=50, help='how many samples to visualize')
    parser.add_argument('--vis_alpha', type=float, default=0.5, help='overlay alpha')


    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
 
    main_worker(0, args, config)