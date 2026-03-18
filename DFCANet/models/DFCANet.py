from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn
import os
import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_


class RadialFreqAttn(nn.Module):

    def __init__(self, bins=16):
        super().__init__()
        self.bins = bins
        self.mlp = nn.Sequential(
            nn.Linear(bins, bins * 2),
            nn.GELU(),
            nn.Linear(bins * 2, bins),
            nn.Sigmoid()
        )

    def forward(self, mag):  # mag: (B,C,H,Wf)
        B, C, H, Wf = mag.shape
        device = mag.device
        yy = torch.linspace(-1, 1, H, device=device).view(H, 1).expand(H, Wf)
        xx = torch.linspace(0, 1, Wf, device=device).view(1, Wf).expand(H, Wf)  
        r = torch.sqrt(xx * xx + yy * yy).clamp(0, 1)  # (H,Wf)
        bin_idx = torch.clamp((r * (self.bins - 1)).long(), 0, self.bins - 1) 
        mag_flat = mag.view(B, C, -1)  # (B,C,H*Wf)
        idx_flat = bin_idx.view(-1)    # (H*Wf,)
        oh = F.one_hot(idx_flat, num_classes=self.bins).float()  # (H*Wf,bins)
        denom = oh.sum(dim=0).clamp_min(1.0)                     # (bins,)
        # (B,C,bins) = (B,C,H*Wf) @ (H*Wf,bins) / denom
        bin_mean = torch.matmul(mag_flat, oh) / denom
        w = self.mlp(bin_mean)  # (B,C,bins)
        w_map = w[:, :, bin_idx]  
        return w_map


class DynamicFreqFusion(nn.Module):

    def __init__(self, nc=3, hidden=24, bins=16):
        super().__init__()
        self.freq_attn = RadialFreqAttn(bins=bins)
        self.freq_net = nn.Sequential(
            nn.Conv2d(2 * nc, hidden, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * nc, 1, 1, 0),
        )

        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.norm = nn.GroupNorm(1, 2 * nc)

    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='ortho')           
        real = x_freq.real
        imag = x_freq.imag
        mag = torch.sqrt(real * real + imag * imag + 1e-8)
        mag_log = torch.log1p(mag)
        w = self.freq_attn(mag_log)                     
        real_g = real * w
        imag_g = imag * w

        f = torch.cat([real_g, imag_g], dim=1)            
        f = self.norm(f)
        delta = self.freq_net(f)                         
        delta_real, delta_imag = delta[:, :C], delta[:, C:]
        real2 = real + delta_real
        imag2 = imag + delta_imag

        x_out = torch.complex(real2, imag2)
        x_rec = torch.fft.irfft2(x_out, s=(H, W), norm='ortho')

        return x + self.alpha * (x_rec - x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.5, dynamic_drop_prob=False):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.dynamic_drop_prob = dynamic_drop_prob  # Flag to enable dynamic drop probability

    def forward(self, x):
        if self.training:
            drop_prob = self.drop_prob
            if self.dynamic_drop_prob:
                # Example: Reduce drop_prob gradually during training (or based on other criteria)
                drop_prob = max(0.0, self.drop_prob * (1 - self.get_progress()))
            
            if drop_prob > 0.0:
                keep_prob = 1.0 - drop_prob
                random_tensor = torch.bernoulli(torch.full(x.shape[:-1], keep_prob, device=x.device))  # Efficient random tensor generation
                random_tensor = random_tensor.unsqueeze(-1)  # Make it broadcastable to match x shape
                output = x / keep_prob * random_tensor
                return output
        return x

    def get_progress(self):
        # Example: return training progress as a value between 0 and 1
        # This function can be customized to change the drop probability dynamically.
        return 0  # Placeholder for a dynamic value, e.g., based on epoch or training step


class ImageTextInteractionFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, drop_path_prob=0.0, use_dynamic_drop_prob=False):
        super().__init__()

        self.img2txt_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                                  dropout=dropout, batch_first=True)
        self.img2txt_norm = nn.LayerNorm(dim)

        self.txt2img_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                                  dropout=dropout, batch_first=True)
        self.txt2img_norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_prob=drop_path_prob, dynamic_drop_prob=use_dynamic_drop_prob)

    def forward(self, img_feats, txt_feats, img_pad_mask=None, txt_pad_mask=None):

        img2txt, _ = self.img2txt_attn(
            query=img_feats, key=txt_feats, value=txt_feats,
            key_padding_mask=txt_pad_mask
        )
        img_feats = self.img2txt_norm(img_feats + self.drop_path(img2txt))

        # Text queries Image
        txt2img, _ = self.txt2img_attn(
            query=txt_feats, key=img_feats, value=img_feats,
            key_padding_mask=img_pad_mask
        )
        txt_feats = self.txt2img_norm(txt_feats + self.drop_path(txt2img))

        # FFN (separate residual)
        img_feats = self.ffn_norm(img_feats + self.drop_path(self.ffn(img_feats)))
        txt_feats = self.ffn_norm(txt_feats + self.drop_path(self.ffn(txt_feats)))

        return img_feats, txt_feats


class CrossModalAttentionFusion(nn.Module):
    """
    A stack of cross-modal fusion blocks.
    It can optionally project image/text features into a common fusion_dim.
    """
    def __init__(self, vision_dim, text_dim, fusion_dim=None, num_heads=12,
                 dropout=0.1, drop_path_prob=0.0, num_layers=4):
        super().__init__()

        if fusion_dim is None:
            # default: require same dim
            fusion_dim = vision_dim
        self.fusion_dim = fusion_dim

        # If dims mismatch, project to fusion_dim, then (optionally) project back
        self.need_proj = (vision_dim != fusion_dim) or (text_dim != fusion_dim)
        if self.need_proj:
            self.img_in = nn.Linear(vision_dim, fusion_dim)
            self.txt_in = nn.Linear(text_dim, fusion_dim)
            self.img_out = nn.Linear(fusion_dim, vision_dim)
            self.txt_out = nn.Linear(fusion_dim, text_dim)
        else:
            self.img_in = self.txt_in = self.img_out = self.txt_out = nn.Identity()

        self.layers = nn.ModuleList([
            ImageTextInteractionFusionBlock(dim=fusion_dim, num_heads=num_heads,
                                  dropout=dropout, drop_path_prob=drop_path_prob)
            for _ in range(num_layers)
        ])

        self.final_img_norm = nn.LayerNorm(vision_dim)
        self.final_txt_norm = nn.LayerNorm(text_dim)

    def forward(self, image_embeds, text_embeds, image_atts=None, text_atts=None, text_pad_mask=None):

        img_pad_mask = None
        if image_atts is not None:
            img_pad_mask = (image_atts == 0)

        if text_pad_mask is None and text_atts is not None:
            text_pad_mask = (text_atts == 0)

        # project in
        img = self.img_in(image_embeds)
        txt = self.txt_in(text_embeds)

        # stacked fusion
        for blk in self.layers:
            img, txt = blk(img, txt, img_pad_mask=img_pad_mask, txt_pad_mask=text_pad_mask)

        # project out (restore original dims)
        img = self.img_out(img)
        txt = self.txt_out(txt)

        # final norms
        img = self.final_img_norm(img)
        txt = self.final_txt_norm(txt)

        return img, txt


class VisualPromptEnhancer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=dropout
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, text_feats, vis_feats, vis_mask=None):
        # Cross-attention
        vis_ctx, _ = self.cross_attn(
            query=text_feats,
            key=vis_feats,
            value=vis_feats,
            key_padding_mask=vis_mask
        )

        gate = self.gate(torch.cat([text_feats, vis_ctx], dim=-1))
        text_feats = text_feats + gate * vis_ctx
        text_feats = self.norm1(text_feats)

        # FFN refine
        text_feats = text_feats + self.ffn(text_feats)
        return self.norm2(text_feats)



    
    
class TaskConditionalFusion(nn.Module):
    def __init__(self, hidden_size=768, num_tasks=5):
        super().__init__()

        self.task_embed = nn.Embedding(num_tasks, hidden_size)

        self.task_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text_feats, task_id):
        """
        task_id: int tensor (bs,)
        """
        task_vec = self.task_embed(task_id)              
        task_vec = task_vec.unsqueeze(1)                 
        task_vec = task_vec.expand_as(text_feats)    

        gate = self.task_gate(torch.cat([text_feats, task_vec], dim=-1))
        text_feats = text_feats + gate * task_vec

        return self.norm(text_feats)


  



class DFCANet(nn.Module):
    def __init__(
        self,
        args=None,
        config=None,
        text_encoder=None,
        tokenizer=None,
        init_deit=True
    ):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        embed_dim = config['embed_dim']

        # ==================== Visual Encoder ====================
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        if init_deit:
            ckpt_path = "/bvg/code/MultiModal-DeepFake/deit_base_patch16_224-b5f2ef4d.pth"
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"DeiT checkpoint not found: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint["model"]
            state_dict['pos_embed'] = interpolate_pos_embed(
                state_dict['pos_embed'], self.visual_encoder
            )
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        vision_width = config['vision_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertForTokenClassification.from_pretrained(
            text_encoder,
            config=bert_config,
            label_smoothing=config['label_smoothing']
        )

        text_width = self.text_encoder.config.hidden_size

        self.visual_prompt_enhancer = VisualPromptEnhancer(hidden_size=text_width)

        self.task_conditional_fusion = TaskConditionalFusion(
            hidden_size=text_width,
            num_tasks=5         
        )

        # ==================== Projection Heads ====================
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        # ==================== Task Heads ====================
        self.itm_head = self.build_mlp(text_width, 2)
        self.bbox_head = self.build_mlp(text_width, 4)
        self.cls_head = self.build_mlp(text_width, 4)

        # ==================== Momentum Models ====================
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)

        self.text_encoder_m = BertForTokenClassification.from_pretrained(
            text_encoder,
            config=bert_config,
            label_smoothing=config['label_smoothing']
        )

        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]

        self.copy_params()

        # ==================== Queues ====================
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

        # ==================== Localization Modules ====================
        self.norm_layer_aggr = nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))

        self.aggregator = nn.MultiheadAttention(
            text_width, 12, dropout=0.0, batch_first=True
        )

        self.norm_layer_it_cross_atten = nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(
            text_width, 12, dropout=0.0, batch_first=True

        fusion_layers = config.get('fusion_layers', 4)
        fusion_heads = config.get('fusion_heads', 12)
        fusion_dropout = config.get('fusion_dropout', 0.1)
        fusion_drop_path = config.get('fusion_drop_path', 0.0)
        fusion_dim = config.get('fusion_dim', text_width)  # 推荐默认 = text_width

        self.fusion_module = CrossModalAttentionFusion(
            vision_dim=vision_width,
            text_dim=text_width,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            drop_path_prob=fusion_drop_path,
            num_layers=fusion_layers
        )
        self.fusion_module_m = CrossModalAttentionFusion(
            vision_dim=vision_width,
            text_dim=text_width,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            drop_path_prob=fusion_drop_path,
            num_layers=fusion_layers
        )


        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)
        self.freq_aggregator = DynamicFreqFusion(nc=3, hidden=24, bins=16)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        target_bbox = target_bbox.to(output_coord.device, non_blocking=True)
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image, label, text, fake_image_box, fake_text_pos, alpha=0, is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)

            # ================= Task IDs =================
            TASK_MAC, TASK_BIC, TASK_MLC, TASK_IMG, TASK_TMG = range(5)

            # ================= Labels ===================
            multicls_label, real_label_pos = get_multi_label(label, image)

            # ================= Encoders =================
            image = self.freq_aggregator(image)
            image_embeds = self.visual_encoder(image)          # [B, N, Dv]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)

            text_output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode='text'
            )
            text_embeds = text_output.last_hidden_state         # [B, L, Dt]
            txt_pad_mask = (text.attention_mask == 0)


            if self.fusion_module is not None:
                image_embeds_fused, _ = self.fusion_module(
                    image_embeds=image_embeds,
                    text_embeds=text_embeds.detach(),
                    image_atts=image_atts,
                    text_atts=text.attention_mask,
                    text_pad_mask=txt_pad_mask
                )
            else:
                image_embeds_fused = image_embeds


            text_prompted = self.visual_prompt_enhancer(
                text_feats=text_embeds,                
                vis_feats=image_embeds_fused
            )


            task_id_bic = torch.full(
                (text_prompted.size(0),),
                TASK_BIC,
                dtype=torch.long,
                device=text_prompted.device
            )
            text_bic = self.task_conditional_fusion(
                text_feats=text_prompted,
                task_id=task_id_bic
            )


            image_feat = F.normalize(self.vision_proj(image_embeds_fused[:, 0]), dim=-1)
            text_feat  = F.normalize(self.text_proj(text_embeds[:, 0]), dim=-1)

            with torch.no_grad():
                self._momentum_update()

                image_embeds_m = self.visual_encoder_m(image)
                image_atts_m = torch.ones(image_embeds_m.size()[:-1], dtype=torch.long, device=image.device)

                text_output_m = self.text_encoder_m.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                    mode='text'
                )
                text_embeds_m = text_output_m.last_hidden_state

                if self.fusion_module_m is not None:
                    image_embeds_m_fused, _ = self.fusion_module_m(
                        image_embeds=image_embeds_m,
                        text_embeds=text_embeds_m.detach(),
                        image_atts=image_atts_m,
                        text_atts=text.attention_mask,
                        text_pad_mask=txt_pad_mask
                    )
                else:
                    image_embeds_m_fused = image_embeds_m

                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m_fused[:, 0]), dim=-1)
                text_feat_m  = F.normalize(self.text_proj_m(text_embeds_m[:, 0]), dim=-1)

                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
                text_feat_all  = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()], dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m  @ image_feat_all / self.temp

                sim_targets = torch.zeros_like(sim_i2t_m)
                sim_targets[real_label_pos, real_label_pos] = 1

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat  @ image_feat_all / self.temp

            loss_MAC = (
                -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean() +
                -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            ) / 2

            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            output_pos = self.text_encoder.bert(
                encoder_embeds=text_bic,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_fused,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion'
            )

            itm_labels = torch.ones(image.size(0), dtype=torch.long, device=image.device)
            itm_labels[real_label_pos] = 0
            loss_BIC = F.cross_entropy(self.itm_head(output_pos.last_hidden_state[:, 0]), itm_labels)

            loss_MLC = F.binary_cross_entropy_with_logits(
                self.cls_head(output_pos.last_hidden_state[:, 0]),
                multicls_label.float()
            )


            img_text_tokens = text_bic
            cls_tokens_local = self.cls_token_local.expand(image.size(0), -1, -1)

            local_feat_it = image_embeds_fused + self.it_cross_attn(
                query=self.norm_layer_it_cross_atten(image_embeds_fused),
                key=self.norm_layer_it_cross_atten(img_text_tokens),
                value=self.norm_layer_it_cross_atten(img_text_tokens),
                key_padding_mask=txt_pad_mask
            )[0]

            local_feat_aggr = self.aggregator(
                query=self.norm_layer_aggr(cls_tokens_local),
                key=self.norm_layer_aggr(local_feat_it[:, 1:]),
                value=self.norm_layer_aggr(local_feat_it[:, 1:])
            )[0]

            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)


            token_label = text.attention_mask[:, 1:].clone()
            token_label[token_label == 0] = -100
            token_label[token_label == 1] = 0

            for b, fake_pos in enumerate(fake_text_pos):
                for p in fake_pos:
                    if 0 <= p < token_label.size(1):
                        token_label[b, p] = 1

            token_out = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_fused,
                encoder_attention_mask=image_atts,
                labels=token_label,
                return_dict=True
            )

            loss_TMG = token_out.loss

            return loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC


        else:
            # Inference mode
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_embeds = text_output.last_hidden_state

            # Forward the positive image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds=text_embeds, attention_mask=text.attention_mask,
                                                encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
                                                return_dict=True, mode='fusion')

            ##================= IMG ========================## 
            bs = image.size(0)
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone()  # [:,1:] for ignoring class token
            local_feat_padding_mask_text = text_attention_mask_clone == 0  # 0 = pad token

            local_feat_it_cross_attn = image_embeds + self.it_cross_attn(
                query=self.norm_layer_it_cross_atten(image_embeds),
                key=self.norm_layer_it_cross_atten(text_embeds),
                value=self.norm_layer_it_cross_atten(text_embeds),
                key_padding_mask=local_feat_padding_mask_text
            )[0]

            local_feat_aggr = self.aggregator(
                query=self.norm_layer_aggr(cls_tokens_local),
                key=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :]),
                value=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :])
            )[0]

            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()

            ##================= BIC ========================## 
            logits_real_fake = self.itm_head(output_pos.last_hidden_state[:, 0, :])

            ##================= MLC ========================## 
            logits_multicls = self.cls_head(output_pos.last_hidden_state[:, 0, :])

            ##================= TMG ========================##   
            input_ids = text.input_ids.clone()
            logits_tok = self.text_encoder(input_ids, attention_mask=text.attention_mask,
                                        encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
                                        return_dict=True, return_logits=True)

            return logits_real_fake, logits_multicls, output_coord, logits_tok
    


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
