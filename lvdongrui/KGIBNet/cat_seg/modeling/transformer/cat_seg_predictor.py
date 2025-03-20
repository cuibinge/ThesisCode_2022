# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch
import pandas as pd

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates

import numpy as np
import open_clip
class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        kg_path: str,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        
        transformer = Aggregator(
            kg_path = kg_path,
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            classname=json.load(open(train_class_json, 'r')) if self.training else json.load(open(test_class_json, 'r'))
            )
        self.transformer = transformer
        
        self.tokens = None
        self.cache = None

        self.class_weights = nn.Parameter(torch.ones(1))
        self.kg_xlsx = kg_path
        self.tokens_kg = None
        self.num_kg = None
        self.cache_kg = None

        self.lambda_kg = nn.Parameter(torch.ones(1))
        # self.cross_attn = nn.MultiheadAttention(512, 8)
        # self.norm = nn.LayerNorm(512)

        # self.gate = nn.Sequential(
        #     nn.Linear(2 * 512, 512),
        #     nn.Sigmoid()
        # )


    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["kg_path"] = cfg.MODEL.SEM_SEG_HEAD.KG_PATH

        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        return ret

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text_emb = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        # print(text.shape)  # torch.Size([4, 1, 512])
        kg_emb = self.get_kg_embeds(text, self.kg_xlsx, self.clip_model, prompt)

        # # 交叉注意力+残差连接融合
        # text_emb_sq = text_emb.squeeze(1)
        # kg_emb_sq = kg_emb.squeeze(1)
        # attn_output, _ = self.cross_attn(text_emb_sq, kg_emb_sq, kg_emb_sq)
        # attn_output = self.norm(attn_output + text_emb_sq)
        # attn_output = attn_output.unsqueeze(1)

        # 门控机制
        # combined = torch.cat([text_emb, kg_emb], dim=-1)  # [类别数, 1, 1024]
        # gate = self.gate(combined)  # [类别数, 1, 512]
        # attn_output = gate * text_emb + (1 - gate) * kg_emb

        # 直接相加
        attn_output = text_emb + self.lambda_kg * kg_emb
        # attn_output = text_emb
        
        text = attn_output.repeat(x.shape[0], 1, 1, 1)
        out = self.transformer(x, text, vis)

        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        # print(classnames)
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings

    def get_kg_embeds(self, classnames, kg_path, clip_model, prompt=None):
        if self.cache_kg is not None and not self.training:
            return self.cache_kg
        if self.tokens_kg is None:
            tokens = []
            sentence_token_list = []
            classnames_num_dict = {}
            kg = pd.read_excel(kg_path, header=None)
            for classname in classnames:
                num = 0
                for index, row in kg.iterrows():
                    head, relation, tail = row[0], row[1], row[2]
                    if head == classname:
                        num += 1
                        classnames_num_dict[classname] = num
                        sentence = f"The {head} is {relation} the {tail}"
                        tokens_list = self.tokenizer(sentence).cuda() if self.tokenizer is not None else clip.tokenize(sentence).cuda()
                        sentence_token_list.append(tokens_list)
        
            tokens.append(sentence_token_list)

            if prompt is None:
                self.tokens_kg = tokens
                self.num_kg = classnames_num_dict
        elif self.tokens_kg is not None and prompt is None:
            tokens = self.tokens_kg
            classnames_num_dict = self.num_kg
        
        # print(classnames_num_dict)
        head_token = torch.stack([t.squeeze(0) for t in tokens[0]], dim=0)
        kg_embeddings = clip_model.encode_text(head_token, prompt)
        
        # 获取每个类别的加权
        grouped_avg_embeddings = []
        index = 0
        for classname in classnames:
            size = classnames_num_dict[classname]  # 获取该类别的三元组数量
            group = kg_embeddings[index:index + size]  # 获取该类别的三元组嵌入
            
            # 获取类别的自学习权重，并扩展为每个三元组的数量
            class_weights = self.class_weights.expand(size)  # 类别的学习权重
            weighted_group = group * class_weights.unsqueeze(1)  # 对该类别的三元组进行加权
            group_avg = weighted_group.mean(dim=0)  # 对嵌入进行平均化
            grouped_avg_embeddings.append(group_avg)
            index += size

        # 将所有类别的平均嵌入堆叠起来
        kg_embeddings = torch.stack(grouped_avg_embeddings)
        
        # 归一化每个类别的嵌入
        kg_embeddings = kg_embeddings / kg_embeddings.norm(dim=-1, keepdim=True)
        
        # 将嵌入的形状调整为 [类别数, 1, 512]
        kg_embeddings = kg_embeddings.unsqueeze(1)

        if not self.training:
            self.cache_kg = kg_embeddings
        
        return kg_embeddings

    def get_kg_embeds_(self, classnames, kg_path, clip_model, prompt=None):
        if self.cache_kg is not None and not self.training:
            return self.cache_kg
        if self.tokens_kg is None:
            tokens = []
            head_token_list = []
            relation_token_list = []
            tail_token_list = []
            classnames_num_dict = {}
            kg = pd.read_excel(kg_path, header=None)
            for classname in classnames:
                num = 0
                for index, row in kg.iterrows():
                    head, relation, tail = row[0], row[1], row[2]
                    if head == classname:
                        num += 1
                        classnames_num_dict[classname] = num
                        if self.tokenizer is not None:
                            head_token_list.append(self.tokenizer(head).cuda())
                            relation_token_list.append(self.tokenizer(relation).cuda())
                            tail_token_list.append(self.tokenizer(tail).cuda())
                        else:
                            head_token_list.append(clip.tokenize(head).cuda())
                            relation_token_list.append(clip.tokenize(relation).cuda())
                            tail_token_list.append(clip.tokenize(tail).cuda())
            tokens.append(head_token_list)
            tokens.append(relation_token_list)
            tokens.append(tail_token_list)
            if prompt is None:
                self.tokens_kg = tokens
                self.num_kg = classnames_num_dict
        elif self.tokens_kg is not None and prompt is None:
            tokens = self.tokens_kg
            classnames_num_dict = self.num_kg
        head_token = torch.stack([t.squeeze(0) for t in tokens[0]], dim=0)
        relation_token = torch.stack([t.squeeze(0) for t in tokens[1]], dim=0)
        tail_token = torch.stack([t.squeeze(0) for t in tokens[2]], dim=0)
        head_embeddings = clip_model.encode_text(head_token, prompt)
        relation_embeddings = clip_model.encode_text(relation_token, prompt)
        tail_embeddings = clip_model.encode_text(tail_token, prompt)
        kg_embeddings = head_embeddings + relation_embeddings - tail_embeddings
        grouped_avg_embeddings = []
        index = 0
        for size in classnames_num_dict.values():
            group = kg_embeddings[index:index + size]
            group_avg = group.mean(dim=0)
            grouped_avg_embeddings.append(group_avg)
            index += size
        kg_embeddings = torch.stack(grouped_avg_embeddings)
        kg_embeddings = kg_embeddings / kg_embeddings.norm(dim=-1, keepdim=True)
        kg_embeddings = kg_embeddings.unsqueeze(1)

        if not self.training:
            self.cache_kg = kg_embeddings
        return kg_embeddings