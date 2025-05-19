import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from torchvision.models import vgg16, VGG16_Weights
from torch.nn.utils import spectral_norm, clip_grad_norm_


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt 
from collections import defaultdict 
from sklearn.model_selection import train_test_split
from torchinfo import summary 
from pathlib import Path
import wandb

os.environ["WANDB_API_KEY"] = "f9bd53ddbed845e1c532581b230e7da2dbc3673f" 

# --------------- Константы и Конфигурация ---------------
BATCH_SIZE = 16 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200    
Z_CH = 128     
PATCH_SHAPE = (448, 64)  

ALPHABET_STR = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
VOCAB_SIZE = len(ALPHABET_STR) + 1 # +1 для <PAD> токена (индекс 0)
CHAR_EMB_DIM = 128         # Размерность эмбеддинга одного символа
CHAR_RNN_HIDDEN_DIM = 256   # Размерность скрытого состояния RNN/GRU для текста (в одну сторону)
CHAR_RNN_LAYERS = 2         # Количество слоев RNN

# Параметры обучения
LR_G = 1e-4
LR_D = 5e-5
TARGET_KL_WEIGHT = 0.001    
GAN_WEIGHT = 0.07    
PERC_WEIGHT = 0.2   
RECON_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0

KL_ANNEAL_EPOCHS = 15  
START_KL_WEIGHT = 1e-7

# Параметры для ReduceLROnPlateau
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.95
SCHEDULER_PATIENCE = 15
SCHEDULER_THRESHOLD = 0.0001
SCHEDULER_MIN_LR = 1e-7

WANDB_PROJECT = "VAE-GAN"
WANDB_ENTITY = None 
WANDB_RUN_NAME = "unet_char_emb_v3" 
WANDB_SAVE_CODE = True


# --------------- ИЗМЕНЕННЫЙ CharacterTokenEncoder (возвращает пространственные признаки) ---------------
class CharacterTokenEncoder(nn.Module):
    def __init__(
        self,
        alphabet_str,
        emb_dim,
        rnn_hidden_dim,
        rnn_layers,
        target_feature_width,   # W_base_text
        target_feature_height=4 # H_base_text (новое!)
    ):
        super().__init__()
        self.alphabet = alphabet_str
        self.vocab_size = len(alphabet_str) + 1
        self.char_to_idx = {char: i + 1 for i, char in enumerate(alphabet_str)}
        self.pad_idx = 0

        self.embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=self.pad_idx)
        self.rnn = nn.GRU(
            emb_dim,
            rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if rnn_layers > 1 else 0
        )
        self.rnn_output_dim = rnn_hidden_dim * 2  # text_channels_in для SpatialFiLM

        self.target_feature_width = target_feature_width
        self.target_feature_height = target_feature_height

        # 1d conv для более локальных признаков по длине текста
        self.conv1d = nn.Conv1d(
            self.rnn_output_dim, self.rnn_output_dim, kernel_size=3, padding=1
        )

        # Learnable spatial positional encodings (по ширине W и по H)
        self.register_parameter(
            "pos_enc",
            nn.Parameter(
                torch.randn(1, self.rnn_output_dim, target_feature_height, target_feature_width) * 0.02
            )
        )

    def tokens_to_indices(self, text_list, max_len_chars):
        batch_indices = []
        for text in text_list:
            indices = [self.char_to_idx.get(char, self.pad_idx) for char in text]
            indices = indices[:max_len_chars]
            padded_indices = indices + [self.pad_idx] * (max_len_chars - len(indices))
            batch_indices.append(torch.tensor(padded_indices, dtype=torch.long))
        return torch.stack(batch_indices)

    def forward(self, texts_batch, max_len_chars_for_tokenization=60):
        """
        Returns: (B, rnn_output_dim, target_feature_height, target_feature_width)
        """
        # (B, L)
        indices_tensor = self.tokens_to_indices(texts_batch, max_len_chars_for_tokenization)
        indices_tensor = indices_tensor.to(self.embedding.weight.device)
        embedded_chars = self.embedding(indices_tensor)  # (B, L, EmbDim)
        rnn_outputs, _ = self.rnn(embedded_chars)        # (B, L, rnn_output_dim)

        # Permute for 1d conv: (B, rnn_output_dim, L)
        x = rnn_outputs.permute(0, 2, 1)
        x = self.conv1d(x)   # (B, rnn_output_dim, L)

        # Adaptive pooling по ширине до W; потом дублируем по высоте до H
        x = F.adaptive_avg_pool1d(x, self.target_feature_width)  # (B, rnn_output_dim, W)
        x = x.unsqueeze(2)  # (B, rnn_output_dim, 1, W)
        x = x.expand(-1, -1, self.target_feature_height, -1) # (B, rnn_output_dim, H, W)

        # Добавим learnable positional encoding: (B, C, H, W) + (1, C, H, W)
        x = x + self.pos_enc

        return x

# --------------- НОВЫЙ SpatialFiLMLayer ---------------
class SpatialFiLMLayer(nn.Module):
    def __init__(self, text_channels_in, num_features_main): # target_h/w больше не нужны здесь
        super().__init__()
        # Свертки для предсказания gamma и beta из пространственных текстовых признаков
        # Вход будет (B, text_channels_in, H_main, W_main) после interpolate
        # Выход должен быть (B, num_features_main * 2, H_main, W_main)
        self.param_predictor = nn.Sequential(
            nn.Conv2d(text_channels_in, text_channels_in, kernel_size=3, padding=1, bias=False), # bias=False если есть BN
            nn.BatchNorm2d(text_channels_in), # Можно добавить BN
            nn.ReLU(inplace=True),
            nn.Conv2d(text_channels_in, num_features_main * 2, kernel_size=1)
        )
        self.num_features_main = num_features_main

    def forward(self, x_main, spatial_text_features_base):
        # x_main: (B, num_features_main, H_main, W_main) - основные признаки
        # spatial_text_features_base: (B, text_channels_in, 1, W_base_text) - выход CharacterTokenEncoder
        
        h_main, w_main = x_main.shape[2], x_main.shape[3]
        
        # Апсемплируем базовые текстовые признаки до размеров x_main
        text_features_aligned = F.interpolate(spatial_text_features_base, 
                                              size=(h_main, w_main), 
                                              mode='bilinear', align_corners=False)
        # text_features_aligned форма: (B, text_channels_in, H_main, W_main)

        gamma_beta_spatial = self.param_predictor(text_features_aligned)
        # gamma_beta_spatial форма: (B, num_features_main * 2, H_main, W_main)
        
        gamma = gamma_beta_spatial[:, :self.num_features_main, :, :]
        beta = gamma_beta_spatial[:, self.num_features_main:, :, :]
        
        return gamma * x_main + beta

# --------------- VAEEncoderWithSkips (без изменений, как в paste.txt) ---------------

class VAEEncoderWithSkips(nn.Module):
    def __init__(self, in_ch=4, z_ch=Z_CH,
                 skip_chans=[32, 64, 128], bottleneck_ch=256):
        super().__init__()
        # 3 уровня!
        self.e_conv1 = self._conv_block(in_ch, skip_chans[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.e_conv2 = self._conv_block(skip_chans[0], skip_chans[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.e_conv3 = self._conv_block(skip_chans[1], skip_chans[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck_conv = self._conv_block(skip_chans[2], bottleneck_ch)

        self.feature_map_h = PATCH_SHAPE[1] // 8    # так как 3 пула (2^3=8)
        self.feature_map_w = PATCH_SHAPE[0] // 8
        self.mu_head = nn.Conv2d(bottleneck_ch, z_ch, kernel_size=(self.feature_map_h, self.feature_map_w))
        self.logvar_head = nn.Conv2d(bottleneck_ch, z_ch, kernel_size=(self.feature_map_h, self.feature_map_w))

    def _conv_block(self, in_c, out_c, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1_out = self.e_conv1(x)       # [B, 32, H, W]
        p1_out = self.pool1(s1_out)
        s2_out = self.e_conv2(p1_out)  # [B, 64, H/2, W/2]
        p2_out = self.pool2(s2_out)
        s3_out = self.e_conv3(p2_out)  # [B, 128, H/4, W/4]
        p3_out = self.pool3(s3_out)
        bottleneck = self.bottleneck_conv(p3_out)   # [B, 256, H/8, W/8]
        mu = self.mu_head(bottleneck)
        logvar = self.logvar_head(bottleneck)
        # Важно!! Возвращай ([s1, s2, s3]) — skip'ы от shallow to deep
        return mu, logvar, [s1_out, s2_out, s3_out]
    
class GatedSkipConnection(nn.Module):
    def __init__(self, channels, alpha_init=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * alpha_init)
    def forward(self, skip_feat):
        return skip_feat * torch.sigmoid(self.alpha)
    
# --------------- НОВЫЙ VAEDecoderWithSpatialFiLM ---------------

class VAEDecoderWithSpatialFiLM(nn.Module):
    def __init__(self, z_ch, text_channels_in, out_ch_image, patch_h, patch_w,
                 skip_chans=[32,64,128], bottleneck_ch=256):
        super().__init__()
        self.initial_h = patch_h // 8
        self.initial_w = patch_w // 8

        self.skip_gates = nn.ModuleList([
            GatedSkipConnection(skip_chans[2]),  # deepest (s3)
            GatedSkipConnection(skip_chans[1]),  # middle  (s2)
            GatedSkipConnection(skip_chans[0]),  # shallow (s1)
        ])

        self.bottleneck_proc = nn.Sequential( 
            nn.ConvTranspose2d(z_ch + text_channels_in, bottleneck_ch,
                               kernel_size=(self.initial_h, 1), stride=1, padding=0),
            nn.BatchNorm2d(bottleneck_ch), 
            nn.ReLU(inplace=True)
        )

        # 1st up:  bottleneck -> s3
        self.up_tconv1 = nn.ConvTranspose2d(bottleneck_ch, skip_chans[2], kernel_size=2, stride=2)
        after_cat1 = skip_chans[2] + skip_chans[2]
        self.spatial_film1 = SpatialFiLMLayer(text_channels_in, after_cat1)
        self.conv_block1 = self._conv_block(after_cat1, skip_chans[2])

        # 2nd up:  s3 -> s2
        self.up_tconv2 = nn.ConvTranspose2d(skip_chans[2], skip_chans[1], kernel_size=2, stride=2)
        after_cat2 = skip_chans[1] + skip_chans[1]
        self.spatial_film2 = SpatialFiLMLayer(text_channels_in, after_cat2)
        self.conv_block2 = self._conv_block(after_cat2, skip_chans[1])

        # 3rd up:  s2 -> s1
        self.up_tconv3 = nn.ConvTranspose2d(skip_chans[1], skip_chans[0], kernel_size=2, stride=2)
        after_cat3 = skip_chans[0] + skip_chans[0]
        self.spatial_film3 = SpatialFiLMLayer(text_channels_in, after_cat3)
        self.conv_block3 = self._conv_block(after_cat3, skip_chans[0])

        self.final_image_conv = nn.Conv2d(skip_chans[0], out_ch_image, kernel_size=1)
        self.output_activation_fn = nn.Sigmoid()

    def _conv_block(self, in_c, out_c, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, z_latents, spatial_text_features_base, skips_list):
        z_expanded_spatial = z_latents.expand(-1, -1, 1, self.initial_w)
        combined_bottleneck_input = torch.cat([z_expanded_spatial, spatial_text_features_base], dim=1)
        x_current = self.bottleneck_proc(combined_bottleneck_input)  # (B, bottleneck_ch, H/8, W/8)

        # 1st skip (deepest): s3
        x_upsampled = self.up_tconv1(x_current)
        gated_skip3 = self.skip_gates[0](skips_list[2])
        x_concat = torch.cat([x_upsampled, gated_skip3], dim=1)
        x_modulated = self.spatial_film1(x_concat, spatial_text_features_base)
        x_current = self.conv_block1(x_modulated)

        # 2nd skip: s2
        x_upsampled = self.up_tconv2(x_current)
        gated_skip2 = self.skip_gates[1](skips_list[1])
        x_concat = torch.cat([x_upsampled, gated_skip2], dim=1)
        x_modulated = self.spatial_film2(x_concat, spatial_text_features_base)
        x_current = self.conv_block2(x_modulated)

        # 3rd skip (shallowest): s1
        x_upsampled = self.up_tconv3(x_current)
        gated_skip1 = self.skip_gates[2](skips_list[0])
        x_concat = torch.cat([x_upsampled, gated_skip1], dim=1)
        x_modulated = self.spatial_film3(x_concat, spatial_text_features_base)
        x_current = self.conv_block3(x_modulated)

        output_img = self.final_image_conv(x_current)
        output_img = self.output_activation_fn(output_img)
        return output_img
    
# --------------- НОВАЯ ОСНОВНАЯ МОДЕЛЬ ---------------
class VAEGAN_UNet_SpatialFiLM(nn.Module):
    def __init__(self, in_ch_style=4, z_ch_style=Z_CH, out_ch_img=3,
                 alphabet_str_text=ALPHABET_STR, char_emb_dim_text=CHAR_EMB_DIM,
                 char_rnn_hidden_dim_text=CHAR_RNN_HIDDEN_DIM, char_rnn_layers_text=CHAR_RNN_LAYERS):
        super().__init__()
        
        # Ширина базовой карты текстовых признаков (W_base_text)
        self.text_feature_base_width = PATCH_SHAPE[0] // 16 

        self.char_text_encoder_module = CharacterTokenEncoder(
            alphabet_str=alphabet_str_text,
            emb_dim=char_emb_dim_text,
            rnn_hidden_dim=char_rnn_hidden_dim_text,
            rnn_layers=char_rnn_layers_text,
            target_feature_width=self.text_feature_base_width,
            target_feature_height=4
        )
        
        text_channels_for_film = self.char_text_encoder_module.rnn_output_dim

        self.style_vae_encoder_module = VAEEncoderWithSkips(in_ch=in_ch_style, z_ch=z_ch_style)
        self.image_vae_decoder_module = VAEDecoderWithSpatialFiLM(
            z_ch=z_ch_style,
            text_channels_in=text_channels_for_film,
            out_ch_image=out_ch_img,
            patch_h=PATCH_SHAPE[1],
            patch_w=PATCH_SHAPE[0]
        )
        print(f"VAEGAN_UNet_SpatialFiLM initialized. Base text feature map channels: {text_channels_for_film}, width: {self.text_feature_base_width}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image_for_style_in, mask_for_style_in, texts_batch_list_in):
        style_encoder_input = torch.cat([image_for_style_in, mask_for_style_in], dim=1)
        mu_style, logvar_style, skips_from_enc = self.style_vae_encoder_module(style_encoder_input)
        z_style = self.reparameterize(mu_style, logvar_style)
        
        spatial_text_features_base = self.char_text_encoder_module(texts_batch_list_in)
        # spatial_text_features_base форма: (B, rnn_output_dim, 1, text_feature_base_width)
        
        generated_image = self.image_vae_decoder_module(z_style, spatial_text_features_base, skips_from_enc)
        
        return generated_image, mu_style, logvar_style

# --------------- Discriminator (без изменений, как в paste.txt) ---------------
class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        from torch.nn.utils import spectral_norm
        self.body = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(512, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    def forward(self, x): 
        return self.body(x)


# --------------- Датасет и Загрузчик ---------------
class MarkDatasetAnyBBox(Dataset):
    def __init__(self, json_dir, ru_image_dir, en_image_dir, mask_dir, out_shape=(448, 64)):
        self.json_dir = json_dir
        self.ru_image_dir = ru_image_dir
        self.en_image_dir = en_image_dir
        self.mask_dir = mask_dir
        self.out_shape = out_shape # W, H
        self.samples = []
        print(f"Инициализация датасета MarkDatasetAnyBBox с out_shape (Ш, В): {self.out_shape}")

        annotation_files_list = sorted(os.listdir(json_dir)) # Сортировка для детерминизма
        for fname in tqdm(annotation_files_list, desc="Чтение аннотаций", unit="файл"):
            if not fname.lower().endswith(".json"):
                continue
            
            img_name_base_part = Path(fname).stem # img_name_base из вашего кода

            ru_path_found_for_item, en_path_found_for_item, mask_path_found_for_item = None, None, None
            found_paths_for_item = False # found_paths из вашего кода
            for ext_item in [".jpg", ".png", ".jpeg", ".webp"]:
                ru_filename_item = f"{img_name_base_part}_ru{ext_item}"
                en_filename_item = f"{img_name_base_part}_en{ext_item}"
                mask_filename_item = f"{img_name_base_part}_ru.png"

                ru_path_item = os.path.join(ru_image_dir, ru_filename_item)
                en_path_item = os.path.join(en_image_dir, en_filename_item)
                mask_path_item = os.path.join(mask_dir, mask_filename_item)

                if os.path.exists(ru_path_item):
                    ru_path_found_for_item = ru_path_item
                    if os.path.exists(en_path_item): en_path_found_for_item = en_path_item
                    if os.path.exists(mask_path_item): mask_path_found_for_item = mask_path_item
                    found_paths_for_item = True
                    break 
            
            if not found_paths_for_item:
                continue

            try:
                with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as f_json:
                    annotations_list_in_file = json.load(f_json) # annots из вашего кода
            except Exception as e_json:
                # print(f"Предупреждение: Ошибка чтения JSON {fname}: {e_json}") # Закомментировано для чистоты лога
                continue
            
            for item_data_json in annotations_list_in_file: # item из вашего кода
                if not isinstance(item_data_json, dict):
                    continue
                
                bbox_ru_data_json = item_data_json.get("bbox_ru")
                bbox_en_data_json = item_data_json.get("bbox_en")
                text_data_json = item_data_json.get("text")

                is_bbox_ru_valid_check = (isinstance(bbox_ru_data_json, list) and 
                                          len(bbox_ru_data_json) == 4 and 
                                          all(isinstance(p_coord, list) and len(p_coord) == 2 for p_coord in bbox_ru_data_json))
                if not is_bbox_ru_valid_check:
                    continue
                if not isinstance(text_data_json, str):
                    continue
                
                is_bbox_en_valid_check = (isinstance(bbox_en_data_json, list) and 
                                          len(bbox_en_data_json) == 4 and 
                                          all(isinstance(p_coord_en, list) and len(p_coord_en) == 2 for p_coord_en in bbox_en_data_json))
                if bbox_en_data_json and not is_bbox_en_valid_check:
                    bbox_en_data_json = None # Считаем отсутствующим, если формат неверный

                self.samples.append({
                    "ru_image_path": ru_path_found_for_item, 
                    "en_image_path": en_path_found_for_item, 
                    "mask_path": mask_path_found_for_item, 
                    "bbox_ru": bbox_ru_data_json, 
                    "bbox_en": bbox_en_data_json, 
                    "text": text_data_json
                })
        print(f"Инициализация датасета завершена. Найдено {len(self.samples)} сэмплов.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_item_meta = self.samples[idx] # item из вашего кода
        try:
            image_ru_pil_item = Image.open(sample_item_meta['ru_image_path']).convert("RGB") # image_ru из вашего кода
            
            image_en_pil_item = Image.new("RGB", image_ru_pil_item.size, (0,0,0)) # image_en из вашего кода
            if sample_item_meta['en_image_path'] and os.path.exists(sample_item_meta['en_image_path']):
                 image_en_pil_item = Image.open(sample_item_meta['en_image_path']).convert("RGB")
                           
            mask_pil_item = Image.new("L", image_ru_pil_item.size, 0) # mask из вашего кода
            if sample_item_meta['mask_path'] and os.path.exists(sample_item_meta['mask_path']):
                mask_pil_item = Image.open(sample_item_meta['mask_path']).convert("L")
                       
            bbox_ru_coords_item = sample_item_meta['bbox_ru'] # bbox_ru из вашего кода
            bbox_en_coords_item = sample_item_meta['bbox_en'] # bbox_en из вашего кода
            text_string_item = sample_item_meta['text']   # text из вашего кода

            ru_patch_pil = perspective_crop(image_ru_pil_item, bbox_ru_coords_item, self.out_shape)
            
            en_patch_pil = Image.new("RGB", self.out_shape, (0,0,0))
            if bbox_en_coords_item: # Если есть bbox_en, вырезаем en_patch
                en_patch_pil = perspective_crop(image_en_pil_item, bbox_en_coords_item, self.out_shape)
            
            mask_patch_pil = perspective_crop(mask_pil_item, bbox_ru_coords_item, self.out_shape)

            transform_img_to_tensor = T.ToTensor() # to_tensor из вашего кода
            ru_patch_tensor = transform_img_to_tensor(ru_patch_pil)
            en_patch_tensor = transform_img_to_tensor(en_patch_pil)
            mask_patch_tensor = transform_img_to_tensor(mask_patch_pil)
            
            return ru_patch_tensor, en_patch_tensor, mask_patch_tensor, text_string_item
        except Exception as e_getitem:
            dummy_rgb_tensor_val = torch.zeros((3, self.out_shape[1], self.out_shape[0])) # dummy_tensor
            dummy_mask_tensor_val = torch.zeros((1, self.out_shape[1], self.out_shape[0])) # dummy_mask
            return dummy_rgb_tensor_val, dummy_rgb_tensor_val, dummy_mask_tensor_val, ""

def perspective_crop(image_pil, bbox, out_shape): 
    if image_pil is None: return Image.new('RGB', out_shape, 0)
    img_np = np.array(image_pil)
    try:
        pts_src_np = np.array(bbox, dtype=np.float32).reshape(4, 2)
    except Exception:
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))
    if pts_src_np.shape != (4, 2):
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

    width_out, height_out = out_shape
    pts_dst_np = np.array([[0, 0], [width_out - 1, 0], [width_out - 1, height_out - 1], [0, height_out - 1]], dtype=np.float32)
    try:
        M_transform = cv2.getPerspectiveTransform(pts_src_np, pts_dst_np)
        patch_np_transformed = cv2.warpPerspective(img_np, M_transform, (width_out, height_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if img_np.ndim == 3 and patch_np_transformed.ndim == 2: patch_np_transformed = cv2.cvtColor(patch_np_transformed, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 2 and patch_np_transformed.ndim == 3: patch_np_transformed = cv2.cvtColor(patch_np_transformed, cv2.COLOR_RGB2GRAY)
        if patch_np_transformed.dtype != np.uint8: patch_np_transformed = np.clip(patch_np_transformed, 0, 255).astype(np.uint8)
        return Image.fromarray(patch_np_transformed)
    except Exception:
         return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

def safe_collate(batch_list_input): 
    filtered_batch_list = list(filter(lambda x_item: x_item is not None and x_item[0] is not None, batch_list_input))
    if not filtered_batch_list:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(filtered_batch_list)
    except Exception as e_collate:
        return None

# --------------- Функции Потерь (из paste-2.txt, отформатировано) ---------------
@torch.no_grad()
def get_vgg_feat(device_input): 
    vgg_weights = VGG16_Weights.IMAGENET1K_V1 
    vgg_model_instance = vgg16(weights=vgg_weights).features[:16].to(device_input).eval() # vgg из вашего кода
    normalize_transform_instance = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize из вашего кода
    return vgg_model_instance, normalize_transform_instance

def perceptual_loss(fake_img_input, real_img_input, vgg_model, vgg_normalize_transform): # fake, real, vgg, vgg_normalize
    fake_normalized_img = vgg_normalize_transform(fake_img_input) # fake_norm
    real_normalized_img = vgg_normalize_transform(real_img_input) # real_norm
    return F.l1_loss(vgg_model(fake_normalized_img), vgg_model(real_normalized_img))

def hinge_loss(predictions_input, target_val_input): 
    if target_val_input == 1:
        return F.relu(1.0 - predictions_input).mean()
    elif target_val_input == 0:
        return F.relu(1.0 + predictions_input).mean()
    else:
        return -predictions_input.mean()


# --------------- Циклы Обучения и Валидации ---------------
@torch.no_grad()
def val_loop(val_loader_input,
             model_generator,         # Генератор (переименован для ясности)
             model_discriminator,     # Дискриминатор
             criterion_recon_loss_fn, # Функция L1 потерь
             criterion_gan_loss_fn,   # Функция Hinge потерь
             vgg_model_instance, vgg_normalizer_instance, # Для перцептивной потери
             epoch_num_current, device_input, current_kl_weight_for_val=TARGET_KL_WEIGHT, show_patches_count=16):
    """
    Цикл валидации: вычисляет ВСЕ потери (как в train_loop) и логирует изображения в wandb.
    Без autocast.
    """
    model_generator.eval()
    model_discriminator.eval() # Переводим дискриминатор тоже в eval

    # Аккумуляторы для всех потерь
    total_val_recon_loss_accum = 0.0
    total_val_kl_loss_accum = 0.0
    total_val_gan_g_loss_accum = 0.0
    total_val_perc_loss_accum = 0.0
    total_val_disc_loss_accum = 0.0
    
    total_val_generator_loss_accum = 0.0 # Общая потеря генератора на валидации
    shown_patches = 0
    total_samples_processed_val = 0
    log_images_list_wandb = []

    progress_bar_val = tqdm(val_loader_input, desc=f"Эпоха {epoch_num_current} Валидация", unit="batch",
                            dynamic_ncols=True, leave=False, ascii=True)

    for batch_data_item_val in progress_bar_val:
        if batch_data_item_val is None:
            continue
        
        ru_patch_val, en_patch_val, mask_patch_val, text_en_val_list = batch_data_item_val
        ru_patch_val = ru_patch_val.to(device_input)
        en_patch_val = en_patch_val.to(device_input)
        mask_patch_val = mask_patch_val.to(device_input)
        current_batch_size_val = en_patch_val.size(0)

        # Forward pass генератора (без autocast)
        try:
            fake_patch_en_val, mu_val, logvar_val = model_generator(ru_patch_val, mask_patch_val, text_en_val_list)
        except Exception as e_fwd_g_val:
            print(f"\nОшибка инференса генератора на валидации: {e_fwd_g_val}")
            continue

        # --- Вычисление потерь Генератора (как в train_loop, но без backward() и step()) ---
        # 1. Потери Дискриминатора для оценки GAN Loss Генератора
        # Пропускаем fake_patch_en_val через дискриминатор (без .detach() здесь, т.к. это для оценки G)
        fake_preds_for_G_val = model_discriminator(fake_patch_en_val)
        
        # 2. Recon Loss
        recon_loss_val = criterion_recon_loss_fn(fake_patch_en_val, en_patch_val)
        
        # 3. KL Loss
        kl_loss_val_items = -0.5 * torch.mean(1 + logvar_val - mu_val.pow(2) - logvar_val.exp(), dim=[1, 2, 3])
        kl_loss_val = torch.mean(kl_loss_val_items)
        
        # 4. GAN Loss для Генератора
        gan_loss_g_val = criterion_gan_loss_fn(fake_preds_for_G_val, None) # Target None для генератора
        
        # 5. Perceptual Loss
        perc_loss_val = perceptual_loss(fake_patch_en_val, en_patch_val, vgg_model_instance, vgg_normalizer_instance)
        
        # Общая потеря Генератора (для логирования, не для оптимизации здесь)
        # Используем те же веса, что и в train_loop (предполагается, что они доступны глобально или переданы)
        # Глобальные веса: RECON_WEIGHT, KL_WEIGHT, GAN_WEIGHT, PERC_WEIGHT
        loss_G_val_total = (RECON_WEIGHT * recon_loss_val +
                            current_kl_weight_for_val * kl_loss_val +
                            GAN_WEIGHT * gan_loss_g_val +
                            PERC_WEIGHT * perc_loss_val)

        # --- Вычисление потерь Дискриминатора (как в train_loop, но без backward() и step()) ---
        real_preds_val_d = model_discriminator(en_patch_val)
        loss_D_real_val = criterion_gan_loss_fn(real_preds_val_d, 1)
        # Важно: используем .detach() для fake_patch_en_val при вычислении потерь D,
        # так как здесь мы оцениваем D, а не обучаем G через D.
        fake_preds_detached_val_d = model_discriminator(fake_patch_en_val.detach())
        loss_D_fake_val = criterion_gan_loss_fn(fake_preds_detached_val_d, 0)
        loss_D_val_total = (loss_D_real_val + loss_D_fake_val) * 0.5
        
        # Аккумуляция всех потерь
        total_val_generator_loss_accum += loss_G_val_total.item() * current_batch_size_val
        total_val_disc_loss_accum += loss_D_val_total.item() * current_batch_size_val
        total_val_recon_loss_accum += recon_loss_val.item() * current_batch_size_val
        total_val_kl_loss_accum += kl_loss_val.item() * current_batch_size_val
        total_val_gan_g_loss_accum += gan_loss_g_val.item() * current_batch_size_val
        total_val_perc_loss_accum += perc_loss_val.item() * current_batch_size_val
        
        total_samples_processed_val += current_batch_size_val

        
        if shown_patches < show_patches_count:
                ru_patch_cpu = ru_patch_val[:show_patches_count-shown_patches].cpu()
                en_patch_cpu = en_patch_val[:show_patches_count-shown_patches].cpu()
                fake_patch_en_cpu = fake_patch_en_val[:show_patches_count-shown_patches].cpu()
                texts_to_show = text_en_val_list[:show_patches_count-shown_patches]

                for i_img in range(en_patch_cpu.size(0)):
                    if shown_patches >= show_patches_count: break
                    real_p, fake_p, ru_p = en_patch_cpu[i_img], fake_patch_en_cpu[i_img], ru_patch_cpu[i_img]
                    text_label = texts_to_show[i_img][:50] + "..." if len(texts_to_show[i_img]) > 50 else texts_to_show[i_img]
                    caption = f"Epoch {epoch_num_current} | Target: '{text_label}' (Val)"
                    log_images_list_wandb.append(wandb.Image(ru_p, caption=f"{caption} | Input RU"))
                    log_images_list_wandb.append(wandb.Image(real_p, caption=f"{caption} | GT EN"))
                    log_images_list_wandb.append(wandb.Image(fake_p, caption=f"{caption} | FAKE EN"))
                    shown_patches += 1


    # Вычисление средних потерь
    avg_val_gen_loss = total_val_generator_loss_accum / total_samples_processed_val if total_samples_processed_val > 0 else float('inf')
    avg_val_disc_loss = total_val_disc_loss_accum / total_samples_processed_val if total_samples_processed_val > 0 else float('inf')
    avg_val_recon_loss = total_val_recon_loss_accum / total_samples_processed_val if total_samples_processed_val > 0 else float('inf')
    avg_val_kl_loss = total_val_kl_loss_accum / total_samples_processed_val if total_samples_processed_val > 0 else float('inf')
    avg_val_gan_g_loss = total_val_gan_g_loss_accum / total_samples_processed_val if total_samples_processed_val > 0 else float('inf')
    avg_val_perc_loss = total_val_perc_loss_accum / total_samples_processed_val if total_samples_processed_val > 0 else float('inf')

    print(f"\nValidation Эпоха {epoch_num_current}:")
    print(f"  Avg G Loss: {avg_val_gen_loss:.4f}, Avg D Loss: {avg_val_disc_loss:.4f}")
    print(f"  Avg Recon: {avg_val_recon_loss:.4f}, Avg KL: {avg_val_kl_loss:.4f}")
    print(f"  Avg GAN_G: {avg_val_gan_g_loss:.4f}, Avg Perc: {avg_val_perc_loss:.4f}")
    print(f"  KL(raw)={avg_val_kl_loss:.4f}, KL(weighted)={(current_kl_weight_for_val*avg_val_kl_loss):.4f}")

    # Логирование в wandb
    log_dict_val_wandb = {
        "val/generator_loss": avg_val_gen_loss,
        "val/discriminator_loss": avg_val_disc_loss,
        "val/recon_loss": avg_val_recon_loss,
        "val/kl_loss_raw":avg_val_kl_loss, 
        "val/kl_loss_weighted":(current_kl_weight_for_val*avg_val_kl_loss),
        "val/gan_loss_g": avg_val_gan_g_loss,
        "val/perceptual_loss": avg_val_perc_loss,
    }
    if log_images_list_wandb:
        log_dict_val_wandb["validation/examples"] = log_images_list_wandb
    
    if wandb.run: 
        wandb.log(log_dict_val_wandb, step=epoch_num_current)

    print("Завершен цикл валидации.")
    model_generator.train()
    model_discriminator.train() 

    return avg_val_recon_loss



# --- Адаптированный train_loop ---
def train_loop(train_loader_input, val_loader_input, 
               model_generator, model_discriminator, 
               opt_G_instance, opt_D_instance,
               scheduler_G_instance, scheduler_D_instance,
               criterion_recon_loss_fn, criterion_gan_loss_fn, 
               epoch_num_current, best_val_loss_so_far_input, 
               device_input, save_dir_path,
               val_loop_fn_input=None, current_kl_weight_for_epoch=TARGET_KL_WEIGHT): 
    
    model_generator.train()
    model_discriminator.train()
    
    total_loss_G_epoch_accum = 0.0
    total_loss_D_epoch_accum = 0.0
    total_recon_loss_epoch_accum = 0.0
    total_kl_loss_epoch_accum = 0.0
    total_gan_loss_epoch_accum = 0.0
    total_perc_loss_epoch_accum = 0.0
    
    vgg_model_for_train_loss, vgg_normalizer_for_train_loss = get_vgg_feat(device_input)

    progress_bar_train = tqdm(train_loader_input, desc=f"Эпоха {epoch_num_current+1} Трен. (KLw={current_kl_weight_for_epoch:.2e})", unit="batch",
                              dynamic_ncols=True, leave=False, ascii=True)

    for batch_data_item_train in progress_bar_train:
        if batch_data_item_train is None:
            continue
        
        ru_patch_train, en_patch_train, mask_patch_train, text_en_train_list = batch_data_item_train
        ru_patch_train = ru_patch_train.to(device_input)
        en_patch_train = en_patch_train.to(device_input)
        mask_patch_train = mask_patch_train.to(device_input)

        # Forward pass генератора (без autocast, как в вашем train_loop)
        try:
            fake_patch_en_train, mu_train, logvar_train = model_generator(ru_patch_train, mask_patch_train, text_en_train_list)
        except Exception as e_fwd_g_train:
            print(f"\nОшибка forward pass генератора: {e_fwd_g_train}")
            continue

        # --- Обучение Дискриминатора ---
        opt_D_instance.zero_grad()
        real_preds_train = model_discriminator(en_patch_train)
        loss_D_real_train = criterion_gan_loss_fn(real_preds_train, 1)
        fake_preds_detached_train = model_discriminator(fake_patch_en_train.detach())
        loss_D_fake_train = criterion_gan_loss_fn(fake_preds_detached_train, 0)
        loss_D_batch_train = (loss_D_real_train + loss_D_fake_train) * 0.5
        loss_D_batch_train.backward()
        opt_D_instance.step()

        # --- Обучение Генератора ---
        opt_G_instance.zero_grad()
        fake_preds_for_G_train = model_discriminator(fake_patch_en_train) # Не detach()
        recon_loss_train = criterion_recon_loss_fn(fake_patch_en_train, en_patch_train)
        
        kl_loss_train_items = -0.5 * torch.mean(1 + logvar_train - mu_train.pow(2) - logvar_train.exp(), dim=[1, 2, 3])
        kl_loss_train = torch.mean(kl_loss_train_items)
        
        gan_loss_g_train = criterion_gan_loss_fn(fake_preds_for_G_train, None)
        perc_loss_train = perceptual_loss(fake_patch_en_train, en_patch_train, 
                                          vgg_model_for_train_loss, vgg_normalizer_for_train_loss)
        
        loss_G_total_batch = (RECON_WEIGHT * recon_loss_train +
                              current_kl_weight_for_epoch * kl_loss_train +
                              GAN_WEIGHT * gan_loss_g_train +
                              PERC_WEIGHT * perc_loss_train)
        loss_G_total_batch.backward()
        clip_grad_norm_(model_generator.parameters(), max_norm=GRAD_CLIP_NORM)
        opt_G_instance.step()

        # Аккумуляция потерь
        total_loss_G_epoch_accum += loss_G_total_batch.item()
        total_loss_D_epoch_accum += loss_D_batch_train.item()
        total_recon_loss_epoch_accum += recon_loss_train.item()
        total_kl_loss_epoch_accum += kl_loss_train.item()
        total_gan_loss_epoch_accum += gan_loss_g_train.item()
        total_perc_loss_epoch_accum += perc_loss_train.item()

    # Вычисление средних потерь по эпохе обучения
    num_batches_train_epoch = len(train_loader_input)
    avg_G_train_epoch = total_loss_G_epoch_accum / max(1, num_batches_train_epoch)
    avg_D_train_epoch = total_loss_D_epoch_accum / max(1, num_batches_train_epoch)
    avg_recon_train_epoch = total_recon_loss_epoch_accum / max(1, num_batches_train_epoch)
    avg_kl_train_epoch = total_kl_loss_epoch_accum / max(1, num_batches_train_epoch)
    avg_gan_train_epoch = total_gan_loss_epoch_accum / max(1, num_batches_train_epoch)
    avg_perc_train_epoch = total_perc_loss_epoch_accum / max(1, num_batches_train_epoch)

    kl_loss_weighted_train = current_kl_weight_for_epoch * avg_kl_train_epoch
    
    print(f"\nЭпоха {epoch_num_current+1} Средние Потери (Train): G={avg_G_train_epoch:.4f}, D={avg_D_train_epoch:.4f}, "
          f"Recon={avg_recon_train_epoch:.4f}, KL(raw)={avg_kl_train_epoch:.4f}, KL(w)={kl_loss_weighted_train:.4f}, GAN_G={avg_gan_train_epoch:.4f}, Perc={avg_perc_train_epoch:.4f}")

    if wandb.run:
        wandb.log({
            "epoch": epoch_num_current + 1, # Логируем актуальную эпоху
            "train/generator_loss": avg_G_train_epoch,
            "train/discriminator_loss": avg_D_train_epoch,
            "train/recon_loss": avg_recon_train_epoch,
            "train/kl_loss": kl_loss_weighted_train,
            "train/gan_loss_g": avg_gan_train_epoch,
            "train/perceptual_loss": avg_perc_train_epoch,
            "train_params/current_kl_weight": current_kl_weight_for_epoch,
            "learning_rate/generator": opt_G_instance.param_groups[0]['lr'],
            "learning_rate/discriminator": opt_D_instance.param_groups[0]['lr']
        }, step=epoch_num_current + 1) # step для wandb

    current_avg_val_recon_loss_epoch = float('inf')
    if val_loop_fn_input is not None and val_loader_input is not None:
        # VGG модель для валидации (можно создать один раз вне цикла эпох, если не меняется)
        vgg_model_for_val_loss, vgg_normalizer_for_val_loss = get_vgg_feat(device_input)
        
        current_avg_val_recon_loss_epoch = val_loop_fn_input(
            val_loader_input, 
            model_generator, 
            model_discriminator, # Передаем дискриминатор
            criterion_recon_loss_fn, 
            criterion_gan_loss_fn, # Передаем GAN loss
            vgg_model_for_val_loss, vgg_normalizer_for_val_loss, # Передаем VGG компоненты
            epoch_num_current + 1, # Эпоха для валидации (следующая после текущей обучающей)
            device_input,
            current_kl_weight_for_val=current_kl_weight_for_epoch
        )

    if val_loader_input is not None and current_avg_val_recon_loss_epoch != float('inf'):
        scheduler_G_instance.step(current_avg_val_recon_loss_epoch)
        scheduler_D_instance.step(current_avg_val_recon_loss_epoch)

    # Сохранение чекпоинтов (логика из вашего train_loop)
    os.makedirs(save_dir_path, exist_ok=True)
    checkpoint_to_save_dict = {
        'epoch': epoch_num_current,
        'model_state_dict': model_generator.state_dict(),
        'disc_state_dict': model_discriminator.state_dict(),
        'opt_G_state_dict': opt_G_instance.state_dict(),
        'opt_D_state_dict': opt_D_instance.state_dict(),
        'scheduler_G_state_dict': scheduler_G_instance.state_dict(),
        'scheduler_D_state_dict': scheduler_D_instance.state_dict(),
        'best_val_loss': best_val_loss_so_far_input
    }
    torch.save(checkpoint_to_save_dict, os.path.join(save_dir_path, 'last_checkpoint.pth'))

    if current_avg_val_recon_loss_epoch < best_val_loss_so_far_input:
        best_val_loss_so_far_input = current_avg_val_recon_loss_epoch
        checkpoint_to_save_dict['best_val_loss'] = best_val_loss_so_far_input # Обновляем
        
        best_model_file_path = os.path.join(save_dir_path, 'best_model.pth')
        torch.save(checkpoint_to_save_dict, best_model_file_path)
        print(f"Сохранение лучшей модели с Validation Recon Loss: {current_avg_val_recon_loss_epoch:.4f} "
              f"(best_val_loss_so_far теперь: {best_val_loss_so_far_input:.4f})")
                
    elif current_avg_val_recon_loss_epoch != float('inf'):
        print(f"Текущая метрика Val Recon ({current_avg_val_recon_loss_epoch:.4f}) не лучше лучшей ({best_val_loss_so_far_input:.4f})")

    return best_val_loss_so_far_input


# --- Адаптированный main ---
def main():
    global DEVICE, ALPHABET_STR, VOCAB_SIZE, CHAR_EMB_DIM, CHAR_RNN_HIDDEN_DIM, CHAR_RNN_LAYERS, KL_ANNEAL_EPOCHS, START_KL_WEIGHT
    global TARGET_KL_WEIGHT

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate_g": LR_G,
            "learning_rate_d": LR_D, "z_dim": Z_CH,
            # "text_dim": TEXT_CH, # TEXT_CH больше не используется напрямую
            "patch_width": PATCH_SHAPE[0], "patch_height": PATCH_SHAPE[1],
            "target_kl_weight":TARGET_KL_WEIGHT, # Изменено
            "kl_anneal_epochs": KL_ANNEAL_EPOCHS, # Новый параметр
            "start_kl_weight": START_KL_WEIGHT, "gan_weight": GAN_WEIGHT,
            "recon_weight": RECON_WEIGHT, "perc_weight": PERC_WEIGHT,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "alphabet_str": ALPHABET_STR,
            "char_emb_dim": CHAR_EMB_DIM,
            "char_rnn_hidden": CHAR_RNN_HIDDEN_DIM, 
            "char_rnn_layers": CHAR_RNN_LAYERS,
            "initial_device": DEVICE, # Запоминаем исходный DEVICE
            "scheduler_mode": SCHEDULER_MODE, "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE, "scheduler_threshold": SCHEDULER_THRESHOLD,
            "scheduler_min_lr": SCHEDULER_MIN_LR
        }
    )
    config = wandb.config # Используем config из wandb
    DEVICE = config.initial_device # Восстанавливаем DEVICE из config
    # Обновляем глобальные переменные из config, если они там могут быть изменены
    ALPHABET_STR = config.alphabet_str
    VOCAB_SIZE = len(ALPHABET_STR) + 1 # Обновляем VOCAB_SIZE
    CHAR_EMB_DIM = config.char_emb_dim
    CHAR_RNN_HIDDEN_DIM = config.char_rnn_hidden
    CHAR_RNN_LAYERS = config.char_rnn_layers
    TARGET_KL_WEIGHT=config.target_kl_weight
    KL_ANNEAL_EPOCHS = config.kl_anneal_epochs
    START_KL_WEIGHT = config.start_kl_weight
    print(f"Начало обучения с KL Annealing (Start: {START_KL_WEIGHT}, Target: {TARGET_KL_WEIGHT}, Epochs: {KL_ANNEAL_EPOCHS})...")

    print(f"Используемое устройство: {DEVICE}")
    print(f"Алфавит (длина {VOCAB_SIZE-1}): '{ALPHABET_STR}'")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    print("Создание датасета...")
    dataset_json_dir_path = '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/all_annotations'
    dataset_ru_img_dir_path = '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/aug_ru'
    dataset_en_img_dir_path = '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/aug_en'
    dataset_mask_dir_path = '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/masks_from_ru_bbox'
    
    dataset_main_instance = MarkDatasetAnyBBox( 
        json_dir=dataset_json_dir_path, 
        ru_image_dir=dataset_ru_img_dir_path,
        en_image_dir=dataset_en_img_dir_path, 
        mask_dir=dataset_mask_dir_path,
        out_shape=(config.patch_width, config.patch_height)
    )

    print("Разделение данных на Train/Val...")
    all_img_paths_list = [sample['ru_image_path'] for sample in dataset_main_instance.samples if sample.get('ru_image_path')] # all_img_paths
    unique_imgs_list = sorted(list(set(all_img_paths_list))) # unique_imgs
    if not unique_imgs_list:
        print("Ошибка: Нет валидных путей к изображениям в датасете."); wandb.finish(); return
    
    validation_split_ratio = config.get("validation_split_size", 0.1) # Используем из config или 0.1
    try:
        train_imgs_paths_list, val_imgs_paths_list = train_test_split(unique_imgs_list, test_size=validation_split_ratio, random_state=42) # train_imgs, val_imgs
    except ValueError as e_split:
        print(f"Ошибка разделения данных: {e_split}. Используем все данные для обучения.");
        train_imgs_paths_list = unique_imgs_list
        val_imgs_paths_list = []
        
    train_indices_list = [i for i, s_item in enumerate(dataset_main_instance.samples) if s_item.get('ru_image_path') in train_imgs_paths_list] # train_idx
    val_indices_list = [i for i, s_item in enumerate(dataset_main_instance.samples) if s_item.get('ru_image_path') in val_imgs_paths_list] # val_idx
    print(f"Train/Val сэмплы: {len(train_indices_list)} / {len(val_indices_list)}")

    print("Создание DataLoader'ов...")
    num_workers_dataloader = config.get("num_dataloader_workers", 2) # num_workers из вашего кода
    
    train_loader_instance = DataLoader( # train_loader
        Subset(dataset_main_instance, train_indices_list), batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers_dataloader, pin_memory=True, drop_last=True, collate_fn=safe_collate
    )
    val_loader_instance = None # val_loader
    if val_indices_list:
        val_loader_instance = DataLoader(
            Subset(dataset_main_instance, val_indices_list), batch_size=config.batch_size, shuffle=False,
            num_workers=num_workers_dataloader, pin_memory=True, collate_fn=safe_collate
        )
    else:
        print("Нет валидационных данных. Планировщик LR и сохранение лучшей модели по val_loss не будут работать.")

    
    print("Инициализация моделей (VAEGAN_UNet_CharEmb)...")
    # Инициализация новой модели
    model_generator = VAEGAN_UNet_SpatialFiLM(
        in_ch_style=4, 
        z_ch_style=config.z_dim, 
        out_ch_img=3,
        alphabet_str_text=config.alphabet_str,
        char_emb_dim_text=config.char_emb_dim,
        char_rnn_hidden_dim_text=config.char_rnn_hidden,
        char_rnn_layers_text=config.char_rnn_layers
    ).to(DEVICE)
    # print(model_generator)
    
    model_discriminator = Discriminator(in_ch=3).to(DEVICE) 
    
    criterion_reconstruction_loss = nn.L1Loss().to(DEVICE) 
    criterion_gan_loss_fn = hinge_loss 

    optimizer_G_instance = optim.Adam(model_generator.parameters(), lr=config.learning_rate_g, betas=(0.5, 0.999)) # opt_G
    optimizer_D_instance = optim.Adam(model_discriminator.parameters(), lr=config.learning_rate_d, betas=(0.5, 0.999)) # opt_D

    scheduler_G_instance = torch.optim.lr_scheduler.ReduceLROnPlateau( # scheduler_G
        optimizer_G_instance, mode=config.scheduler_mode, factor=config.scheduler_factor,
        patience=config.scheduler_patience, threshold=config.scheduler_threshold,
        min_lr=config.scheduler_min_lr
    )
    scheduler_D_instance = torch.optim.lr_scheduler.ReduceLROnPlateau( # scheduler_D
        optimizer_D_instance, mode=config.scheduler_mode, factor=config.scheduler_factor,
        patience=config.scheduler_patience, threshold=config.scheduler_threshold,
        min_lr=config.scheduler_min_lr
    )

    start_epoch_count = 0 # start_epoch
    best_val_metric_overall = float('inf') # best_val
    
    checkpoints_directory = './checkpoints_vaegan_unet_char' # save_dir, новая папка для новой архитектуры
    os.makedirs(checkpoints_directory, exist_ok=True)
    last_checkpoint_path_full = os.path.join(checkpoints_directory, 'last_checkpoint.pth') # checkpoint_path

    
    if os.path.exists(last_checkpoint_path_full): 
        print(f"Попытка возобновления с чекпоинта: {last_checkpoint_path_full}")
        try:
            checkpoint_data = torch.load(last_checkpoint_path_full, map_location=DEVICE) # checkpoint
          
            model_generator.load_state_dict(checkpoint_data['model_state_dict'], strict=False) # strict=False для гибкости
            model_discriminator.load_state_dict(checkpoint_data['disc_state_dict'], strict=False)
            
            optimizer_G_instance.load_state_dict(checkpoint_data['opt_G_state_dict'])
            optimizer_D_instance.load_state_dict(checkpoint_data['opt_D_state_dict'])
            
            if 'scheduler_G_state_dict' in checkpoint_data:
                scheduler_G_instance.load_state_dict(checkpoint_data['scheduler_G_state_dict'])
            if 'scheduler_D_state_dict' in checkpoint_data:
                scheduler_D_instance.load_state_dict(checkpoint_data['scheduler_D_state_dict'])
                
            start_epoch_count = checkpoint_data['epoch'] + 1
            best_val_metric_overall = checkpoint_data.get('best_val_loss', float('inf'))
            print(f"Загружен чекпоинт эпохи {checkpoint_data['epoch']}. Старт с {start_epoch_count}.")
            print(f"Лучшая валидационная потеря из чекпоинта: {best_val_metric_overall:.4f}")

            for opt_state_item in optimizer_G_instance.state.values(): # state
                for k_key, v_val in opt_state_item.items(): # k, v
                    if isinstance(v_val, torch.Tensor): opt_state_item[k_key] = v_val.to(DEVICE)
            for opt_state_item_d in optimizer_D_instance.state.values():
                for k_key_d, v_val_d in opt_state_item_d.items():
                    if isinstance(v_val_d, torch.Tensor): opt_state_item_d[k_key_d] = v_val_d.to(DEVICE)
        except Exception as e_load_ckpt:
            print(f"Ошибка загрузки чекпоинта или несовместимость: {e_load_ckpt}. Старт с нуля.")
            start_epoch_count = 0; best_val_metric_overall = float('inf')
    else:
        print("Чекпоинт не найден. Обучение с нуля.")
        start_epoch_count = 0; best_val_metric_overall = float('inf')



    print(f"Начало обучения на {DEVICE} для {config.epochs} эпох...")
    try:
        for epoch_num_current in range(start_epoch_count, config.epochs): # epoch
            current_kl_for_epoch_main = TARGET_KL_WEIGHT
            if epoch_num_current < KL_ANNEAL_EPOCHS:
                current_kl_for_epoch_main = START_KL_WEIGHT + (TARGET_KL_WEIGHT - START_KL_WEIGHT) * (epoch_num_current / max(1, KL_ANNEAL_EPOCHS -1))

            best_val_metric_overall = train_loop(
                train_loader_instance, val_loader_instance,
                model_generator, model_discriminator, 
                optimizer_G_instance, optimizer_D_instance,
                scheduler_G_instance, scheduler_D_instance,
                criterion_reconstruction_loss, criterion_gan_loss_fn,
                epoch_num_current, best_val_metric_overall, # epoch, best_val
                device_input=DEVICE,
                save_dir_path=checkpoints_directory, # save_dir
                val_loop_fn_input=val_loop if val_loader_instance else None,
                current_kl_weight_for_epoch=current_kl_for_epoch_main 
            )

        print("Обучение завершено.")
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
    finally:
        if wandb.run and wandb.run.id:
             wandb.finish()
        print("Сессия wandb завершена.")

if __name__ == "__main__":
    main()
