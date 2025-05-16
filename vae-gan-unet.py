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


ALPHABET_STR = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
VOCAB_SIZE = len(ALPHABET_STR) + 1 # +1 для <PAD> токена (индекс 0)
CHAR_EMB_DIM = 128          # Размерность эмбеддинга одного символа
CHAR_RNN_HIDDEN_DIM = 256   # Размерность скрытого состояния RNN/GRU для текста (в одну сторону)
CHAR_RNN_LAYERS = 2         # Количество слоев RNN

# Параметры обучения
LR_G = 1e-4
LR_D = 1e-4
KL_WEIGHT = 0.001    
GAN_WEIGHT = 0.15    
PERC_WEIGHT = 0.1   
RECON_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0

# Параметры для ReduceLROnPlateau
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.95
SCHEDULER_PATIENCE = 15
SCHEDULER_THRESHOLD = 0.0001
SCHEDULER_MIN_LR = 1e-7

WANDB_PROJECT = "VAE-GAN"
WANDB_ENTITY = None 
WANDB_RUN_NAME = "unet_char_emb_no_dataparallel_v1" 
WANDB_SAVE_CODE = True

# --------------- Новая Архитектура Моделей ---------------

class CharacterTokenEncoder(nn.Module):
    def __init__(self, alphabet_str, emb_dim, rnn_hidden_dim, rnn_layers,
                 target_feature_width):
        super().__init__()
        self.alphabet = alphabet_str
        self.vocab_size = len(alphabet_str) + 1 # +1 для PAD токена (индекс 0)
        self.char_to_idx = {char: i + 1 for i, char in enumerate(alphabet_str)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(alphabet_str)} # Для возможной отладки
        self.pad_idx = 0

        self.embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=self.pad_idx)
        self.rnn = nn.GRU(
            emb_dim, rnn_hidden_dim, num_layers=rnn_layers,
            batch_first=True, bidirectional=True, dropout=0.1 if rnn_layers > 1 else 0
        )
        
        self.rnn_output_dim = rnn_hidden_dim * 2 # Из-за bidirectional
        self.target_feature_width = target_feature_width # W_feat

        # Адаптивный пулинг для приведения к target_feature_width
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_feature_width)

        print(f"CharacterTokenEncoder initialized:")
        print(f"  Vocab Size: {self.vocab_size}, Embedding Dim: {emb_dim}")
        print(f"  RNN: Hidden Dim (per direction): {rnn_hidden_dim}, Layers: {rnn_layers}, Bidirectional: True")
        print(f"  RNN Output Dim (total): {self.rnn_output_dim}")
        print(f"  Target Output Feature Width (spatial): {target_feature_width}")

    def tokens_to_indices(self, text_list, max_len_chars):
        batch_indices = []
        for text in text_list:
            indices = [self.char_to_idx.get(char, self.pad_idx) for char in text] # Используем pad_idx для OOV
            indices = indices[:max_len_chars] # Обрезаем, если длиннее
            # Паддинг до max_len_chars
            padded_indices = indices + [self.pad_idx] * (max_len_chars - len(indices))
            batch_indices.append(torch.tensor(padded_indices, dtype=torch.long))
        return torch.stack(batch_indices)

    def forward(self, texts_batch, max_len_chars_for_tokenization=60): # texts_batch: список строк
        # max_len_chars_for_tokenization - максимальная длина текста для токенизации
        
        indices_tensor = self.tokens_to_indices(texts_batch, max_len_chars_for_tokenization)
        indices_tensor = indices_tensor.to(self.embedding.weight.device) # (B, MaxLenChars)
        
        embedded_chars = self.embedding(indices_tensor)  # (B, MaxLenChars, EmbDim)
        
        rnn_outputs, _ = self.rnn(embedded_chars) # (B, MaxLenChars, RnnOutputDim)
        
        # (B, MaxLenChars, RnnOutputDim) -> (B, RnnOutputDim, MaxLenChars) для пулинга
        text_features_permuted = rnn_outputs.permute(0, 2, 1)
        
        # Приводим к целевой ширине (W_feat)
        # (B, RnnOutputDim, MaxLenChars) -> (B, RnnOutputDim, target_feature_width)
        text_features_pooled = self.adaptive_pool(text_features_permuted)
        
        # Добавляем "высоту" = 1, чтобы получить (B, RnnOutputDim, 1, target_feature_width)
        text_features_4d = text_features_pooled.unsqueeze(2)
        
        return text_features_4d


class VAEEncoderWithSkips(nn.Module):
    def __init__(self, in_ch=4, z_ch=Z_CH):
        super().__init__()
        self.e_conv1 = self._conv_block(in_ch, 64)          # Output: B, 64, H, W
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B, 64, H/2, W/2
        
        self.e_conv2 = self._conv_block(64, 128)            # Output: B, 128, H/2, W/2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B, 128, H/4, W/4
        
        self.e_conv3 = self._conv_block(128, 256)           # Output: B, 256, H/4, W/4
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B, 256, H/8, W/8
        
        self.e_conv4 = self._conv_block(256, 512)           # Output: B, 512, H/8, W/8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B, 512, H/16, W/16
        
        self.bottleneck_conv = self._conv_block(512, 1024)  # Output: B, 1024, H/16, W/16
        
        feature_map_h = PATCH_SHAPE[1] // 16
        feature_map_w = PATCH_SHAPE[0] // 16
        self.mu_head = nn.Conv2d(1024, z_ch, kernel_size=(feature_map_h, feature_map_w))
        self.logvar_head = nn.Conv2d(1024, z_ch, kernel_size=(feature_map_h, feature_map_w))
        print("VAEEncoderWithSkips initialized.")

    def _conv_block(self, in_c, out_c, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1_out = self.e_conv1(x)
        p1_out = self.pool1(s1_out)
        
        s2_out = self.e_conv2(p1_out)
        p2_out = self.pool2(s2_out)
        
        s3_out = self.e_conv3(p2_out)
        p3_out = self.pool3(s3_out)
        
        s4_out = self.e_conv4(p3_out)
        p4_out = self.pool4(s4_out)
        
        bottleneck_features = self.bottleneck_conv(p4_out)
        
        mu = self.mu_head(bottleneck_features)
        logvar = self.logvar_head(bottleneck_features)
        
        skips_for_decoder = [s1_out, s2_out, s3_out, s4_out] # От самого "мелкого" к "глубокому"
        return mu, logvar, skips_for_decoder


class VAEDecoderWithSkips(nn.Module):
    def __init__(self, z_ch=Z_CH, text_feat_channels=(CHAR_RNN_HIDDEN_DIM * 2), out_ch_image=3):
        super().__init__()
        
        self.initial_h = PATCH_SHAPE[1] // 16
        self.initial_w = PATCH_SHAPE[0] // 16

        # Каналы skip-соединений от энкодера (в порядке от глубокого к мелкому для удобства)
        # s4: 512 (H/8, W/8), s3: 256 (H/4, W/4), s2: 128 (H/2, W/2), s1: 64 (H, W)
        skip_channels_from_encoder = [512, 256, 128, 64] 
        
        # Начальный блок, объединяющий z и текстовые признаки
        # z: (B, z_ch, 1, 1), text_features: (B, text_feat_channels, 1, initial_w)
        # Цель: получить (B, 1024, initial_h, initial_w)
        self.bottleneck_upsample = nn.Sequential(
            nn.ConvTranspose2d(z_ch + text_feat_channels, 1024, 
                               kernel_size=(self.initial_h, self.initial_w), # Преобразует (1,W_feat) в (H_feat,W_feat)
                               stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Декодирующие блоки
        self.d_upconv1 = self._upconv_block(1024 + skip_channels_from_encoder[0], 512) # С s4
        self.d_upconv2 = self._upconv_block(512  + skip_channels_from_encoder[1], 256) # С s3
        self.d_upconv3 = self._upconv_block(256  + skip_channels_from_encoder[2], 128) # С s2
        self.d_upconv4 = self._upconv_block(128  + skip_channels_from_encoder[3], 64)  # С s1
        
        self.final_image_conv = nn.Conv2d(64, out_ch_image, kernel_size=1) # 1x1 свертка
        self.output_activation_fn = nn.Sigmoid()
        print("VAEDecoderWithSkips initialized.")

    def _upconv_block(self, in_c, out_c, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2), # Апсемплинг
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, z_latents, text_features_input, skips_list):
        # z_latents: (B, z_ch, 1, 1)
        # text_features_input: (B, text_feat_channels, 1, initial_w)
        # skips_list: [s1, s2, s3, s4] (s1 - самый "мелкий", s4 - самый "глубокий")

        # Расширяем z до (B, z_ch, 1, initial_w) для конкатенации с текстом
        z_expanded_spatial = z_latents.expand(-1, -1, 1, self.initial_w)
        
        combined_bottleneck_input = torch.cat([z_expanded_spatial, text_features_input], dim=1)
        
        # Начальный апсемплинг объединенных z и текстовых признаков
        d0_features = self.bottleneck_upsample(combined_bottleneck_input) # (B, 1024, initial_h, initial_w)
        
        # Апсемплинг с skip-connections
        # skips_list[3] это s4, skips_list[0] это s1
        d1_input = torch.cat([d0_features, skips_list[3]], dim=1) # s4 (самый глубокий)
        d1_features = self.d_upconv1(d1_input)
        
        d2_input = torch.cat([d1_features, skips_list[2]], dim=1) # s3
        d2_features = self.d_upconv2(d2_input)
        
        d3_input = torch.cat([d2_features, skips_list[1]], dim=1) # s2
        d3_features = self.d_upconv3(d3_input)
        
        d4_input = torch.cat([d3_features, skips_list[0]], dim=1) # s1 (самый мелкий)
        d4_features = self.d_upconv4(d4_input)
        
        output_img = self.final_image_conv(d4_features)
        output_img = self.output_activation_fn(output_img)
        
        return output_img


class VAEGAN_UNet_CharEmb(nn.Module):
    def __init__(self, in_ch_for_style_encoder=4, z_ch_for_style=Z_CH, out_ch_for_image=3,
                 alphabet_str_for_text=ALPHABET_STR, char_emb_dim_for_text=CHAR_EMB_DIM,
                 char_rnn_hidden_dim_for_text=CHAR_RNN_HIDDEN_DIM, char_rnn_layers_for_text=CHAR_RNN_LAYERS):
        super().__init__()
        
        self.text_feature_target_spatial_width = PATCH_SHAPE[0] // 16 

        self.char_text_encoder_module = CharacterTokenEncoder(
            alphabet_str=alphabet_str_for_text,
            emb_dim=char_emb_dim_for_text,
            rnn_hidden_dim=char_rnn_hidden_dim_for_text,
            rnn_layers=char_rnn_layers_for_text,
            target_feature_width=self.text_feature_target_spatial_width
        )
        
        text_encoder_output_feature_channels = self.char_text_encoder_module.rnn_output_dim

        self.style_vae_encoder_module = VAEEncoderWithSkips(in_ch=in_ch_for_style_encoder, z_ch=z_ch_for_style)
        self.image_vae_decoder_module = VAEDecoderWithSkips(
            z_ch=z_ch_for_style,
            text_feat_channels=text_encoder_output_feature_channels,
            out_ch_image=out_ch_for_image
        )
        print(f"VAEGAN_UNet_CharEmb initialized. Text encoder provides {text_encoder_output_feature_channels} channels to decoder.")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image_for_style_input, mask_for_style_input, texts_batch_list_input):
        style_encoder_input_cat = torch.cat([image_for_style_input, mask_for_style_input], dim=1)
        mu_style, logvar_style, skips_list_from_encoder = self.style_vae_encoder_module(style_encoder_input_cat)
        z_style_sample = self.reparameterize(mu_style, logvar_style)
        
        text_features_from_encoder = self.char_text_encoder_module(texts_batch_list_input)
        
        generated_output_image = self.image_vae_decoder_module(z_style_sample, text_features_from_encoder, skips_list_from_encoder)
        
        return generated_output_image, mu_style, logvar_style

class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
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
def val_loop(val_loader, model, criterion_recon, epoch, device, show_patches=16):
    """Цикл валидации: вычисляет потери и логирует изображения в wandb."""
    model.eval()
    shown_patches = 0
    print("Запуск цикла валидации...")
    log_images = []
    total_val_recon_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(val_loader, desc=f"Эпоха {epoch} Валидация", unit="batch", dynamic_ncols=True, leave=False, ascii=True)
    for batch_data in progress_bar:
        if batch_data is None: continue
        ru_patch, en_patch, mask_patch, text_en = batch_data
        ru_patch, en_patch, mask_patch = ru_patch.to(device), en_patch.to(device), mask_patch.to(device)
        batch_size = en_patch.size(0)

        try: fake_patch_en, mu, logvar = model(ru_patch, mask_patch, text_en)
        except Exception as e: print(f"\nОшибка инференса на валидации: {e}"); continue

        # Вычисление и аккумуляция потерь реконструкции
        recon_loss = criterion_recon(fake_patch_en, en_patch)
        total_val_recon_loss += recon_loss.item() * batch_size
        total_samples += batch_size

        # Логика для визуализации
        if shown_patches < show_patches:
            ru_patch_cpu = ru_patch[:show_patches-shown_patches].cpu()
            en_patch_cpu = en_patch[:show_patches-shown_patches].cpu()
            fake_patch_en_cpu = fake_patch_en[:show_patches-shown_patches].cpu()
            texts_to_show = text_en[:show_patches-shown_patches]

            for i in range(len(en_patch_cpu)):
                if shown_patches >= show_patches: break
                real_p, fake_p, ru_p = en_patch_cpu[i], fake_patch_en_cpu[i], ru_patch_cpu[i]
                text_label = texts_to_show[i][:50] + "..." if len(texts_to_show[i]) > 50 else texts_to_show[i]
                caption = f"Epoch {epoch} | Target: '{text_label}'"
                log_images.append(wandb.Image(ru_p, caption=f"{caption} | Input RU Patch"))
                log_images.append(wandb.Image(real_p, caption=f"{caption} | Ground Truth EN Patch"))
                log_images.append(wandb.Image(fake_p, caption=f"{caption} | Generated FAKE Patch"))
                shown_patches += 1

    # Вычисление средней потери
    avg_val_recon_loss = total_val_recon_loss / total_samples if total_samples > 0 else 0.0
    print(f"\nValidation Средняя Recon Loss: {avg_val_recon_loss:.4f}")

    # Логирование в wandb
    log_dict = { "val/recon_loss": avg_val_recon_loss }
    if log_images: log_dict["validation/examples"] = log_images
    wandb.log(log_dict, step=epoch) # Логируем с номером текущей эпохи

    print("Завершен цикл валидации.")
    model.train()
    return avg_val_recon_loss


# --- Адаптированный train_loop ---
def train_loop(train_loader, val_loader, model, disc, opt_G, opt_D,
               scheduler_G, scheduler_D, # <--- ДОБАВЛЕНО
               criterion_recon, criterion_gan, epoch, best_val_loss_so_far, device, save_dir, # best_val переименован в best_val_loss_so_far
               val_loop_fn=None): # Добавлены параметры для val_loop
    """Основной цикл обучения на одну эпоху."""
    model.train()
    disc.train()
    
    total_loss_G_epoch = 0.0
    total_loss_D_epoch = 0.0
    total_recon_loss_epoch = 0.0
    total_kl_loss_epoch = 0.0
    total_gan_loss_epoch = 0.0
    total_perc_loss_epoch = 0.0
    
    vgg_model_for_loss, vgg_normalizer_for_loss = get_vgg_feat(device)

    # progress_bar_train = tqdm(
    #     train_loader,
    #     desc=f"Эпоха {epoch+1} Обучение",
    #     unit="batch",
    #     mininterval=1.0,
    #     dynamic_ncols=True,
    #     leave=False,
    #     ascii=True
    # )

    for batch_data in tqdm(train_loader): 
        if batch_data is None:
            continue
        
        ru_patch, en_patch, mask_patch, text_en = batch_data
        ru_patch = ru_patch.to(device)
        en_patch = en_patch.to(device)
        mask_patch = mask_patch.to(device)

        try:
            # В вашем коде не было autocast здесь, оставляем так
            fake_patch_en, mu, logvar = model(ru_patch, mask_patch, text_en)
        except Exception as e:
            print(f"\nОшибка forward pass генератора: {e}")
            continue # Пропускаем этот батч

        # Обучение Дискриминатора
        opt_D.zero_grad()
        # В вашем коде не было autocast здесь
        real_preds = disc(en_patch)
        loss_D_real = criterion_gan(real_preds, 1)
        fake_preds_detached = disc(fake_patch_en.detach())
        loss_D_fake = criterion_gan(fake_preds_detached, 0)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # Обучение Генератора
        opt_G.zero_grad()
        # В вашем коде не было autocast здесь
        fake_preds = disc(fake_patch_en) # Не detach()
        recon_loss = criterion_recon(fake_patch_en, en_patch)
        # KL loss из вашего кода:
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl_loss = torch.mean(kl_loss)
        gan_loss = criterion_gan(fake_preds, None)
        perc_loss = perceptual_loss(fake_patch_en, en_patch, vgg_model_for_loss, vgg_normalizer_for_loss)
        
        loss_G = (RECON_WEIGHT * recon_loss +
                  KL_WEIGHT * kl_loss +
                  GAN_WEIGHT * gan_loss +
                  PERC_WEIGHT * perc_loss)
        loss_G.backward()
        clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        opt_G.step()

        total_loss_G_epoch += loss_G.item()
        total_loss_D_epoch += loss_D.item()
        total_recon_loss_epoch += recon_loss.item()
        total_kl_loss_epoch += kl_loss.item()
        total_gan_loss_epoch += gan_loss.item()
        total_perc_loss_epoch += perc_loss.item()

    num_batches_train = len(train_loader)
    avg_G_train = total_loss_G_epoch / max(1, num_batches_train)
    avg_D_train = total_loss_D_epoch / max(1, num_batches_train)
    avg_recon_train = total_recon_loss_epoch / max(1, num_batches_train)
    avg_kl_train = total_kl_loss_epoch / max(1, num_batches_train)
    avg_gan_train = total_gan_loss_epoch / max(1, num_batches_train)
    avg_perc_train = total_perc_loss_epoch / max(1, num_batches_train)
    
    print(f"\nЭпоха {epoch+1} Средние Потери (Train): G={avg_G_train:.4f}, D={avg_D_train:.4f}, "
          f"Recon={avg_recon_train:.4f}, KL={avg_kl_train:.4f}, GAN_G={avg_gan_train:.4f}, Perc={avg_perc_train:.4f}")

    if wandb.run:
        wandb.log({
            "epoch": epoch + 1,
            "train/generator_loss": avg_G_train,
            "train/discriminator_loss": avg_D_train,
            "train/recon_loss": avg_recon_train,
            "train/kl_loss": avg_kl_train,
            "train/gan_loss_g": avg_gan_train,
            "train/perceptual_loss": avg_perc_train,
            "learning_rate/generator": opt_G.param_groups[0]['lr'],
            "learning_rate/discriminator": opt_D.param_groups[0]['lr']
        }, step=epoch + 1)

    current_avg_val_loss = float('inf') # avg_val_loss из вашего кода
    if val_loop_fn is not None and val_loader is not None:
        current_avg_val_loss = val_loop_fn(
            val_loader, model, criterion_recon, epoch + 1, device, # epoch+1 для val_loop
        )

    # --- ДОБАВЛЕНО: Шаг планировщиков ---
    if val_loader is not None and current_avg_val_loss != float('inf'):
        scheduler_G.step(current_avg_val_loss)
        scheduler_D.step(current_avg_val_loss) # Можно использовать одну и ту же метрику
    # --- КОНЕЦ ДОБАВЛЕНИЯ ---

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_to_save = { # Имя переменной checkpoint из вашего кода
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(), # Сохраняем состояние планировщика
        'scheduler_D_state_dict': scheduler_D.state_dict(), # Сохраняем состояние планировщика
        'best_val_loss': best_val_loss_so_far # best_val из вашего кода
    }
    torch.save(checkpoint_to_save, os.path.join(save_dir, 'last_checkpoint.pth'))

    # best_val из вашего кода - это best_val_loss_so_far
    if current_avg_val_loss < best_val_loss_so_far:
        best_val_loss_so_far = current_avg_val_loss # Обновляем
        checkpoint_to_save['best_val_loss'] = best_val_loss_so_far # Обновляем в чекпоинте для best_model
        
        best_model_filepath = os.path.join(save_dir, 'best_model.pth') # Путь к файлу лучшей модели
        torch.save(checkpoint_to_save, best_model_filepath)
        print(f"Сохранение лучшей модели с Validation Recon Loss: {current_avg_val_loss:.4f} "
              f"(best_val_loss_so_far теперь: {best_val_loss_so_far:.4f})")
        
        
    elif current_avg_val_loss != float('inf'): # current_val_metric из вашего кода
        print(f"Текущая метрика Val Recon ({current_avg_val_loss:.4f}) не лучше лучшей ({best_val_loss_so_far:.4f})")

    return best_val_loss_so_far



# --- Адаптированный main ---
def main():
    global DEVICE, ALPHABET_STR, VOCAB_SIZE, CHAR_EMB_DIM, CHAR_RNN_HIDDEN_DIM, CHAR_RNN_LAYERS

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate_g": LR_G,
            "learning_rate_d": LR_D, "z_dim": Z_CH,
            # "text_dim": TEXT_CH, # TEXT_CH больше не используется напрямую
            "patch_width": PATCH_SHAPE[0], "patch_height": PATCH_SHAPE[1],
            "kl_weight": KL_WEIGHT, "gan_weight": GAN_WEIGHT,
            "recon_weight": RECON_WEIGHT, "perc_weight": PERC_WEIGHT,
            "grad_clip_norm": GRAD_CLIP_NORM,
            # "text_encoder_model": TRANSFORMER_MODEL_NAME, # Больше не используется
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
    model_generator = VAEGAN_UNet_CharEmb( 
        in_ch_for_style_encoder=4, 
        z_ch_for_style=config.z_dim, 
        out_ch_for_image=3,
        alphabet_str_for_text=config.alphabet_str,
        char_emb_dim_for_text=config.char_emb_dim,
        char_rnn_hidden_dim_for_text=config.char_rnn_hidden,
        char_rnn_layers_for_text=config.char_rnn_layers
    ).to(DEVICE)
    print(model_generator)
    
    model_discriminator = Discriminator(in_ch=3).to(DEVICE) 
    print(model_discriminator)

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
        print("Чекпоинт не найден или загрузка для новой архитектуры отключена. Обучение с нуля.")
        start_epoch_count = 0; best_val_metric_overall = float('inf')



    print(f"Начало обучения на {DEVICE} для {config.epochs} эпох...")
    try:
        for epoch_num_current in range(start_epoch_count, config.epochs): # epoch
            best_val_metric_overall = train_loop(
                train_loader_instance, val_loader_instance,
                model_generator, model_discriminator, 
                optimizer_G_instance, optimizer_D_instance,
                scheduler_G_instance, scheduler_D_instance,
                criterion_reconstruction_loss, criterion_gan_loss_fn,
                epoch_num_current, best_val_metric_overall, # epoch, best_val
                device=DEVICE,
                save_dir=checkpoints_directory, # save_dir
                val_loop_fn=val_loop if val_loader_instance else None, # val_loop_fn
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
