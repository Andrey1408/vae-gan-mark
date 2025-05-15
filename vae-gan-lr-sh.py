# --------------- Импорты ---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from torchvision.models import vgg16, VGG16_Weights
from torch.nn.utils import spectral_norm, clip_grad_norm_
from sentence_transformers import SentenceTransformer

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt # Оставлено, но не используется для отображения в этом скрипте
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torchinfo import summary # Оставлено, но summary вызовы могут быть закомментированы для скорости
from pathlib import Path
import wandb

os.environ["WANDB_API_KEY"] = "f9bd53ddbed845e1c532581b230e7da2dbc3673f"

# --------------- Константы и Конфигурация ---------------
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150
Z_CH = 128
TEXT_CH = 64
PATCH_SHAPE = (448, 64)  # (Ширина, Высота)
TRANSFORMER_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LR_G = 1e-4
LR_D = 1e-4
KL_WEIGHT = 0.005 
GAN_WEIGHT = 0.1
PERC_WEIGHT = 0.05
RECON_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0

# Параметры для ReduceLROnPlateau
SCHEDULER_MODE = 'min'      # 'min' для метрик, которые нужно минимизировать (например, loss)
SCHEDULER_FACTOR = 0.2    # Фактор уменьшения LR (lr = lr * factor)
SCHEDULER_PATIENCE = 10   # Количество эпох без улучшения, после которых LR снижается
SCHEDULER_THRESHOLD = 0.0001 # Порог для измерения значительного изменения
SCHEDULER_MIN_LR = 1e-7     # Минимальный LR, до которого может снизиться
SCHEDULER_VERBOSE = True    # Выводить сообщение при снижении LR

WANDB_PROJECT = "VAE-GAN"
WANDB_ENTITY = None
WANDB_RUN_NAME = "a10_ReduceLROnP_KL005" # Обновленное имя для отражения изменений
WANDB_SAVE_CODE = True

# --------------- Архитектура Моделей ---------------
class VAEEncoder(nn.Module):
    """Энкодер VAE, принимающий изображение и маску."""
    def __init__(self, in_ch=4, z_ch=Z_CH):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        feature_map_h = PATCH_SHAPE[1] // 16
        feature_map_w = PATCH_SHAPE[0] // 16
        self.mu_head = nn.Conv2d(1024, z_ch, kernel_size=(feature_map_h, feature_map_w))
        self.logvar_head = nn.Conv2d(1024, z_ch, kernel_size=(feature_map_h, feature_map_w))

    def forward(self, x):
        h = self.feat(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    """Декодер VAE, принимающий латентный код стиля и текстовый эмбеддинг."""
    def __init__(self, z_ch=Z_CH, text_ch=TEXT_CH, out_ch=3):
        super().__init__()
        in_ch_decoder = z_ch + text_ch # Переименовал для ясности
        start_h = PATCH_SHAPE[1] // 16
        start_w = PATCH_SHAPE[0] // 16
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_ch_decoder, 1024, kernel_size=(start_h, start_w), stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decode(z)

class TransformerTextEncoder(nn.Module):
    """Кодирует текст с помощью Sentence Transformer и проецирует эмбеддинг."""
    def __init__(self, model_name=TRANSFORMER_MODEL_NAME, out_dim=TEXT_CH):
        super().__init__()
        # Это устройство будет использоваться для SBERT и его проекционного слоя.
        # Оно определяется на основе глобального DEVICE, но может быть CPU, если SBERT не загрузится на CUDA.
        self.sbert_device = DEVICE 
        print(f"TransformerTextEncoder: Попытка загрузки модели SBERT '{model_name}' на устройство '{self.sbert_device}'")
        try:
            self.model = SentenceTransformer(model_name, device=self.sbert_device)
        except Exception as e:
            print(f"TransformerTextEncoder: Ошибка загрузки SBERT на '{self.sbert_device}': {e}.")
            if self.sbert_device != "cpu": # Если уже не CPU, пробуем CPU
                print("TransformerTextEncoder: Попытка загрузки SBERT на 'cpu'.")
                self.sbert_device = "cpu"
                try:
                    self.model = SentenceTransformer(model_name, device=self.sbert_device)
                except Exception as e_cpu:
                    print(f"TransformerTextEncoder: КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить SBERT и на 'cpu': {e_cpu}")
                    raise
            else: # Если изначально было CPU и не удалось, то это фатально
                print(f"TransformerTextEncoder: КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить SBERT на 'cpu': {e}")
                raise
        
        print(f"TransformerTextEncoder: Модель SBERT успешно загружена и будет использоваться на устройстве '{self.sbert_device}'.")
        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"TransformerTextEncoder: Размерность эмбеддинга SBERT: {embedding_dim}")
        self.fc = nn.Linear(embedding_dim, out_dim)
        self.out_dim = out_dim
        print(f"TransformerTextEncoder: Эмбеддинг будет спроецирован в размерность: {out_dim}")
        self.fc.to(self.sbert_device) # Перемещаем fc слой на то же устройство, что и SBERT

    def forward(self, texts):
        # Убедимся, что модель и fc на правильном устройстве (на случай, если sbert_device был изменен на cpu)
        self.model.to(self.sbert_device)
        self.fc.to(self.sbert_device)
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.sbert_device)
        except Exception as e:
            print(f"\nTransformerTextEncoder: Ошибка при кодировании текстов: {e}")
            print(f"Проблемные тексты (первые 5): {texts[:5]}")
            return torch.zeros((len(texts), self.out_dim), device=self.sbert_device)
        
        projected_embeddings = self.fc(embeddings) # fc уже на self.sbert_device
        return projected_embeddings

def spatial_broadcast(text_emb, spatial_shape):
    """Размножает текстовый эмбеддинг до пространственных размеров."""
    B, C = text_emb.shape
    H, W = spatial_shape
    return text_emb.view(B, C, 1, 1).expand(B, C, H, W)

class VAEGAN(nn.Module):
    """Основная модель VAE-GAN, объединяющая компоненты."""
    def __init__(self, in_ch=4, z_ch=Z_CH, text_ch=TEXT_CH, out_ch=3):
        super().__init__()
        self.encoder = VAEEncoder(in_ch=in_ch, z_ch=z_ch)
        self.text_encoder = TransformerTextEncoder(out_dim=text_ch)
        self.decoder = VAEDecoder(z_ch=z_ch, text_ch=text_ch, out_ch=out_ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, mask, texts):
        # image, mask, mu, logvar, z будут на основном DEVICE модели
        x = torch.cat([image, mask], dim=1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # text_encoder возвращает эмбеддинги на своем sbert_device
        text_emb_on_sbert_device = self.text_encoder(texts)
        # Перемещаем эмбеддинги на устройство, где находится z, если они разные
        text_emb = text_emb_on_sbert_device.to(z.device) 
        
        text_brd = spatial_broadcast(text_emb, z.shape[2:])
        
        zc = torch.cat([z, text_brd], dim=1) # z и text_brd теперь на одном устройстве
        recon = self.decoder(zc)
        return recon, mu, logvar

class Discriminator(nn.Module):
    """Дискриминатор PatchGAN."""
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

# --------------- Геометрические Утилиты ---------------
def perspective_crop(image_pil, bbox, out_shape):
    """Вырезает и преобразует полигональную область bbox в прямоугольный патч."""
    if image_pil is None:
        return Image.new('RGB' if len(out_shape) == 2 else 'L', out_shape, 0)

    img_np = np.array(image_pil)
    try:
        pts_src = np.array(bbox, dtype=np.float32).reshape(4, 2)
    except Exception as e:
        # print(f"Ошибка преобразования bbox в NumPy массив: {e}, bbox: {bbox}")
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

    if pts_src.shape != (4, 2):
        # print(f"Предупреждение: bbox должен иметь форму (4,2), но получено {pts_src.shape}. bbox: {bbox}")
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

    width, height = out_shape
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    
    try:
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        patch_np = cv2.warpPerspective(img_np, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        if img_np.ndim == 3 and patch_np.ndim == 2:
            patch_np = cv2.cvtColor(patch_np, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 2 and patch_np.ndim == 3:
             patch_np = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        
        if patch_np.dtype != np.uint8:
            patch_np = np.clip(patch_np, 0, 255).astype(np.uint8)
            
        return Image.fromarray(patch_np)
    except Exception as e:
        # print(f"Ошибка в cv2.getPerspectiveTransform или cv2.warpPerspective: {e}")
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

# --------------- Датасет и Загрузчик ---------------
class MarkDatasetAnyBBox(Dataset):
    """Загружает пары изображений, маски и аннотации."""
    def __init__(self, json_dir, ru_image_dir, en_image_dir, mask_dir, out_shape=(448, 64)):
        self.json_dir = json_dir
        self.ru_image_dir = ru_image_dir
        self.en_image_dir = en_image_dir
        self.mask_dir = mask_dir
        self.out_shape = out_shape
        self.samples = []
        print(f"Инициализация датасета MarkDatasetAnyBBox с out_shape (Ш, В): {self.out_shape}")

        annotation_files = sorted(os.listdir(json_dir)) # Сортировка для детерминизма
        for fname in tqdm(annotation_files, desc="Чтение аннотаций", unit="файл"):
            if not fname.lower().endswith(".json"):
                continue
            
            img_name_base = Path(fname).stem
            ru_path_found, en_path_found, mask_path_found = None, None, None
            found_corresponding_paths = False
            
            for ext in [".jpg", ".png", ".jpeg", ".webp"]:
                ru_filename = f"{img_name_base}_ru{ext}"
                en_filename = f"{img_name_base}_en{ext}"
                mask_filename = f"{img_name_base}_ru.png"

                ru_filepath = os.path.join(ru_image_dir, ru_filename)
                en_filepath = os.path.join(en_image_dir, en_filename)
                mask_filepath = os.path.join(mask_dir, mask_filename)

                if os.path.exists(ru_filepath):
                    ru_path_found = ru_filepath
                    found_corresponding_paths = True
                    if os.path.exists(en_filepath):
                        en_path_found = en_filepath
                    if os.path.exists(mask_filepath):
                        mask_path_found = mask_filepath
                    break 
            
            if not found_corresponding_paths:
                continue

            try:
                with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as f:
                    annotations_in_file = json.load(f)
            except Exception as e:
                # print(f"\nПредупреждение: Ошибка чтения JSON файла {fname}: {e}")
                continue
            
            for item_data in annotations_in_file:
                if not isinstance(item_data, dict):
                    continue
                
                bbox_ru_coords = item_data.get("bbox_ru")
                bbox_en_coords = item_data.get("bbox_en")
                text_content = item_data.get("text")

                is_bbox_ru_valid = isinstance(bbox_ru_coords, list) and len(bbox_ru_coords) == 4 and \
                                   all(isinstance(p, list) and len(p) == 2 for p in bbox_ru_coords)
                if not is_bbox_ru_valid:
                    continue
                if not isinstance(text_content, str):
                    continue
                
                is_bbox_en_valid = isinstance(bbox_en_coords, list) and len(bbox_en_coords) == 4 and \
                                   all(isinstance(p, list) and len(p) == 2 for p in bbox_en_coords)
                if bbox_en_coords and not is_bbox_en_valid:
                    bbox_en_coords = None

                self.samples.append({
                    "ru_image_path": ru_path_found,
                    "en_image_path": en_path_found,
                    "mask_path": mask_path_found,
                    "bbox_ru": bbox_ru_coords,
                    "bbox_en": bbox_en_coords,
                    "text": text_content
                })
        print(f"Инициализация датасета завершена. Найдено {len(self.samples)} сэмплов.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_meta = self.samples[idx]
        try:
            image_ru_pil = Image.open(item_meta['ru_image_path']).convert("RGB")
            
            image_en_pil = Image.open(item_meta['en_image_path']).convert("RGB") \
                           if item_meta['en_image_path'] and os.path.exists(item_meta['en_image_path']) \
                           else Image.new("RGB", image_ru_pil.size, (0,0,0))
                           
            mask_pil = Image.open(item_meta['mask_path']).convert("L") \
                       if item_meta['mask_path'] and os.path.exists(item_meta['mask_path']) \
                       else Image.new("L", image_ru_pil.size, 0)
                       
            bbox_ru_item = item_meta['bbox_ru']
            bbox_en_item = item_meta['bbox_en']
            text_item = item_meta['text']

            ru_patch_pil = perspective_crop(image_ru_pil, bbox_ru_item, self.out_shape)
            if bbox_en_item:
                en_patch_pil = perspective_crop(image_en_pil, bbox_en_item, self.out_shape)
            else:
                en_patch_pil = Image.new("RGB", self.out_shape, (0,0,0))
            mask_patch_pil = perspective_crop(mask_pil, bbox_ru_item, self.out_shape)

            transform_to_tensor = T.ToTensor()
            ru_tensor_item = transform_to_tensor(ru_patch_pil)
            en_tensor_item = transform_to_tensor(en_patch_pil)
            mask_tensor_item = transform_to_tensor(mask_patch_pil)
            
            return ru_tensor_item, en_tensor_item, mask_tensor_item, text_item
        except Exception as e:
            # print(f"Ошибка обработки сэмпла {idx}: {item_meta.get('ru_image_path', 'N/A')}. Ошибка: {e}")
            dummy_rgb_tensor = torch.zeros((3, self.out_shape[1], self.out_shape[0]))
            dummy_mask_tensor = torch.zeros((1, self.out_shape[1], self.out_shape[0]))
            return dummy_rgb_tensor, dummy_rgb_tensor, dummy_mask_tensor, ""

def safe_collate(batch):
    """Безопасная функция сборки батча, отфильтровывающая None элементы."""
    filtered_batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not filtered_batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(filtered_batch)
    except Exception as e:
        return None

# --------------- Функции Потерь ---------------
@torch.no_grad()
def get_vgg_feat(device):
    """Загружает VGG16 и слой нормализации для перцептивной потери."""
    weights = VGG16_Weights.IMAGENET1K_V1
    vgg_model = vgg16(weights=weights).features[:16].to(device).eval()
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return vgg_model, normalize_transform

def perceptual_loss(fake_img, real_img, vgg_model, normalize_transform):
    """Вычисляет перцептивную потерю L1 между признаками VGG."""
    fake_normalized = normalize_transform(fake_img)
    real_normalized = normalize_transform(real_img)
    return F.l1_loss(vgg_model(fake_normalized), vgg_model(real_normalized))

def hinge_loss(predictions, target_is_real):
    """Реализация Hinge Loss для GAN."""
    if target_is_real == 1:
        return F.relu(1.0 - predictions).mean()
    elif target_is_real == 0:
        return F.relu(1.0 + predictions).mean()
    else:
        return -predictions.mean()

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

    return best_val_loss_so_far # Возвращаем обновленное значение best_val_loss_so_far

# --------------- Основная Функция ---------------
def main():
    """Главная функция: инициализация, загрузка данных, обучение."""
    global DEVICE
    
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate_g": LR_G,
            "learning_rate_d": LR_D, "z_dim": Z_CH, "text_dim": TEXT_CH,
            "patch_width": PATCH_SHAPE[0], "patch_height": PATCH_SHAPE[1],
            "kl_weight": KL_WEIGHT, "gan_weight": GAN_WEIGHT,
            "recon_weight": RECON_WEIGHT, "perc_weight": PERC_WEIGHT,
            "grad_clip_norm": GRAD_CLIP_NORM, "text_encoder_model": TRANSFORMER_MODEL_NAME,
            "initial_device": DEVICE,
            # Параметры для ReduceLROnPlateau
            "scheduler_mode": SCHEDULER_MODE, "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE, "scheduler_threshold": SCHEDULER_THRESHOLD,
            "scheduler_min_lr": SCHEDULER_MIN_LR
        }
    )
    config_wandb = wandb.config
    DEVICE = config_wandb.initial_device # Убедимся, что DEVICE соответствует конфигу
    print(f"Используемое устройство: {DEVICE}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Создание датасета...")
    dataset_json_dir = config_wandb.get("json_dir", '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/all_annotations')
    dataset_ru_img_dir = config_wandb.get("ru_image_dir", '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/aug_ru')
    dataset_en_img_dir = config_wandb.get("en_image_dir", '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/aug_en')
    dataset_mask_dir = config_wandb.get("mask_dir", '/home/ubuntu/.cache/kagglehub/datasets/andrey101/marketing-data-new/versions/1/marketing_materials_big/masks_from_ru_bbox')
    
    dataset_instance = MarkDatasetAnyBBox(
        json_dir=dataset_json_dir, ru_image_dir=dataset_ru_img_dir,
        en_image_dir=dataset_en_img_dir, mask_dir=dataset_mask_dir,
        out_shape=(config_wandb.patch_width, config_wandb.patch_height)
    )

    print("Разделение данных на Train/Val...")
    all_image_paths_dataset = [sample['ru_image_path'] for sample in dataset_instance.samples]
    unique_imgs_dataset = sorted(list(set(filter(None, all_image_paths_dataset))))
    if not unique_imgs_dataset:
        print("Ошибка: Нет валидных путей к изображениям."); wandb.finish(); return
    
    validation_split_size = config_wandb.get("validation_split_size", 0.1)
    try:
        train_imgs_paths, val_imgs_paths = train_test_split(unique_imgs_dataset, test_size=validation_split_size, random_state=42)
    except ValueError as e:
        print(f"Ошибка разделения данных: {e}. Используем все данные для обучения.");
        train_imgs_paths = unique_imgs_dataset
        val_imgs_paths = []
        
    train_indices = [i for i, s in enumerate(dataset_instance.samples) if s['ru_image_path'] in train_imgs_paths]
    val_indices = [i for i, s in enumerate(dataset_instance.samples) if s['ru_image_path'] in val_imgs_paths]
    print(f"Train/Val сэмплы: {len(train_indices)} / {len(val_indices)}")

    print("Создание DataLoader'ов...")
    num_dataloader_workers = config_wandb.get("num_dataloader_workers", 2)
    
    train_data_loader = DataLoader(
        Subset(dataset_instance, train_indices), batch_size=config_wandb.batch_size, shuffle=True,
        num_workers=num_dataloader_workers, pin_memory=True, drop_last=True, collate_fn=safe_collate
    )
    val_data_loader = None
    if val_indices:
        val_data_loader = DataLoader(
            Subset(dataset_instance, val_indices), batch_size=config_wandb.batch_size, shuffle=False,
            num_workers=num_dataloader_workers, pin_memory=True, collate_fn=safe_collate
        )
    else:
        print("Нет валидационных данных. Планировщик LR не будет корректно работать.")

    print("Инициализация моделей...")
    model_vaegan = VAEGAN(in_ch=4, z_ch=config_wandb.z_dim, text_ch=config_wandb.text_dim, out_ch=3).to(DEVICE)
    discriminator_model = Discriminator(in_ch=3).to(DEVICE)


    criterion_reconstruction = nn.L1Loss().to(DEVICE)
    criterion_gan_hinge = hinge_loss

    optimizer_generator = optim.Adam(model_vaegan.parameters(), lr=config_wandb.learning_rate_g, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator_model.parameters(), lr=config_wandb.learning_rate_d, betas=(0.5, 0.999))

    # --- ДОБАВЛЕНО: Инициализация Планировщиков ---
    scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_generator, mode=config_wandb.scheduler_mode, factor=config_wandb.scheduler_factor,
        patience=config_wandb.scheduler_patience, threshold=config_wandb.scheduler_threshold,
        min_lr=config_wandb.scheduler_min_lr, verbose=SCHEDULER_VERBOSE
    )
    scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_discriminator, mode=config_wandb.scheduler_mode, factor=config_wandb.scheduler_factor,
        patience=config_wandb.scheduler_patience, threshold=config_wandb.scheduler_threshold,
        min_lr=config_wandb.scheduler_min_lr, verbose=SCHEDULER_VERBOSE
    )
    # --- КОНЕЦ ДОБАВЛЕНИЯ ---

    start_epoch_number = 0
    best_validation_metric = float('inf')
    checkpoints_save_directory = './checkpoints_vaegan_wandb'
    os.makedirs(checkpoints_save_directory, exist_ok=True)
    last_checkpoint_file_path = os.path.join(checkpoints_save_directory, 'last_checkpoint.pth')

    if os.path.exists(last_checkpoint_file_path):
        print(f"Возобновление с чекпоинта: {last_checkpoint_file_path}")
        try:
            checkpoint_data_loaded = torch.load(last_checkpoint_file_path, map_location=DEVICE)
            
            model_to_load = model_vaegan.module if isinstance(model_vaegan, nn.DataParallel) else model_vaegan
            disc_to_load = discriminator_model.module if isinstance(discriminator_model, nn.DataParallel) else discriminator_model

            model_to_load.load_state_dict(checkpoint_data_loaded['model_state_dict'])
            disc_to_load.load_state_dict(checkpoint_data_loaded['disc_state_dict'])
            
            optimizer_generator.load_state_dict(checkpoint_data_loaded['opt_G_state_dict'])
            optimizer_discriminator.load_state_dict(checkpoint_data_loaded['opt_D_state_dict'])
            
            # --- ДОБАВЛЕНО: Загрузка состояний планировщиков ---
            if 'scheduler_G_state_dict' in checkpoint_data_loaded:
                scheduler_generator.load_state_dict(checkpoint_data_loaded['scheduler_G_state_dict'])
                print("Состояние scheduler_G успешно загружено.")
            if 'scheduler_D_state_dict' in checkpoint_data_loaded:
                scheduler_discriminator.load_state_dict(checkpoint_data_loaded['scheduler_D_state_dict'])
                print("Состояние scheduler_D успешно загружено.")
            # --- КОНЕЦ ДОБАВЛЕНИЯ ---
                
            start_epoch_number = checkpoint_data_loaded['epoch'] + 1
            best_validation_metric = checkpoint_data_loaded.get('best_val_loss', float('inf'))
            
            print(f"Загружен чекпоинт эпохи {checkpoint_data_loaded['epoch']}. Старт с {start_epoch_number}.")
            print(f"Лучшая валидационная потеря из чекпоинта: {best_validation_metric:.4f}")

            for opt_instance in [optimizer_generator, optimizer_discriminator]:
                for state_dict_opt_val in opt_instance.state.values(): # Переименовано
                    for key_opt_state, value_opt_state in state_dict_opt_val.items(): # Переименовано
                        if isinstance(value_opt_state, torch.Tensor):
                            state_dict_opt_val[key_opt_state] = value_opt_state.to(DEVICE)
        except Exception as e:
            print(f"Ошибка загрузки чекпоинта: {e}. Старт с нуля.")
            start_epoch_number = 0
            best_validation_metric = float('inf')


    print(f"Начало обучения на {DEVICE} для {config_wandb.epochs} эпох...")
    try:
        for epoch_counter in range(start_epoch_number, config_wandb.epochs):
            best_validation_metric = train_loop(
                train_data_loader, val_data_loader,
                model_vaegan, discriminator_model,
                optimizer_generator, optimizer_discriminator,
                scheduler_generator, scheduler_discriminator, 
                criterion_reconstruction, criterion_gan_hinge,
                epoch_counter, best_validation_metric,
                device=DEVICE,
                save_dir=checkpoints_save_directory,
                val_loop_fn=val_loop if val_data_loader else None
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
