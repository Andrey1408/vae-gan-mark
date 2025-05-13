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
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torchinfo import summary
from pathlib import Path 
import wandb 
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = secret_value_0
# --------------- Константы и Конфигурация ---------------
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
Z_CH = 128 # Размерность латентного пространства VAE
TEXT_CH = 64 # Целевая размерность текстового эмбеддинга после проекции
PATCH_SHAPE = (448, 64) # (Ширина, Высота)
TRANSFORMER_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LR_G = 1e-4 # Learning Rate Генератора
LR_D = 1e-4 # Learning Rate Дискриминатора
KL_WEIGHT = 0.01 # Вес KL дивергенции
GAN_WEIGHT = 0.1 # Вес GAN потерь для генератора
PERC_WEIGHT = 0.05 # Вес перцептивных потерь
RECON_WEIGHT = 1.0 # Вес L1 потерь реконструкции
GRAD_CLIP_NORM = 1.0 # Максимальная норма для отсечения градиентов

WANDB_PROJECT = "VAE-GAN" # Название вашего проекта в wandb
WANDB_ENTITY = None # Ваш username или команда в wandb (можно оставить None)
WANDB_RUN_NAME = "old test 100 epoch" # Имя запуска (можно задать или оставить None для автогенерации)
WANDB_SAVE_CODE = True
# --------------- Архитектура Моделей ---------------

class VAEEncoder(nn.Module):
    """Энкодер VAE, принимающий изображение и маску."""
    def __init__(self, in_ch=4, z_ch=Z_CH):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 1024, 3, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(True),
        )
        feature_map_h = PATCH_SHAPE[1] // 16
        feature_map_w = PATCH_SHAPE[0] // 16
        self.mu_head     = nn.Conv2d(1024, z_ch, kernel_size=(feature_map_h, feature_map_w))
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
        in_ch = z_ch + text_ch
        start_h = PATCH_SHAPE[1] // 16
        start_w = PATCH_SHAPE[0] // 16
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 1024, kernel_size=(start_h, start_w), stride=1, padding=0), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Выход в диапазоне [0, 1]
        )
    def forward(self, z): return self.decode(z)

class TransformerTextEncoder(nn.Module):
    """Кодирует текст с помощью Sentence Transformer и проецирует эмбеддинг."""
    def __init__(self, model_name=TRANSFORMER_MODEL_NAME, out_dim=TEXT_CH):
        super().__init__()
        self.device = DEVICE
        print(f"Загрузка модели Sentence Transformer: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
             print(f"Ошибка загрузки модели {model_name}: {e}")
             print("Попытка загрузки с CPU...")
             self.device = "cpu"
             self.model = SentenceTransformer(model_name, device=self.device)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Размерность эмбеддинга модели: {embedding_dim}")
        self.fc = nn.Linear(embedding_dim, out_dim)
        self.out_dim = out_dim
        print(f"Текстовый эмбеддинг будет спроецирован в размерность: {out_dim}")
        self.fc.to(self.device)

    def forward(self, texts):
        try:
            # Перемещаем модель на устройство перед encode на всякий случай
            self.model.to(self.device)
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        except Exception as e:
             print(f"\nОшибка при кодировании текстов: {e}")
             print(f"Проблемные тексты (первые 5): {texts[:5]}")
             return torch.zeros((len(texts), self.out_dim), device=self.device)
        projected_embeddings = self.fc(embeddings.to(self.fc.weight.device)) # Убедимся, что эмбеддинги на том же устройстве, что и fc
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
        # Рассмотреть добавление skip connections (U-Net стиль) здесь

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, mask, texts):
        x = torch.cat([image, mask], dim=1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        text_emb = self.text_encoder(texts)
        text_brd = spatial_broadcast(text_emb, z.shape[2:])
        zc = torch.cat([z, text_brd], dim=1)
        recon = self.decoder(zc)
        return recon, mu, logvar

class Discriminator(nn.Module):
    """Дискриминатор PatchGAN со Spectral Normalization и Instance Normalization."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)), nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Финальный слой для карты предсказаний
        )
    def forward(self, x): return self.body(x)

# --------------- Геометрические Утилиты ---------------

def perspective_crop(image_pil, bbox, out_shape):
    """Вырезает и преобразует полигональную область bbox в прямоугольный патч."""
    img = np.array(image_pil)
    try:
        pts_src = np.array(bbox, dtype=np.float32).reshape(4, 2)
    except Exception as e:
        print(f"Ошибка преобразования bbox в NumPy массив: {e}, bbox: {bbox}")
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))
    if pts_src.shape != (4, 2):
        print(f"Предупреждение: bbox должен иметь форму (4,2), но получено {pts_src.shape}. bbox: {bbox}")
        return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

    width, height = out_shape
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        patch = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Коррекция каналов
        if img.ndim == 3 and patch.ndim == 2: patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 2 and patch.ndim == 3: patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        # Коррекция типа данных
        if patch.dtype != np.uint8: patch = np.clip(patch, 0, 255).astype(np.uint8)
        return Image.fromarray(patch)
    except Exception as e:
         print(f"Ошибка в cv2.getPerspectiveTransform или cv2.warpPerspective: {e}")
         return Image.new(image_pil.mode, out_shape, 0 if image_pil.mode == 'L' else (0,0,0))

def perspective_unwarp(patch_pil, bbox, canvas_shape):
    """Применяет обратное перспективное преобразование к патчу."""
    patch = np.array(patch_pil)
    h, w = patch.shape[:2]
    pts_src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    pts_dst = np.float32(bbox).reshape(4, 2)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas_h, canvas_w = canvas_shape[0], canvas_shape[1]
    cv2.warpPerspective(patch, M, (canvas_w, canvas_h), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_LINEAR)
    return canvas

def pad_to_fixed_size(img, target_w, target_h, fill=0):
    """Дополняет изображение до фиксированного размера (не используется)."""
    w, h = img.size
    if w == target_w and h == target_h: return img
    fill_color = fill
    if img.mode == "RGB": fill_color = (fill, fill, fill)
    new_img = Image.new(img.mode, (target_w, target_h), color=fill_color)
    paste_x = max(0, (target_w - w) // 2)
    paste_y = max(0, (target_h - h) // 2)
    new_img.paste(img, (paste_x, paste_y))
    return new_img

# --------------- Датасет и Загрузчик ---------------

class MarkDatasetAnyBBox(Dataset):
    """Загружает пары изображений, маски и аннотации."""
    def __init__(self, json_dir, ru_image_dir, en_image_dir, mask_dir, out_shape=(448, 64)):
        self.json_dir = json_dir
        self.ru_image_dir = ru_image_dir
        self.en_image_dir = en_image_dir
        self.mask_dir = mask_dir
        self.out_shape = out_shape # W, H
        self.samples = []
        print(f"Инициализация датасета с out_shape (Ш, В): {self.out_shape}")

        for fname in os.listdir(json_dir):
            if not fname.lower().endswith(".json"): continue
            img_name_base = Path(fname).stem

            ru_path_found, en_path_found, mask_path_found = None, None, None
            found_paths = False
            for ext in [".jpg", ".png", ".jpeg", ".webp"]:
                ru_filename = f"{img_name_base}_ru{ext}"
                en_filename = f"{img_name_base}_en{ext}"
                mask_filename = f"{img_name_base}_ru.png" # Маска всегда _ru.png

                ru_path = os.path.join(ru_image_dir, ru_filename)
                en_path = os.path.join(en_image_dir, en_filename)
                mask_path = os.path.join(mask_dir, mask_filename)

                if os.path.exists(ru_path):
                    ru_path_found = ru_path
                    if os.path.exists(en_path): en_path_found = en_path
                    if os.path.exists(mask_path): mask_path_found = mask_path
                    found_paths = True; break
            if not found_paths: continue

            try:
                with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as f: annots = json.load(f)
            except Exception as e: print(f"Предупреждение: Ошибка чтения {fname}: {e}"); continue

            for idx, item in enumerate(annots):
                if not isinstance(item, dict): continue
                bbox_ru_data = item.get("bbox_ru"); bbox_en_data = item.get("bbox_en"); text_data = item.get("text")

                # Валидация
                if not (isinstance(bbox_ru_data, list) and len(bbox_ru_data) == 4 and all(isinstance(p, list) and len(p) == 2 for p in bbox_ru_data)): continue
                if not isinstance(text_data, str): continue
                if bbox_en_data and not (isinstance(bbox_en_data, list) and len(bbox_en_data) == 4 and all(isinstance(p, list) and len(p) == 2 for p in bbox_en_data)): bbox_en_data = None

                self.samples.append({"ru_image_path": ru_path_found, "en_image_path": en_path_found, "mask_path": mask_path_found, "bbox_ru": bbox_ru_data, "bbox_en": bbox_en_data, "text": text_data})
        print(f"Инициализация завершена. Найдено {len(self.samples)} сэмплов.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image_ru = Image.open(item['ru_image_path']).convert("RGB")
            image_en = Image.open(item['en_image_path']).convert("RGB") if item['en_image_path'] else Image.new("RGB", image_ru.size, (0,0,0))
            mask = Image.open(item['mask_path']).convert("L") if item['mask_path'] else Image.new("L", image_ru.size, 0)
            bbox_ru = item['bbox_ru']; bbox_en = item['bbox_en']; text = item['text']

            ru_patch = perspective_crop(image_ru, bbox_ru, self.out_shape)
            if bbox_en: en_patch = perspective_crop(image_en, bbox_en, self.out_shape)
            else: en_patch = Image.new("RGB", self.out_shape, (0,0,0)) # Черный en_patch если bbox_en нет
            mask_patch = perspective_crop(mask, bbox_ru, self.out_shape) # Маска всегда по bbox_ru

            to_tensor = T.ToTensor()
            ru_tensor = to_tensor(ru_patch); en_tensor = to_tensor(en_patch); mask_tensor = to_tensor(mask_patch)
            return ru_tensor, en_tensor, mask_tensor, text

        except Exception as e:
            print(f"Ошибка обработки {idx}: {item.get('ru_image_path', 'N/A')}. Ошибка: {e}")
            dummy_tensor = torch.zeros((3, self.out_shape[1], self.out_shape[0]))
            dummy_mask = torch.zeros((1, self.out_shape[1], self.out_shape[0]))
            return dummy_tensor, dummy_tensor, dummy_mask, ""

def safe_collate(batch):
    """Безопасная функция сборки батча, отфильтровывающая None."""
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        print("Предупреждение: Не удалось загрузить весь батч.")
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------- Функции Потерь ---------------

@torch.no_grad()
def get_vgg_feat(device):
    """Загружает VGG16 и слой нормализации для перцептивной потери."""
    weights = VGG16_Weights.IMAGENET1K_V1
    vgg = vgg16(weights=weights).features[:16].to(device).eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return vgg, normalize

def perceptual_loss(fake, real, vgg, vgg_normalize):
    """Вычисляет перцептивную потерю L1 между признаками VGG."""
    fake_norm = vgg_normalize(fake); real_norm = vgg_normalize(real)
    return F.l1_loss(vgg(fake_norm), vgg(real_norm))

def hinge_loss(preds, target):
    """Реализация Hinge Loss для GAN."""
    if target == 1: # Для реальных (дискриминатор)
        return F.relu(1.0 - preds).mean()
    elif target == 0: # Для фейковых (дискриминатор)
        return F.relu(1.0 + preds).mean()
    else: # Для генератора (target is None)
        return -preds.mean()

# --------------- Циклы Обучения и Валидации ---------------

@torch.no_grad()
def val_loop(val_loader, model, criterion_recon, epoch, device, show_patches=8):
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
               criterion_recon, criterion_gan, epoch, best_val, device, save_dir,
               val_loop_fn=None):
    """Основной цикл обучения на одну эпоху."""
    model.train(); disc.train()
    total_loss_G, total_loss_D, total_recon_loss, total_kl_loss, total_gan_loss, total_perc_loss = 0, 0, 0, 0, 0, 0
    vgg, vgg_normalize = get_vgg_feat(device)

    # Настраиваем tqdm для чистого вывода
    # progress_bar = tqdm(
    #     train_loader,
    #     desc=f"Эпоха {epoch+1} Обучение",
    #     unit="batch",
    #     mininterval=2.0, # Увеличиваем интервал обновления
    #     dynamic_ncols=True,
    #     leave=False,
    #     ascii=True # Используем ASCII символы
    # )

    for batch_data in tqdm(train_loader):
        if batch_data is None: continue
        ru_patch, en_patch, mask_patch, text_en = batch_data
        ru_patch, en_patch, mask_patch = ru_patch.to(device), en_patch.to(device), mask_patch.to(device)

        try: fake_patch_en, mu, logvar = model(ru_patch, mask_patch, text_en)
        except Exception as e: print(f"\nОшибка forward pass: {e}"); continue

        # Обучение Дискриминатора
        opt_D.zero_grad()
        real_preds = disc(en_patch)
        loss_D_real = criterion_gan(real_preds, 1)
        fake_preds_detached = disc(fake_patch_en.detach())
        loss_D_fake = criterion_gan(fake_preds_detached, 0)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(); opt_D.step()

        # Обучение Генератора
        opt_G.zero_grad()
        fake_preds = disc(fake_patch_en)
        recon_loss = criterion_recon(fake_patch_en, en_patch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]); kl_loss = torch.mean(kl_loss)
        gan_loss = criterion_gan(fake_preds, None) # Цель None для hinge loss генератора
        perc_loss = perceptual_loss(fake_patch_en, en_patch, vgg, vgg_normalize)
        loss_G = (RECON_WEIGHT * recon_loss + KL_WEIGHT * kl_loss + GAN_WEIGHT * gan_loss + PERC_WEIGHT * perc_loss)
        loss_G.backward(); clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM); opt_G.step()

        # Аккумуляция потерь
        total_loss_G += loss_G.item(); total_loss_D += loss_D.item(); total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item(); total_gan_loss += gan_loss.item(); total_perc_loss += perc_loss.item()

    # Вычисление и вывод средних потерь обучения
    num_batches = len(train_loader); avg_G, avg_D, avg_recon, avg_kl, avg_gan, avg_perc = [l / max(1, num_batches) for l in [total_loss_G, total_loss_D, total_recon_loss, total_kl_loss, total_gan_loss, total_perc_loss]]
    print(f"\nЭпоха {epoch+1} Средние Потери (Train): G={avg_G:.4f}, D={avg_D:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, GAN_G={avg_gan:.4f}, Perc={avg_perc:.4f}")

    # Логирование трейн метрик в wandb
    wandb.log({
        "epoch": epoch + 1,
        "train/generator_loss": avg_G, "train/discriminator_loss": avg_D,
        "train/recon_loss": avg_recon, "train/kl_loss": avg_kl,
        "train/gan_loss_g": avg_gan, "train/perceptual_loss": avg_perc,
        "learning_rate/generator": opt_G.param_groups[0]['lr'],
        "learning_rate/discriminator": opt_D.param_groups[0]['lr']
    }, step=epoch+1) # Используем epoch+1 как шаг (соответствует номеру эпохи)

    # Валидация
    avg_val_loss = float('inf')
    if val_loop_fn is not None and val_loader is not None:
        avg_val_loss = val_loop_fn(val_loader, model, criterion_recon, epoch+1, device)

    # Сохранение чекпоинтов
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = { 'epoch': epoch, 'model_state_dict': model.state_dict(), 'disc_state_dict': disc.state_dict(), 'opt_G_state_dict': opt_G.state_dict(), 'opt_D_state_dict': opt_D.state_dict(), 'best_val_loss': best_val }
    torch.save(checkpoint, os.path.join(save_dir, 'last_checkpoint.pth'))

    # Сохранение лучшей модели на основе валидационной метрики
    current_val_metric = avg_val_loss
    if current_val_metric < best_val:
        best_val = current_val_metric
        checkpoint['best_val_loss'] = best_val # Обновляем best_val в сохраняемом чекпоинте
        print(f"Сохранение лучшей модели с Validation Recon Loss: {current_val_metric:.4f} (предыдущая: {checkpoint['best_val_loss']:.4f})")
        torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
        wandb.summary["best_val_recon_loss"] = best_val # Обновляем summary в wandb
        try:
            best_artifact_name = f'best-model-checkpoint-run-{wandb.run.id}'
            best_artifact = wandb.Artifact(
                best_artifact_name, 
                type='model',
                description=f'Best model for run {wandb.run.id}, based on validation recon loss (updated at epoch {epoch+1})',
                metadata={'epoch': epoch + 1, 'val_recon_loss': best_val, 'run_id': wandb.run.id}
            )
            best_artifact.add_file(best_model_path, name='best_model.pth') 
            wandb.log_artifact(best_artifact, aliases=['best', f'epoch-{epoch+1}'])
            print(f"Артефакт '{best_artifact_name}' (эпоха {epoch+1}) обновлен/сохранен в wandb с алиасом 'best'.")
        except Exception as e:
            print(f"Ошибка сохранения артефакта best_model в wandb: {e}")
    else:
        print(f"Текущая метрика Val Recon ({current_val_metric:.4f}) не лучше лучшей ({best_val:.4f})")

    return best_val

# --------------- Основная Функция ---------------

def main():
    """Главная функция: инициализация, загрузка данных, обучение."""
    # Инициализация wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate_g": LR_G, "learning_rate_d": LR_D,
            "z_dim": Z_CH, "text_dim": TEXT_CH, "patch_width": PATCH_SHAPE[0], "patch_height": PATCH_SHAPE[1],
            "kl_weight": KL_WEIGHT, "gan_weight": GAN_WEIGHT, "recon_weight": RECON_WEIGHT, "perc_weight": PERC_WEIGHT,
            "grad_clip_norm": GRAD_CLIP_NORM, "text_encoder_model": TRANSFORMER_MODEL_NAME, "device": DEVICE,
            # Добавьте пути к данным в config, если хотите их логировать
            # "json_dir": '/kaggle/input/annot-data/all_annotations',
            # "ru_image_dir": '/kaggle/input/marketing-big-data-true/marketing_materials_big/aug_ru',
            # "en_image_dir": '/kaggle/input/marketing-big-data-true/marketing_materials_big/aug_en',
            # "mask_dir": '/kaggle/input/marketing-big-data-true/marketing_materials_big/masks',
        }
    )
    config = wandb.config # Используем config из wandb

    # Создание датасета
    print("Создание датасета...")
    dataset = MarkDatasetAnyBBox(
        json_dir = '/kaggle/input/marketing-data-new/marketing_materials_big/all_annotations', # Замените или возьмите из config
        ru_image_dir = '/kaggle/input/marketing-data-new/marketing_materials_big/aug_ru', # Замените или возьмите из config
        en_image_dir = '/kaggle/input/marketing-data-new/marketing_materials_big/aug_en', # Замените или возьмите из config
        mask_dir = '/kaggle/input/marketing-data-new/marketing_materials_big/masks_from_ru_bbox', # Замените или возьмите из config
        out_shape = (config.patch_width, config.patch_height)
    )

    # Разделение на Train/Val
    print("Разделение данных на Train/Val...")
    all_img_paths = [sample['ru_image_path'] for sample in dataset.samples]
    unique_imgs = sorted(list(set(filter(None, all_img_paths))))
    if not unique_imgs: print("Ошибка: Нет валидных путей."); wandb.finish(); return
    try: train_imgs, val_imgs = train_test_split(unique_imgs, test_size=0.1, random_state=42)
    except ValueError as e: print(f"Ошибка разделения: {e}"); train_imgs = unique_imgs; val_imgs = []
    train_idx = [i for i, s in enumerate(dataset.samples) if s['ru_image_path'] in train_imgs]
    val_idx = [i for i, s in enumerate(dataset.samples) if s['ru_image_path'] in val_imgs]
    print(f"Train/Val сэмплы: {len(train_idx)} / {len(val_idx)}")

    # Создание Загрузчиков Данных
    print("Создание DataLoader'ов...")
    num_workers = 2
    train_loader = DataLoader( Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=safe_collate)
    val_loader = None
    if val_idx: val_loader = DataLoader( Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)
    else: print("Нет валидационных данных.")

    # Инициализация Моделей
    print("Инициализация моделей...")
    model = VAEGAN(in_ch=4, z_ch=config.z_dim, text_ch=config.text_dim, out_ch=3).to(DEVICE)
    disc = Discriminator(in_ch=3).to(DEVICE)

    criterion_recon = nn.L1Loss().to(DEVICE)
    criterion_gan = hinge_loss 

    # Инициализация Оптимизаторов
    opt_G = optim.Adam(model.parameters(), lr=config.learning_rate_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(disc.parameters(), lr=config.learning_rate_d, betas=(0.5, 0.999))

    # Загрузка Чекпоинта
    start_epoch = 0
    best_val = float('inf')
    save_dir = './checkpoints_vaegan_wandb' # Папка для чекпоинтов
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'last_checkpoint.pth')

    if os.path.exists(checkpoint_path):
        print(f"Возобновление с чекпоинта: {checkpoint_path}")
        try:
             checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
             model.load_state_dict(checkpoint['model_state_dict'])
             disc.load_state_dict(checkpoint['disc_state_dict'])
             opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
             opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
             start_epoch = checkpoint['epoch'] + 1
             best_val = checkpoint.get('best_val_loss', float('inf')) # Используем сохраненное значение best_val_loss
             print(f"Загружен чекпоинт эпохи {checkpoint['epoch']}. Старт с {start_epoch}.")
             print(f"Лучшая валидационная потеря: {best_val:.4f}")
             # Перемещение состояний оптимизатора
             for state in opt_G.state.values():
                 for k, v in state.items():
                     if isinstance(v, torch.Tensor): state[k] = v.to(DEVICE)
             for state in opt_D.state.values():
                 for k, v in state.items():
                     if isinstance(v, torch.Tensor): state[k] = v.to(DEVICE)
        except Exception as e:
             print(f"Ошибка загрузки чекпоинта: {e}. Старт с нуля.")
             start_epoch = 0; best_val = float('inf')

    # Отслеживание моделей с wandb (опционально, может замедлять)
    # wandb.watch(model, log="all", log_freq=max(100, len(train_loader) // 2))
    # wandb.watch(disc, log="all", log_freq=max(100, len(train_loader) // 2))

    # Запуск Обучения
    print(f"Начало обучения на {DEVICE} для {config.epochs} эпох...")
    try:
        for epoch in range(start_epoch, config.epochs):
            # print(f"\n=== Эпоха {epoch+1}/{config.epochs} ===") # Убрано, т.к. train_loop выводит
            current_best_val = train_loop(
                train_loader, val_loader,
                model, disc, opt_G, opt_D,
                criterion_recon, criterion_gan,
                epoch, best_val, # Передаем текущее best_val
                device=DEVICE,
                save_dir=save_dir,
                val_loop_fn=val_loop if val_loader else None
            )
            # Обновляем best_val для следующей эпохи
            best_val = current_best_val # train_loop возвращает обновленное best_val

        print("Обучение завершено.")
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
    finally:
        # Завершение сессии wandb
        wandb.finish()
        print("Сессия wandb завершена.")

if __name__ == "__main__":
    main()
