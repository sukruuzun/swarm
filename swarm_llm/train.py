"""
Swarm-LLM Eğitim Araçları
============================
Model eğitimi, öğrenme hızı zamanlayıcıları ve
dağıtık eğitim yardımcıları.
"""

import math
import time
from typing import Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from swarm_llm.config import SwarmConfig
from swarm_llm.model import SwarmLLM


class CosineWarmupScheduler:
    """
    Kosinüs ısınma öğrenme hızı zamanlayıcısı.

    1. Isınma (warmup): 0 → lr_max (doğrusal)
    2. Azalma (decay): lr_max → lr_min (kosinüs)

    Sığırcık benzetmesi:
    - Isınma: Sürü oluşum aşaması (kuşlar birbirini buluyor)
    - Azalma: Sürü kararlı hale geliyor (hareket azalıyor)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        lr_max: float = 3e-4,
        lr_min: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Doğrusal ısınma
            return self.lr_max * self.current_step / self.warmup_steps
        elif self.current_step >= self.max_steps:
            return self.lr_min
        else:
            # Kosinüs azalma
            progress = (self.current_step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )


class TextDataset(Dataset):
    """
    Basit metin veri seti.

    Token dizisini sabit uzunluklu bloklara böler.
    Her blok bir eğitim örneğidir.
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        """
        Args:
            token_ids: (total_tokens,) tüm token ID'leri
            block_size: Her eğitim örneğinin uzunluğu
        """
        self.token_ids = token_ids
        self.block_size = block_size
        self.num_samples = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1:end + 1]
        return x, y


class SwarmTrainer:
    """
    Swarm-LLM Eğitim Döngüsü.

    Özellikler:
    - Gradient accumulation (küçük GPU'lar için)
    - Gradient clipping (kararlılık)
    - Periyodik değerlendirme
    - VRAM kullanım raporlama
    """

    def __init__(
        self,
        model: SwarmLLM,
        config: SwarmConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        eval_interval: int = 500,
        log_interval: int = 10,
        device: str = 'auto',
    ):
        # Cihaz seçimi
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_interval = eval_interval
        self.log_interval = log_interval

        # Veri yükleyiciler
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=gradient_accumulation_steps,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=gradient_accumulation_steps,
                shuffle=False,
                pin_memory=True,
            )

        # Optimizer (AdamW -- ağırlık çürütme ile)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # LR zamanlayıcı
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            lr_max=config.learning_rate,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Doğrulama kaybını hesapla."""
        if self.val_loader is None:
            return float('nan')

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x, targets=y)
            total_loss += outputs['loss'].item()
            num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def train(self) -> list[dict]:
        """
        Ana eğitim döngüsü.

        Returns:
            Eğitim geçmişi (kayıp, öğrenme hızı, vb.)
        """
        self.model.train()
        history = []

        step = 0
        epoch = 0
        running_loss = 0.0
        start_time = time.time()

        print(f"{'='*60}")
        print(f" Swarm-LLM Eğitimi Başlıyor")
        print(f" Cihaz: {self.device}")
        print(f" Parametreler: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f" Max adım: {self.config.max_steps:,}")
        print(f"{'='*60}\n")

        while step < self.config.max_steps:
            epoch += 1
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # İleri geçiş
                outputs = self.model(x, targets=y)
                loss = outputs['loss'] / self.gradient_accumulation_steps

                # Geri yayılım
                loss.backward()

                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient kırpma
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                running_loss += outputs['loss'].item()
                step += 1

                # Loglama
                if step % self.log_interval == 0:
                    avg_loss = running_loss / self.log_interval
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        step * x.shape[0] * x.shape[1] / elapsed
                    )
                    lr = self.scheduler.get_lr()

                    log_entry = {
                        'step': step,
                        'loss': avg_loss,
                        'lr': lr,
                        'tokens_per_sec': tokens_per_sec,
                        'elapsed_sec': elapsed,
                    }

                    print(
                        f"  Adım {step:>6d} | "
                        f"Kayıp: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Token/s: {tokens_per_sec:.0f}"
                    )
                    running_loss = 0.0

                    # Değerlendirme
                    if step % self.eval_interval == 0:
                        val_loss = self.evaluate()
                        log_entry['val_loss'] = val_loss
                        print(f"  {'─'*40}")
                        print(f"  Doğrulama Kaybı: {val_loss:.4f}")
                        print(f"  {'─'*40}")

                    history.append(log_entry)

                if step >= self.config.max_steps:
                    break

        print(f"\n{'='*60}")
        print(f" Eğitim tamamlandı! Toplam adım: {step}")
        print(f" Son kayıp: {history[-1]['loss']:.4f}" if history else "")
        print(f"{'='*60}")

        return history
