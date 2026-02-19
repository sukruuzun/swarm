"""
🎓 OFFLINE ROUTER KALİBRASYONU — Mac'te Çalışır!
================================================================
A100 gerektirmez. Blokları tek tek yükleyerek:
1. Teacher logits hesaplar (tüm bloklar sırayla)
2. Her 2'li kombo için student logits hesaplar
3. En iyi komboyu bulur
4. Router'ı eğitir

Kullanım:
    python calibrate_router_offline.py
"""

import os
import sys
import time
import itertools

# MPS bellek limiti kaldır
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════════════════
SAVE_DIR = "model_blocks_qwen25_7b"
MODEL_NAME = "Qwen/Qwen2.5-7B"
TOP_K = 2  # Router kaç blok seçecek
CALIBRATION_FILE = os.path.join(SAVE_DIR, "calibration_data.pt")

CALIBRATION_PROMPTS = [
    "The history of artificial intelligence is",
    "In quantum computing, qubits differ from classical bits because",
    "The most important principles of software engineering include",
    "Climate change affects global ecosystems through",
    "Neural networks learn by adjusting weights through",
]

print("="*60)
print("🎓 OFFLINE ROUTER KALİBRASYONU")
print("="*60)

# ── Cihaz ──
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Cihaz: {device}")

# ── Tokenizer ──
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"Tokenizer: {tokenizer.vocab_size} token")

# ── Dosya kontrolü ──
block_files = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith("block_") and f.endswith(".pt")])
num_blocks = len(block_files)
print(f"Bloklar: {num_blocks} adet")

# ── Yardımcı: Tek blok yükle, çalıştır, sil ──
def load_and_run_block(block_idx, hidden_states, position_embeddings, position_ids):
    """Tek bloğu diskten yükle, çalıştır, bellekten sil."""
    block_path = os.path.join(SAVE_DIR, f"block_{block_idx}.pt")
    
    # CPU'ya yükle (mmap ile)
    try:
        block = torch.load(block_path, map_location='cpu', weights_only=False, mmap=True)
    except TypeError:
        block = torch.load(block_path, map_location='cpu', weights_only=False)
    
    # Dtype uyumu
    block = block.to(dtype=hidden_states.dtype)
    block = block.to(device)
    block.eval()
    
    # Çalıştır
    with torch.no_grad():
        if position_embeddings is not None:
            out = block(hidden_states, position_embeddings=position_embeddings,
                       position_ids=position_ids, attention_mask=None)
        else:
            out = block(hidden_states)
    
    # Çıktıyı ayrıştır
    if isinstance(out, tuple):
        result = out[0]
    else:
        result = out
    
    # Bellek temizle
    del block
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    return result

# ── Router.pt'den embed, norm, lm_head yükle ──
print(f"\n📦 Router + embed + norm + lm_head yükleniyor...")
router_data = torch.load(os.path.join(SAVE_DIR, "router.pt"), map_location='cpu', weights_only=False)

embed_layer = router_data["embed_layer"].to(device)
final_norm = router_data["final_norm"].to(device)
lm_head = router_data["lm_head"].to(device)
router = router_data["router"]  # CPU'da kalsın (eğitim için)

# Rotary embeddings
rotary_emb = None
rotary_path = os.path.join(SAVE_DIR, "rotary_emb.pt")
if os.path.exists(rotary_path):
    rotary_emb = torch.load(rotary_path, map_location=device, weights_only=False)
    print(f"✅ Rotary embeddings yüklendi")

embed_dtype = next(embed_layer.parameters()).dtype
print(f"✅ Embed/Norm/LM_Head yüklendi (dtype: {embed_dtype})")

# ── Yardımcı: Hidden states → logits ──
def get_logits(hidden_states):
    """Final norm + LM head ile logits hesapla."""
    x = final_norm(hidden_states.to(final_norm.weight.device))
    return lm_head(x)

# ── Yardımcı: Position embeddings hesapla ──
def get_position_info(seq_len):
    """Position IDs ve position embeddings hesapla."""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = None
    if rotary_emb is not None:
        try:
            dummy = torch.zeros(1, seq_len, embed_layer.weight.shape[1], 
                              dtype=embed_dtype, device=device)
            cos, sin = rotary_emb(dummy, position_ids)
            position_embeddings = (cos, sin)
        except Exception as e:
            print(f"⚠️  Rotary embeddings hatası: {e}")
    return position_ids, position_embeddings

# ═══════════════════════════════════════════════════════════════
# ADIM 1: TEACHER LOGITS (Tüm bloklar, sırayla)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"📖 ADIM 1: Teacher Logits (tüm {num_blocks} blok)")
print(f"{'='*60}")

teacher_data = []

for p_idx, prompt in enumerate(CALIBRATION_PROMPTS):
    print(f"\n  Prompt {p_idx+1}/{len(CALIBRATION_PROMPTS)}: '{prompt[:50]}...'")
    t0 = time.time()
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    
    # Embedding
    x = embed_layer(input_ids).to(dtype=embed_dtype)
    
    # Position
    position_ids, position_embeddings = get_position_info(seq_len)
    
    # Tüm blokları sırayla çalıştır
    for b_idx in range(num_blocks):
        print(f"    Block {b_idx}...", end=" ", flush=True)
        x = load_and_run_block(b_idx, x, position_embeddings, position_ids)
        print("✅")
    
    # Teacher logits (CPU'ya taşı — küçük)
    logits = get_logits(x)
    teacher_logits_cpu = logits.detach().cpu().float()
    
    # Hidden state'i de kaydet (router eğitimi için)
    embed_hidden = embed_layer(input_ids).detach().cpu().float()
    
    teacher_data.append({
        "prompt": prompt,
        "input_ids": input_ids.cpu(),
        "teacher_logits": teacher_logits_cpu,
        "embed_hidden": embed_hidden,
    })
    
    dt = time.time() - t0
    print(f"  ⏱️  {dt:.0f}s | Logits boyutu: {teacher_logits_cpu.shape}")
    
    # Bellek temizle
    del x, logits
    import gc
    gc.collect()

print(f"\n✅ Teacher logits tamamlandı!")

# ═══════════════════════════════════════════════════════════════
# ADIM 2: STUDENT LOGITS (Her 2'li kombo)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"📐 ADIM 2: Student Logits (C({num_blocks},{TOP_K}) = {len(list(itertools.combinations(range(num_blocks), TOP_K)))} kombo)")
print(f"{'='*60}")

combos = list(itertools.combinations(range(num_blocks), TOP_K))
combo_kl_scores = {}

for c_idx, combo in enumerate(combos):
    combo_kls = []
    print(f"\n  Kombo {c_idx+1}/{len(combos)}: Bloklar {combo}")
    
    for p_idx, data in enumerate(teacher_data):
        input_ids = data["input_ids"].to(device)
        teacher_logits = data["teacher_logits"]
        
        # Embedding
        x = embed_layer(input_ids).to(dtype=embed_dtype)
        
        # Position
        seq_len = input_ids.shape[1]
        position_ids, position_embeddings = get_position_info(seq_len)
        
        # Sadece seçilen blokları çalıştır (diğerleri identity/skip)
        for b_idx in range(num_blocks):
            if b_idx in combo:
                print(f"    Block {b_idx}...", end=" ", flush=True)
                x = load_and_run_block(b_idx, x, position_embeddings, position_ids)
                print("✅", end="")
            # else: identity (x = x)
        
        # Student logits
        student_logits = get_logits(x).detach().cpu().float()
        
        # KL divergence
        teacher_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = F.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(student_probs, teacher_probs.exp(), reduction='batchmean', log_target=False)
        combo_kls.append(kl.item())
        
        # Temizle
        del x, student_logits
        import gc
        gc.collect()
    
    avg_kl = sum(combo_kls) / len(combo_kls)
    combo_kl_scores[combo] = avg_kl
    print(f"\n  📊 KL divergence: {avg_kl:.4f}")

# ═══════════════════════════════════════════════════════════════
# ADIM 3: EN İYİ KOMBOYU BUL
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"🏆 SONUÇLAR:")
print(f"{'='*60}")

sorted_combos = sorted(combo_kl_scores.items(), key=lambda x: x[1])

for i, (combo, kl) in enumerate(sorted_combos):
    marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
    print(f"  {marker} Bloklar {combo}: KL = {kl:.4f}")

best_combo = sorted_combos[0][0]
best_kl = sorted_combos[0][1]
print(f"\n🏆 EN İYİ KOMBO: Bloklar {best_combo} (KL = {best_kl:.4f})")

# ═══════════════════════════════════════════════════════════════
# ADIM 4: ROUTER EĞİTİMİ
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"🎓 ADIM 4: Router Eğitimi")
print(f"{'='*60}")

# Hedef dağılım: En iyi 3 kombo üzerinden keskin softmax
top3 = sorted_combos[:3]
target_scores = torch.zeros(num_blocks)
for combo, kl in top3:
    # Düşük KL = iyi → yüksek skor
    score = 1.0 / (kl + 1e-8)
    for b in combo:
        target_scores[b] += score

# Softmax ile normalize et (keskin dağılım)
target_dist = F.softmax(target_scores / 0.1, dim=0)
print(f"  Hedef dağılım: {[f'{v:.3f}' for v in target_dist.tolist()]}")

# Router'ı eğit
router_dtype = next(router.parameters()).dtype
router = router.float()  # Eğitim için float32
optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

for step in range(200):
    total_loss = 0
    for data in teacher_data:
        embed_hidden = data["embed_hidden"].to('cpu')
        
        # Router çıktısı
        probs, indices, aux_loss, weights = router(embed_hidden, pool_input=True)
        
        # Cross-entropy: Router'ın çıktısını hedef dağılıma yaklaştır
        router_log_probs = torch.log(probs.squeeze() + 1e-10)
        loss = F.kl_div(router_log_probs, target_dist, reduction='batchmean')
        
        if aux_loss is not None:
            loss = loss + 0.01 * aux_loss
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if step % 50 == 0:
        print(f"  Adım {step}/200: loss = {total_loss/len(teacher_data):.4f}")

# Inference temperature'ı düşür (keskin seçim)
router.current_temperature = 0.5
router = router.to(dtype=router_dtype)

# ═══════════════════════════════════════════════════════════════
# KAYDET
# ═══════════════════════════════════════════════════════════════
# Router'ı güncelle
router_data["router"] = router
torch.save(router_data, os.path.join(SAVE_DIR, "router.pt"))

# Kalibrasyon verilerini de kaydet (debug için)
calibration_info = {
    "combo_kl_scores": combo_kl_scores,
    "best_combo": best_combo,
    "best_kl": best_kl,
    "target_dist": target_dist,
    "sorted_combos": sorted_combos,
}
torch.save(calibration_info, CALIBRATION_FILE)

print(f"\n✅ Router güncellendi: {SAVE_DIR}/router.pt")
print(f"✅ Kalibrasyon verisi: {CALIBRATION_FILE}")
print(f"\n🎯 Artık top_k={TOP_K} ile çalıştırabilirsin:")
print(f"   Sadece bloklar {best_combo} yüklenecek")
print(f"   Disk I/O: {num_blocks}×1.8 GB → {TOP_K}×1.8 GB ({TOP_K*1.8:.1f} GB)")
print(f"   Hız: ~{num_blocks/TOP_K:.1f}x daha hızlı")
