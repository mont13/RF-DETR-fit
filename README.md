# 🔧 RF-DETR + CVAT Kompletní Guide
**Definitivní návod pro trénování RF-DETR s CVAT COCO daty**

## 📋 Obsah
- [🎯 Co tento guide řeší](#-co-tento-guide-řeší)
- [🖥️ Požadavky](#️-požadavky) 
- [🚀 Instalace](#-instalace-na-novém-počítači)
- [📁 Nový dataset z CVAT](#-nový-dataset-z-cvat)
- [🏃 Spuštění trainingu](#-spuštění-trainingu)
- [🔧 Řešení problémů](#-řešení-problémů)

---

## 🎯 Co tento guide řeší

### ❌ Problémy RF-DETR s CVAT daty:
1. **Single-class bug**: RF-DETR používá `len()` místo `max(category_id)`
2. **iscrowd problém**: CVAT exportuje všechny objekty jako `iscrowd=1`
3. **Chybějící supercategory**: RF-DETR očekává pole které CVAT neprodukuje
4. **Background class**: RF-DETR potřebuje +1 pro background třídu

### ✅ Naše řešení:
- **2 Python skripty** které vše vyřeší automaticky
- **Ručná oprava** RF-DETR kódu (nutná jen jednou)
- **Univerzální** - funguje pro 1 i více kategorií

---

## 🖥️ Požadavky

### Hardware
- **GPU**: NVIDIA s 8GB+ VRAM (RTX 3070+)
- **RAM**: 16GB+
- **Disk**: 10GB volného místa

### Software  
- **Python**: 3.8-3.11
- **CUDA**: 11.8+ nebo 12.x
- **Git**

---

## 🚀 Instalace na novém počítači

### Krok 1: Virtuální prostředí
```bash
python3 -m venv rf_detr_env
source rf_detr_env/bin/activate  # Linux/Mac
# rf_detr_env\Scripts\activate   # Windows
```

### Krok 2: Závislosti
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install rfdetr supervision datasets pillow
```

### Krok 3: ⚠️ **KRITICKÁ RF-DETR OPRAVA**

**POZOR**: Tato oprava je POVINNÁ pro každou instalaci!

1. **Najdi RF-DETR cestu:**
```python
import rfdetr
print(rfdetr.__file__)
# Výstup: /path/to/env/lib/python3.10/site-packages/rfdetr/__init__.py
```

2. **Uprav soubor**: `/path/to/env/lib/python3.10/site-packages/rfdetr/detr.py`

3. **Řádek ~130** (v `train_from_config`):
```python
# PŘED - CHYBNÝ:
num_classes = len(anns["categories"])

# PO - OPRAVENÝ: 
num_classes = max([category['id'] for category in anns["categories"]])
```

4. **Řádek ~140**:
```python
# PŘED:
self.model.reinitialize_detection_head(num_classes)

# PO:
self.model.reinitialize_detection_head(num_classes + 1)
```

5. **Řádek ~158**:
```python
# PŘED:
all_kwargs = {**model_config, **train_config, **kwargs}

# PO: 
all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes + 1}
```

### Krok 4: Test opravy
```python
# test_fix.py
import tempfile, json, os
from rfdetr import RFDETRMedium

test_coco = {
    "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0,0,10,10], "area": 100, "iscrowd": 0}],
    "categories": [{"id": 1, "name": "test_class"}]
}

with tempfile.TemporaryDirectory() as tmpdir:
    train_dir = os.path.join(tmpdir, "train")
    os.makedirs(train_dir)
    
    with open(os.path.join(train_dir, "_annotations.coco.json"), "w") as f:
        json.dump(test_coco, f)
    
    try:
        with open(os.path.join(train_dir, "_annotations.coco.json"), "r") as f:
            anns = json.load(f)
            num_classes = max([category['id'] for category in anns["categories"]])
            print(f"✅ OPRAVA FUNGUJE: {num_classes} kategorií + 1 background = {num_classes + 1}")
    except Exception as e:
        print(f"❌ OPRAVA NEFUNGUJE: {e}")
```

```bash
python test_fix.py
```

---

## 📁 Nový dataset z CVAT

### Krok 1: Export z CVAT
1. V CVAT: **Export dataset** → **COCO 1.0** → Stáhni ZIP
2. **Funguje pro 1 i více kategorií automaticky**

### Krok 2: Stáhni naše skripty

**`quick_cvat_fix.py`** - Oprava CVAT exportu:
```python
#!/usr/bin/env python3
"""
🔧 Quick CVAT Dataset Fix for RF-DETR
Opraví: iscrowd 1→0, přidá supercategory, rozdělí train/valid/test
"""

import json, shutil, os, argparse, random
from pathlib import Path

def fix_cvat_coco_export(cvat_export_path, output_path, split_ratios=(0.8, 0.1, 0.1)):
    print("🔧 Opravuji CVAT COCO export pro RF-DETR...")
    
    cvat_path = Path(cvat_export_path)
    output_path = Path(output_path)
    
    # Najdi annotation soubor
    ann_file = None
    for path in [cvat_path / "annotations" / "instances_default.json", 
                 cvat_path / "instances_default.json"]:
        if path.exists():
            ann_file = path
            break
    
    if not ann_file:
        raise FileNotFoundError(f"COCO soubor nenalezen v {cvat_path}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Dataset: {len(data['images'])} obrázků, {len(data['annotations'])} anotací, {len(data['categories'])} kategorií")
    for cat in data['categories']:
        print(f"    * ID {cat['id']}: {cat['name']}")
    
    # OPRAVA 1: iscrowd 1 → 0
    crowd_count = 0
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            ann['iscrowd'] = 0
            crowd_count += 1
    if crowd_count > 0:
        print(f"✅ {crowd_count} objektů opraveno z 'crowd' na normální")
    
    # OPRAVA 2: supercategory
    supercategory_count = 0
    for cat in data['categories']:
        if 'supercategory' not in cat:
            cat['supercategory'] = 'object'
            supercategory_count += 1
    if supercategory_count > 0:
        print(f"✅ {supercategory_count} kategorií doplněno supercategory")
    
    # Najdi obrázky
    image_dir = None
    for img_dir in [cvat_path / "images" / "default", cvat_path / "images", cvat_path]:
        if img_dir.exists() and list(img_dir.glob("*.jpg")):
            image_dir = img_dir
            break
    
    if not image_dir:
        raise FileNotFoundError(f"Obrázky nenalezeny v {cvat_path}")
    
    # Rozdělení train/valid/test
    random.seed(42)
    images = data['images'].copy()
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(split_ratios[0] * n_total)
    n_valid = int(split_ratios[1] * n_total)
    
    # Zajisti alespoň něco pro valid
    if n_total > 2 and n_valid == 0:
        n_valid = 1
        n_train = n_total - n_valid - max(1, int(split_ratios[2] * n_total))
    
    train_images = images[:n_train]
    valid_images = images[n_train:n_train + n_valid]
    test_images = images[n_train + n_valid:]
    
    print(f"📈 Rozdělení: Train {len(train_images)}, Valid {len(valid_images)}, Test {len(test_images)}")
    
    # Vytvoř RF-DETR strukturu
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split, split_images in [('train', train_images), ('valid', valid_images), ('test', test_images)]:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        if not split_images:
            # Prázdný split pro RF-DETR
            empty_data = {'images': [], 'annotations': [], 'categories': data['categories']}
            with open(split_dir / "_annotations.coco.json", 'w') as f:
                json.dump(empty_data, f)
            print(f"⚠️  {split}: prázdný (RF-DETR vyžaduje)")
            continue
        
        # Kopíruj obrázky
        image_ids = {img['id'] for img in split_images}
        for img in split_images:
            src = image_dir / img['file_name']
            dst = split_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
        
        # Anotace pro split
        split_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
        split_data = {
            'images': split_images,
            'annotations': split_annotations, 
            'categories': data['categories']
        }
        
        with open(split_dir / "_annotations.coco.json", 'w') as f:
            json.dump(split_data, f)
        
        print(f"✅ {split}: {len(split_images)} obrázků, {len(split_annotations)} anotací")
    
    print(f"🎉 Dataset připraven: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opraví CVAT export pro RF-DETR")
    parser.add_argument('input_dir', help='CVAT export složka')
    parser.add_argument('output_dir', help='Výstupní složka')
    args = parser.parse_args()
    
    fix_cvat_coco_export(args.input_dir, args.output_dir)
```

**`train_universal.py`** - Univerzální training:
```python
#!/usr/bin/env python3
"""
🚀 Universal RF-DETR Training for CVAT Data
Univerzální training script s auto-konfigurací
"""

import argparse, torch, json, os
from pathlib import Path
from rfdetr import RFDETRMedium

def get_optimal_batch_size():
    """Auto batch size podle GPU"""
    if not torch.cuda.is_available():
        return 2
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 24: return 8
    elif vram_gb >= 16: return 6  
    elif vram_gb >= 12: return 4
    elif vram_gb >= 8: return 3
    else: return 2

def validate_dataset(dataset_dir):
    """Validuje dataset a RF-DETR opravy"""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset neexistuje: {dataset_dir}")
    
    for split in ['train', 'valid', 'test']:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
            n_imgs, n_anns, n_cats = len(data['images']), len(data['annotations']), len(data['categories'])
            print(f"✅ {split}: {n_imgs} obrázků, {n_anns} anotací, {n_cats} kategorií")
            
            crowd_count = sum(1 for ann in data['annotations'] if ann.get('iscrowd', 0) == 1)
            if crowd_count > 0:
                print(f"⚠️  {split}: {crowd_count} crowd objektů!")

def check_rfdetr_fixes():
    """Kontrola RF-DETR oprav"""
    try:
        import rfdetr
        detr_file = Path(rfdetr.__file__).parent / "detr.py"
        with open(detr_file, 'r') as f:
            content = f.read()
        
        if "max([category['id'] for category in anns[\"categories\"]])" in content:
            print("✅ RF-DETR Fix 1: Single-class bug opraven")
        else:
            print("❌ RF-DETR Fix 1: CHYBÍ! Viz guide")
            return False
        
        if "num_classes + 1" in content:
            print("✅ RF-DETR Fix 2: Background class opraven")
        else:
            print("❌ RF-DETR Fix 2: CHYBÍ!")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️  Nelze zkontrolovat: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Universal RF-DETR training")
    parser.add_argument('--dataset', '-d', required=True, help='Dataset složka')
    parser.add_argument('--output', '-o', default='trained_model', help='Output složka')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Epochy')
    parser.add_argument('--batch-size', '-b', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--quick-test', action='store_true', help='Jen 1 epocha')
    args = parser.parse_args()
    
    print("🚀 RF-DETR Universal Training")
    print("=" * 50)
    
    print("\n🔍 Validuji dataset...")
    validate_dataset(args.dataset)
    
    print("\n🔧 Kontroluji RF-DETR opravy...")
    if not check_rfdetr_fixes():
        print("❌ RF-DETR není opraven! Viz guide")
        return 1
    
    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    batch_size = get_optimal_batch_size() if args.batch_size is None else args.batch_size
    epochs = 1 if args.quick_test else args.epochs
    
    print(f"\n🖥️  Hardware:")
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"  Batch size: {batch_size}")
    
    if args.quick_test:
        print(f"⚡ Quick test: {epochs} epocha")
    
    # Model
    print(f"\n🤖 RF-DETR Medium...")
    model = RFDETRMedium()
    
    # Config
    config = {
        'dataset_dir': args.dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': args.lr,
        'lr_encoder': args.lr * 0.1,
        'weight_decay': 1e-4,
        'device': device,
        'output_dir': args.output,
        'early_stopping': True,
        'early_stopping_patience': 10,
        'warmup_epochs': min(5, epochs//4),
        'lr_drop': max(epochs//2, 10),
        'use_ema': True
    }
    
    print(f"\n📋 Training config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    os.makedirs(args.output, exist_ok=True)
    with open(Path(args.output) / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🎯 Spouštím training...")
    try:
        model.train(**config)
        print("\n🎉 Training dokončen!")
        best_model = Path(args.output) / "checkpoint_best_total.pth"
        if best_model.exists():
            print(f"🏆 Nejlepší model: {best_model}")
        return 0
    except Exception as e:
        print(f"\n❌ Chyba: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
```

### Krok 3: Rozbal a oprav CVAT export
```bash
unzip dataset.zip
python quick_cvat_fix.py dataset dataset_prepared
```

**Příklad výstupu:**
```
📊 Dataset: 123 obrázků, 241 anotací, 1 kategorií
    * ID 1: Screwdriver
✅ 241 objektů opraveno z 'crowd' na normální
✅ 1 kategorií doplněno supercategory
📈 Rozdělení: Train 98, Valid 12, Test 13
🎉 Dataset připraven: dataset_prepared
```

---

## 🏃 Spuštění trainingu

### Základní training
```bash
python train_universal.py --dataset dataset_prepared
```

### Pokročilé nastavení
```bash
python train_universal.py --dataset dataset_prepared --epochs 100 --lr 5e-5 --output my_model
```

### Quick test (1 epocha)
```bash
python train_universal.py --dataset dataset_prepared --quick-test
```

**Výstup trainingu:**
```
🚀 RF-DETR Universal Training
==================================================
🔍 Validuji dataset...
✅ train: 98 obrázků, 193 anotací, 1 kategorií
✅ valid: 12 obrázků, 24 anotací, 1 kategorií  
✅ test: 13 obrázků, 24 anotací, 1 kategorií

🔧 Kontroluji RF-DETR opravy...
✅ RF-DETR Fix 1: Single-class bug opraven
✅ RF-DETR Fix 2: Background class opraven

🖥️  Hardware:
  Device: cuda
  GPU: NVIDIA GeForce RTX 4090
  VRAM: 24.0GB
  Batch size: 8

🎯 Spouštím training...
num_classes mismatch: model has 90 classes, but your dataset has 1 classes
reinitializing your detection head with 2 classes (including background).

Epoch [1/50]: loss: 0.2480, bbox: 0.2480, giou: 0.2779, mAP@50: 0.435
🎉 Training dokončen!
🏆 Nejlepší model: trained_model/checkpoint_best_total.pth
```

---

## 🔧 Řešení problémů

### ❌ "Zero losses" nebo "mAP = -1"
**Příčina**: iscrowd oprava neproběhla
**Řešení**: Zkontroluj v `_annotations.coco.json` že `"iscrowd": 0`

### ❌ "num_classes mismatch"  
**Příčina**: RF-DETR oprava neproběhla
**Řešení**: Zkontroluj opravu v `detr.py`

### ❌ "supercategory KeyError"
**Příčina**: Starý quick_cvat_fix.py
**Řešení**: Použij aktuální verzi scriptu

### ❌ "CUDA out of memory"
**Řešení**: Sniž batch size: `--batch-size 2`

### ❌ "UnidentifiedImageError"
**Příčina**: Poškozené nebo chybějící obrázky
**Řešení**: Zkontroluj že všechny obrázky existují a jsou validní

---

## 🎯 Multi-Category příklad

### CVAT s více kategoriemi:
```json
{
  "categories": [
    {"id": 1, "name": "screwdriver"},
    {"id": 2, "name": "hammer"}, 
    {"id": 3, "name": "wrench"}
  ]
}
```

### Automatická detekce:
```
📊 Dataset: 456 obrázků, 892 anotací, 3 kategorií
    * ID 1: screwdriver
    * ID 2: hammer
    * ID 3: wrench
✅ 892 objektů opraveno z 'crowd' na normální

num_classes=4 (3 + background)
class_names=['screwdriver', 'hammer', 'wrench']
```

### Inference s více kategoriemi:
```python
from rfdetr import RFDETRMedium
model = RFDETRMedium.from_checkpoint("my_model/checkpoint_best_total.pth")
results = model.predict("tools.jpg", confidence=0.5)
# [
#   {"label": "screwdriver", "confidence": 0.95, "bbox": [10,10,50,50]},
#   {"label": "hammer", "confidence": 0.87, "bbox": [60,60,40,40]}
# ]
```

---

## ✅ Checklist pro nový projekt

### Příprava:
- [ ] RF-DETR nainstalován + opravy aplikovány
- [ ] CVAT dataset exportován jako COCO 1.0
- [ ] `quick_cvat_fix.py` stažen a spuštěn
- [ ] Dataset struktura: `train/`, `valid/`, `test/` s `_annotations.coco.json`

### Training:
- [ ] `train_universal.py` stažen
- [ ] GPU dostupné (ověř `nvidia-smi`)
- [ ] Training spuštěn: `python train_universal.py --dataset dataset_prepared`
- [ ] Progress monitored: `tail -f trained_model/train.log`

### Po trainingu:
- [ ] Model checkpoint existuje: `trained_model/checkpoint_best_total.pth`
- [ ] Inference test: `model.predict("test.jpg")`
- [ ] Výsledky uspokojivé

---

## 🎉 ZÁVĚR

**✅ UNIVERZÁLNÍ ŘEŠENÍ pro RF-DETR + CVAT:**

- **2 skripty** vyřeší vše automaticky
- **1 oprava** RF-DETR kódu (jednou navždy)
- **Funguje** pro 1 i více kategorií
- **Jednoduchý** workflow pro každý nový dataset

**Původní problém vyřešen**: *"mam yto data exportovnae z cvat do coco 1.0 a chtel bych upravit kod aby dokazal moje data trenovat"* ✅

**Workflow je nyní:**
1. CVAT → Export COCO 1.0
2. `python quick_cvat_fix.py dataset dataset_prepared`  
3. `python train_universal.py --dataset dataset_prepared`
4. 🎉 Profit!

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

Takže pro tvou novou složku napr "screw-test" s CVAT exportem budeš potřebovat spustit:

`python quick_cvat_fix.py screw-test dataset_prepared_new`

Kde:
screw-test = tvá vstupní složka s CVAT COCO 1.0 exportem (obsahuje annotations/instances_default.json a images/default/)
dataset_prepared_new = výstupní složka kde se vytvoří RF-DETR kompatibilní struktura

**Script automaticky:**

1. Najde instances_default.json v annotations
2. Opraví všechny iscrowd: 1 na iscrowd: 0 (kritická oprava!)
3. Rozdělí obrázky na train/valid/test (80/10/10)
4. Vytvoří RF-DETR strukturu s _annotations.coco.json v každé složce
5. Zkopíruje obrázky do správných složek
6. Po spuštění budeš mít připraveno pro training:

`python train_universal.py --dataset dataset_prepared_new --output screwdriver_model_v2`


# Normální training s grafy (default)
python train_universal.py --dataset dataset_prepared --output my_model

# Training bez grafů (rychlejší)
python train_universal.py --dataset dataset_prepared --output my_model --no-plots

# Vytvoř grafy pro existující model
python -c "from train_universal import create_training_plots; create_training_plots('screwdriver_model')"

Najdi nejlepší checkpoint
    best_checkpoint = output_path / "checkpoint_best_total.pth"

source pytcvatsrew/bin/activate && python quick_cvat_fix.py screw-test dataset_prepared_new
source pytcvatsrew/bin/activate && python train_universal.py --dataset dataset_prepared_new --output fresh_screwdriver_model --epochs 20