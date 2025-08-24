#!/usr/bin/env python3
"""
🚀 Universal RF-DETR Training Script for CVAT Data
Univerzální script pro trénování RF-DETR s CVAT daty

Použití:
python train_universal.py --dataset dataset_prepared --output output_model

Funkce:
- Automatická GPU detekce a optimalizace
- Adaptive batch size podle VRAM
- Progress monitoring
- Early stopping
- Model checkpoints
"""

import argparse
import torch
import json
import os
from pathlib import Path
from rfdetr.detr import RFDETRMedium
import time


def get_optimal_batch_size():
    """Automaticky detekuje optimální batch size podle VRAM"""
    if not torch.cuda.is_available():
        return 2
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if vram_gb >= 24:    # RTX 4090, A100
        return 8
    elif vram_gb >= 16:  # RTX 4080, RTX 3080 Ti
        return 6  
    elif vram_gb >= 12:  # RTX 4070 Ti, RTX 3080
        return 4
    elif vram_gb >= 8:   # RTX 4060 Ti, RTX 3070
        return 3
    else:               # RTX 3060, GTX 1660
        return 2


def validate_dataset(dataset_dir):
    """Validuje dataset strukturu"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset složka neexistuje: {dataset_dir}")
    
    required_splits = ['train']
    optional_splits = ['valid', 'test']
    
    found_splits = []
    for split in required_splits + optional_splits:
        split_dir = dataset_path / split
        ann_file = split_dir / "_annotations.coco.json"
        
        if split_dir.exists() and ann_file.exists():
            # Zkontroluj anotace
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            n_images = len(data['images'])
            n_annotations = len(data['annotations'])
            n_categories = len(data['categories'])
            
            print(f"✅ {split}: {n_images} obrázků, {n_annotations} anotací, {n_categories} tříd")
            found_splits.append(split)
        else:
            if split in required_splits:
                raise FileNotFoundError(f"Požadovaný split '{split}' nenalezen")
            else:
                print(f"⚠️  Volitelný split '{split}' nenalezen")
    
    return found_splits


def check_rf_detr_fixes():
    """Zkontroluje, zda jsou aplikovány RF-DETR opravy"""
    try:
        # Zkus najít RF-DETR installation
        import rfdetr
        rf_detr_path = Path(rfdetr.__file__).parent
        detr_file = rf_detr_path / "detr.py"  # Opravená cesta
        
        if not detr_file.exists():
            print("⚠️  Nelze najít RF-DETR detr.py soubor")
            return False
        
        with open(detr_file, 'r') as f:
            content = f.read()
        
        # Check Fix 1: Single class category bug - hledáme nový kód
        if "max([category['id'] for category in anns[\"categories\"]])" in content:
            print("✅ RF-DETR Fix 1: Single-class category bug opraven")
        else:
            print("❌ RF-DETR Fix 1: CHYBÍ single-class oprava!")
            return False
        
        # Check Fix 2: Background class - hledáme nový kód
        if "num_classes + 1" in content:
            print("✅ RF-DETR Fix 2: Background class opraven")
        else:
            print("❌ RF-DETR Fix 2: CHYBÍ background class oprava!")
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️  Nelze zkontrolovat RF-DETR opravy: {e}")
        return False


def train_model(dataset_dir, output_dir, epochs=15, batch_size=None, no_plots=False):
    """Hlavní training funkce"""
    
    print("🚀 Spouštím RF-DETR trénink...")
    
    # 1. Validuj dataset
    print("\n📋 Validuji dataset...")
    splits = validate_dataset(dataset_dir)
    
    # 2. Zkontroluj RF-DETR opravy
    print("\n🔧 Kontroluji RF-DETR opravy...")
    if not check_rf_detr_fixes():
        print("\n❌ CHYBA: RF-DETR opravy nejsou aplikovány!")
        print("Spusť nejdříve: python quick_cvat_fix.py --apply-rf-detr-fixes")
        return False
    
    # 3. Detekuj hardware
    print(f"\n🖥️  Hardware info:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {vram_gb:.1f} GB")
        
        if batch_size is None:
            batch_size = get_optimal_batch_size()
        print(f"   Doporučený batch size: {batch_size}")
    else:
        print("   ⚠️  CUDA není k dispozici - použiju CPU")
        batch_size = 1
    
    # 4. Vytvoř output složku
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 5. Detekuj počet tříd z datasetu
    print(f"\n🔍 Detekovuji počet tříd z datasetu...")
    dataset_path = Path(dataset_dir)
    train_ann_file = dataset_path / "train" / "_annotations.coco.json"
    with open(train_ann_file, 'r') as f:
        ann_data = json.load(f)
    
    num_classes = len(ann_data['categories'])
    class_names = [c['name'] for c in ann_data['categories']]
    
    print(f"   Počet tříd: {num_classes}")
    print(f"   Třídy: {', '.join(class_names)}")
    
    # 6. Nastav model s automaticky detekovaným počtem tříd
    print(f"\n🧠 Inicializuji RF-DETR model...")
    model = RFDETRMedium(
        num_classes=num_classes,  # Automaticky detekováno
        use_pretrained=True
    )
    
    # 7. Spusť trénink
    print(f"\n🏃 Spouštím trénink na {epochs} epoch...")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Batch size: {batch_size}")
    print(f"   Třídy: {num_classes} ({', '.join(class_names)})")
    
    start_time = time.time()
    
    # Training config jako v starém kódu
    config = {
        'dataset_dir': str(dataset_dir),
        'epochs': epochs,
        'batch_size': batch_size,
        'output_dir': str(output_path),
        'save_period': 5,
        'eval': False,  # Vypni evaluaci kvůli RF-DETR bug s 'info'
        'grad_accum_steps': 4,
        'lr': 1e-4,
        'lr_encoder': 1.5e-4,
        'weight_decay': 1e-4,
        'use_ema': True,
        'ema_decay': 0.993,
        'clip_max_norm': 0.1
    }
    
    try:
        # Použij starý způsob: model.train(**config)
        model.train(**config)
        
        training_time = time.time() - start_time
        print(f"\n✅ Trénink dokončen za {training_time/60:.1f} minut!")
        
        # 7. Generuj grafy pokud je požadováno
        if not no_plots:
            print("\n📊 Generuji grafy...")
            try:
                import subprocess
                result = subprocess.run([
                    'python', 'generate_plots.py', str(output_dir)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Grafy úspěšně vygenerovány!")
                else:
                    print(f"⚠️  Chyba při generování grafů: {result.stderr}")
            except Exception as e:
                print(f"⚠️  Nepodařilo se spustit generate_plots.py: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Chyba při tréningu: {e}")
        return False
        print(f"\n❌ Chyba při tréningu: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Universal RF-DETR Training Script')
    parser.add_argument('--dataset', required=True, help='Cesta k prepared datasetu')
    parser.add_argument('--output', required=True, help='Výstupní složka pro model')
    parser.add_argument('--epochs', type=int, default=15, help='Počet epoch (default: 15)')
    parser.add_argument('--batch-size', type=int, help='Batch size (auto-detect pokud není zadán)')
    parser.add_argument('--no-plots', action='store_true', help='Negeneruj grafy po tréningu')
    
    args = parser.parse_args()
    
    success = train_model(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        no_plots=args.no_plots
    )
    
    if success:
        print("\n🎉 Trénink úspěšně dokončen!")
        if not args.no_plots:
            print(f"📈 Grafy jsou k dispozici v: {args.output}/actual_training_results.png")
    else:
        print("\n💥 Trénink selhal!")
        exit(1)


if __name__ == "__main__":
    main()
