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
import time
from pathlib import Path
from rfdetr.detr import RFDETRMedium


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
            
            found_splits.append(split)
            print(f"✅ {split}: {n_images} obrázků, {n_annotations} anotací, {n_categories} kategorií")
            
            # Zobraz kategorie a jejich statistiky
            print(f"📋 Kategorie v {split}:")
            for cat in data['categories']:
                count = sum(1 for ann in data['annotations'] if ann['category_id'] == cat['id'])
                print(f"  ID {cat['id']}: {cat['name']} - {count} objektů")
            
            # Zkontroluj iscrowd problém
            crowd_count = sum(1 for ann in data['annotations'] if ann.get('iscrowd', 0) == 1)
            if crowd_count > 0:
                print(f"⚠️  {split}: {crowd_count} 'crowd' objektů - budou ignorovány!")
        else:
            if split in required_splits:
                raise FileNotFoundError(f"Chybí {split} split: {split_dir}")
            else:
                print(f"ℹ️  {split}: není k dispozici")
    
    if not found_splits:
        raise ValueError("Žádné platné splits nenalezeny!")
    
    return found_splits


def check_rfdetr_fixes():
    """Zkontroluje, zda jsou aplikovány RF-DETR opravy"""
    try:
        import rfdetr
        detr_file = Path(rfdetr.__file__).parent / "detr.py"
        
        if not detr_file.exists():
            print("⚠️  Nelze najít RF-DETR detr.py soubor")
            return False
        
        with open(detr_file, 'r') as f:
            content = f.read()
        
        # Zkontroluj fix 1: max(category_id) místo len()
        if "max([category['id'] for category in anns[\"categories\"]])" in content:
            print("✅ RF-DETR Fix 1: Single-class category bug opraven")
        else:
            print("❌ RF-DETR Fix 1: CHYBÍ single-class oprava!")
            print("   Musíš ručně opravit detr.py - viz návod v SETUP_GUIDE.md")
            return False
        
        # Zkontroluj fix 2: +1 pro background
        if "num_classes + 1" in content:
            print("✅ RF-DETR Fix 2: Background class opraven")
        else:
            print("❌ RF-DETR Fix 2: CHYBÍ background class oprava!")
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️  Nelze zkontrolovat RF-DETR opravy: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Universal RF-DETR training script for CVAT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:
  python train_universal.py --dataset dataset_prepared --output my_model
  python train_universal.py --dataset data --epochs 100 --batch-size 4
  python train_universal.py --dataset data --output model --lr 5e-5 --device cpu

Tip: Pro první spuštění použij jen:
  python train_universal.py --dataset dataset_prepared
        """
    )
    
    # Dataset a output
    parser.add_argument('--dataset', '-d', required=True, help='Cesta k připravenému datasetu')
    parser.add_argument('--output', '-o', default='trained_model', help='Výstupní složka pro model (default: trained_model)')
    
    # Training parametry
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Počet epoch (default: 20)')
    parser.add_argument('--batch-size', '-b', type=int, default=None, help='Batch size (default: auto)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
    parser.add_argument('--lr-encoder', type=float, default=1e-6, help='Encoder LR (default: 1e-6)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    
    # Hardware
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='Device (default: auto)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')
    
    # Advanced
    parser.add_argument('--no-early-stopping', action='store_true', help='Vypni early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
    parser.add_argument('--warmup-epochs', type=int, default=2, help='Warmup epochs (default: 2)')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    
    # Data augmentation pro malý dataset
    parser.add_argument('--augment', action='store_true', help='Zapni silnější augmentaci pro malý dataset')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha (0.2 doporučeno pro malé datasety)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    
    # Debug
    parser.add_argument('--quick-test', action='store_true', help='Rychlý test - jen 1 epocha')
    parser.add_argument('--skip-validation', action='store_true', help='Přeskoč validation checks')
    
    args = parser.parse_args()
    
    print("🚀 RF-DETR Universal Training Script")
    print("=" * 50)
    
    # Validace
    if not args.skip_validation:
        print("\n🔍 Kontroluji dataset...")
        try:
            found_splits = validate_dataset(args.dataset)
        except Exception as e:
            print(f"❌ Dataset chyba: {e}")
            return 1
        
        print("\n🔧 Kontroluji RF-DETR opravy...")
        if not check_rfdetr_fixes():
            print("❌ RF-DETR není správně opraven! Viz SETUP_GUIDE.md")
            return 1
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n🖥️  Hardware:")
    print(f"  Device: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
    
    # Batch size
    if args.batch_size is None:
        batch_size = get_optimal_batch_size()
        print(f"  Auto batch size: {batch_size}")
    else:
        batch_size = args.batch_size
        print(f"  Manual batch size: {batch_size}")
    
    # Effective batch size
    effective_batch = batch_size * args.grad_accum
    print(f"  Effective batch size: {effective_batch}")
    
    # Quick test mode
    if args.quick_test:
        epochs = 1
        print(f"\n⚡ Quick test mode: {epochs} epocha")
    else:
        epochs = args.epochs
    
    # Model setup
    print(f"\n🤖 Inicializuji RF-DETR Medium...")
    model = RFDETRMedium()
    
    # Training config
    config = {
        'dataset_dir': args.dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'grad_accum_steps': args.grad_accum,
        'lr': args.lr,
        'lr_encoder': args.lr_encoder,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'device': device,
        'output_dir': args.output,
        'early_stopping': not args.no_early_stopping,
        'early_stopping_patience': args.patience,
        'early_stopping_min_delta': 0.001,
        'warmup_epochs': args.warmup_epochs,
        'lr_drop': max(10, epochs // 3),  # LR drop dříve pro malý dataset
        'clip_max_norm': 0.1,
        'use_ema': True,
        'ema_decay': 0.995,
        # Augmentace pro malý dataset
        'use_augment': args.augment,
        'mixup_alpha': args.mixup,
        'dropout_rate': args.dropout,
    }
    
    print(f"\n📋 Training konfigurace:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Výstupní složka
    os.makedirs(args.output, exist_ok=True)
    
    # Ulož konfiguraci
    config_file = Path(args.output) / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Konfigurace uložena: {config_file}")
    
    print(f"\n🎯 Spouštím training...")
    print(f"📁 Output: {args.output}")
    print(f"📊 Progress: tail -f {args.output}/train.log")
    print("-" * 50)
    
    try:
        # Training
        model.train(**config)
        
        print("\n" + "=" * 50)
        print("🎉 Training dokončen!")
        print(f"📁 Model uložen v: {args.output}")
        
        # Najdi nejlepší checkpoint
        checkpoint_best = Path(args.output) / "checkpoint_best_total.pth"
        if checkpoint_best.exists():
            print(f"🏆 Nejlepší model: {checkpoint_best}")
        
        print(f"\n📈 Pro inference použij:")
        print(f"  from rfdetr import RFDETRMedium")
        print(f"  model = RFDETRMedium.from_checkpoint('{checkpoint_best}')")
        print(f"  results = model.predict('image.jpg')")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training přerušen uživatelem")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training chyba: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
