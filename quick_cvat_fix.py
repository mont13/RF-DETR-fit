#!/usr/bin/env python3
"""
🔧 Quick CVAT Dataset Fix for RF-DETR
Automaticky opraví CVAT COCO export pro RF-DETR training

Použití:
    # Rozdělení na train/valid/test (80/10/10)
    import random
    random.seed(42)
    
    images = data['images'].copy()
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(split_ratios[0] * n_total)
    n_valid = int(split_ratios[1] * n_total)
    
    # Zajisti alespoň 1 obrázek pro valid pokud je celkem víc než 2
    if n_total > 2 and n_valid == 0:
        n_valid = 1
        n_train = n_total - n_valid - max(1, int(split_ratios[2] * n_total))
    
    train_images = images[:n_train]
    valid_images = images[n_train:n_train + n_valid]
    test_images = images[n_train + n_valid:]uick_cvat_fix.py dataset_export_folder output_folder

Opravy:
1. iscrowd: 1 → 0 (všechny objekty jako normální, ne crowd)
2. Rozdělí dataset na train/valid/test (80/10/10)
3. Vytvoří RF-DETR kompatibilní strukturu
"""

import json
import shutil
import os
import argparse
from pathlib import Path
import random


def fix_cvat_coco_export(cvat_export_path, output_path, split_ratios=(0.8, 0.1, 0.1)):
    """
    Opraví CVAT COCO export pro RF-DETR training
    
    Args:
        cvat_export_path: Cesta k rozbaleném CVAT exportu
        output_path: Cesta kde vytvořit RF-DETR kompatibilní dataset
        split_ratios: Tuple (train, valid, test) ratios
    """
    print("🔧 Opravuji CVAT COCO export pro RF-DETR...")
    
    cvat_path = Path(cvat_export_path)
    output_path = Path(output_path)
    
    # Najdi annotation soubor
    ann_file = None
    possible_paths = [
        cvat_path / "annotations" / "instances_default.json",
        cvat_path / "instances_default.json",
        cvat_path / "_annotations.coco.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            ann_file = path
            break
    
    if not ann_file:
        raise FileNotFoundError(f"Nenalezen COCO annotation soubor v {cvat_path}")
    
    print(f"📁 Načítám anotace z: {ann_file}")
    
    # Načti anotace
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Dataset obsahuje:")
    print(f"  - {len(data['images'])} obrázků")
    print(f"  - {len(data['annotations'])} anotací")
    print(f"  - {len(data['categories'])} kategorií")
    
    # Vypiš kategorie
    for cat in data['categories']:
        print(f"    * ID {cat['id']}: {cat['name']}")
    
    # ⚠️ KRITICKÁ OPRAVA: iscrowd 1 → 0
    crowd_count = 0
    segmentation_removed = 0
    bbox_fixed = 0
    
    for ann in data['annotations']:
        # Fix iscrowd
        if ann.get('iscrowd', 0) == 1:
            ann['iscrowd'] = 0
            crowd_count += 1
        
        # 🔧 NOVÁ OPRAVA: Odstranit RLE segmentation pro pure detection
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                # Odstranit RLE segmentation
                del ann['segmentation']
                segmentation_removed += 1
        
        # 🔧 NOVÁ OPRAVA: Zajistit správné typy pro bbox
        if 'bbox' in ann:
            bbox = ann['bbox']
            # Převést na int pokud jsou hodnoty celočíselné
            ann['bbox'] = [float(x) for x in bbox]  # RF-DETR chce float
            bbox_fixed += 1
        
        # 🔧 NOVÁ OPRAVA: Zajistit area jako float
        if 'area' in ann:
            ann['area'] = float(ann['area'])

    if crowd_count > 0:
        print(f"✅ OPRAVENO: {crowd_count} objektů změněno z 'crowd' na normální")
    else:
        print("ℹ️  Žádné crowd objekty nenalezeny")
    
    if segmentation_removed > 0:
        print(f"✅ OPRAVENO: {segmentation_removed} RLE segmentations odstraněno (pure detection)")
    
    if bbox_fixed > 0:
        print(f"✅ OPRAVENO: {bbox_fixed} bbox formátů upraveno na float")    # ⚠️ OPRAVA 2: Přidej chybějící supercategory pro RF-DETR
    supercategory_count = 0
    for cat in data['categories']:
        if 'supercategory' not in cat:
            cat['supercategory'] = 'object'
            supercategory_count += 1
    
    if supercategory_count > 0:
        print(f"✅ OPRAVENO: {supercategory_count} kategorií doplněno supercategory")
    
    # Kontrola bounding boxů
    bbox_issues = 0
    for ann in data['annotations']:
        bbox = ann['bbox']
        if len(bbox) != 4 or any(v < 0 for v in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            bbox_issues += 1
    
    if bbox_issues > 0:
        print(f"⚠️  Nalezeno {bbox_issues} problematických bounding boxů")
    
    # Najdi obrázky
    image_dir = None
    possible_image_dirs = [
        cvat_path / "images" / "default",
        cvat_path / "images", 
        cvat_path / "default",
        cvat_path
    ]
    
    for img_dir in possible_image_dirs:
        if img_dir.exists() and any(img_dir.glob("*.jpg")) or any(img_dir.glob("*.png")):
            image_dir = img_dir
            break
    
    if not image_dir:
        raise FileNotFoundError(f"Nenalezena složka s obrázky v {cvat_path}")
    
    print(f"🖼️  Obrázky nalezeny v: {image_dir}")
    
    # Kontrola, že všechny obrázky existují
    missing_images = []
    for img in data['images']:
        img_path = image_dir / img['file_name']
        if not img_path.exists():
            missing_images.append(img['file_name'])
    
    if missing_images:
        print(f"⚠️  Chybí {len(missing_images)} obrázků:")
        for img in missing_images[:5]:  # Ukázat jen prvních 5
            print(f"    - {img}")
        if len(missing_images) > 5:
            print(f"    ... a dalších {len(missing_images) - 5}")
    
    # Rozdělení na train/valid/test
    random.seed(42)
    images = data['images'].copy()
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(split_ratios[0] * n_total)
    n_valid = int(split_ratios[1] * n_total)
    
    train_images = images[:n_train]
    valid_images = images[n_train:n_train + n_valid]
    test_images = images[n_train + n_valid:]
    
    print(f"\n📈 Rozdělení datasetu:")
    print(f"  - Train: {len(train_images)} obrázků ({len(train_images)/n_total*100:.1f}%)")
    print(f"  - Valid: {len(valid_images)} obrázků ({len(valid_images)/n_total*100:.1f}%)")
    print(f"  - Test:  {len(test_images)} obrázků ({len(test_images)/n_total*100:.1f}%)")
    
    # Připrav categories s opravami pro všechny splity
    categories_fixed = []
    for cat in data['categories']:
        cat_fixed = cat.copy()
        if 'supercategory' not in cat_fixed:
            cat_fixed['supercategory'] = 'object'  # RF-DETR očekává supercategory
        categories_fixed.append(cat_fixed)
    
    # Vytvoř RF-DETR strukturu
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split, split_images in [('train', train_images), ('valid', valid_images), ('test', test_images)]:
        if not split_images:
            # Pro prázdné splity vytvoř alespoň prázdnou strukturu (RF-DETR potřebuje všechny)
            split_dir = output_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            empty_data = {
                'info': {
                    'description': f'CVAT dataset - {split} split (empty)',
                    'version': '1.0',
                    'year': 2024,
                    'contributor': 'CVAT',
                    'date_created': '2024-01-01'
                },
                'images': [],
                'annotations': [],
                'categories': categories_fixed
            }
            
            ann_output = split_dir / "_annotations.coco.json"
            with open(ann_output, 'w') as f:
                json.dump(empty_data, f, indent=2)
            
            print(f"⚠️  {split}: prázdný split vytvořen (RF-DETR vyžaduje)")
            continue
            
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Kopíruj obrázky
        image_ids = {img['id'] for img in split_images}
        copied_count = 0
        
        for img in split_images:
            src = image_dir / img['file_name']
            dst = split_dir / img['file_name']
            
            if src.exists():
                shutil.copy2(src, dst)
                copied_count += 1
            else:
                print(f"⚠️  Chybí obrázek: {img['file_name']}")
        
        # Vytvoř anotace pro split
        split_annotations = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
        
        split_data = {
            'info': {
                'description': f'CVAT dataset - {split} split',
                'version': '1.0',
                'year': 2024,
                'contributor': 'CVAT',
                'date_created': '2024-01-01'
            },
            'images': split_images,
            'annotations': split_annotations,
            'categories': categories_fixed
        }
        
        # Ulož jako _annotations.coco.json (RF-DETR formát)
        ann_output = split_dir / "_annotations.coco.json"
        with open(ann_output, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✅ {split}: {copied_count} obrázků, {len(split_annotations)} anotací → {ann_output}")
    
    print(f"\n🎉 Dataset připraven v: {output_path}")
    print(f"\n📋 Další kroky:")
    print(f"1. Zkontroluj strukturu: ls -la {output_path}")
    print(f"2. Spusť training: python train_screwdriver.py")
    print(f"3. Sleduj progress: tail -f output_dir/train.log")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Opraví CVAT COCO export pro RF-DETR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:
  python quick_cvat_fix.py dataset_export dataset_prepared
  python quick_cvat_fix.py /path/to/cvat/export /path/to/prepared/dataset

Tento script:
1. Najde COCO annotation soubor
2. Opraví iscrowd: 1 → 0 (critical fix)
3. Rozdělí na train/valid/test (80/10/10)
4. Vytvoří RF-DETR kompatibilní strukturu
        """
    )
    
    parser.add_argument('input_dir', help='Cesta k CVAT export složce')
    parser.add_argument('output_dir', help='Cesta kde vytvořit připravený dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Podíl train dat (default: 0.8)')
    parser.add_argument('--valid-ratio', type=float, default=0.1, help='Podíl validation dat (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Podíl test dat (default: 0.1)')
    
    args = parser.parse_args()
    
    # Kontrola ratios
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Chyba: Součet ratios je {total_ratio}, musí být 1.0")
        return 1
    
    try:
        fix_cvat_coco_export(
            args.input_dir, 
            args.output_dir,
            (args.train_ratio, args.valid_ratio, args.test_ratio)
        )
        return 0
    except Exception as e:
        print(f"❌ Chyba: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
