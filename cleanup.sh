#!/bin/bash
# 🧹 RF-DETR CVAT Cleanup Script - BEZPEČNÝ
# Smaže JEN nepotřebné debug adresáře (ušetří ~3GB)

echo "🗑️  BEZPEČNÝ RF-DETR CLEANUP"
echo "============================="
echo

echo "📊 SOUČASNÝ STAV:"
du -sh */ 2>/dev/null | sort -hr
echo

echo "❌ CO SMAŽU (JEN DEBUG ADRESÁŘE):"
echo "  debug_test_output/       1.4GB - Debug výstupy"
echo "  final_coco_aware_output/ 1.0GB - Starý experiment"
echo "  final_output/            511MB - Starý experiment" 
echo "  dataset_coco_format/     2.5MB - Neúspěšná konverze"
echo

echo "✅ CO PONECHÁM (DŮLEŽITÉ):"
echo "  pytcvatsrew/             7.4GB - Python environment (NUTNÉ!)"
echo "  screwdriver_images_for_cvat/ 40MB - Zdrojové obrázky (NUTNÉ!)"
echo "  screw-test/              39MB - Test CVAT export (NUTNÉ!)"
echo "  dataset_prepared/        2.9MB - RF-DETR dataset (NUTNÉ!)"
echo

read -p "Pokračovat s bezpečným cleanup? (y/n): " confirm
if [[ $confirm != [yY] ]]; then
    echo "❌ Cleanup zrušen"
    exit 1
fi

echo
echo "🧹 MAŽU JEN DEBUG ADRESÁŘE..."

# BEZPEČNÉ mazání - jen debug adresáře
if [ -d "debug_test_output" ]; then
    echo "❌ Mažu debug_test_output/ (1.4GB)"
    rm -rf debug_test_output/
else
    echo "ℹ️  debug_test_output/ neexistuje"
fi

if [ -d "final_coco_aware_output" ]; then
    echo "❌ Mažu final_coco_aware_output/ (1GB)"
    rm -rf final_coco_aware_output/
else
    echo "ℹ️  final_coco_aware_output/ neexistuje"
fi

if [ -d "final_output" ]; then
    echo "❌ Mažu final_output/ (511MB)"
    rm -rf final_output/
else
    echo "ℹ️  final_output/ neexistuje"
fi

if [ -d "dataset_coco_format" ]; then
    echo "❌ Mažu dataset_coco_format/ (2.5MB)"
    rm -rf dataset_coco_format/
else
    echo "ℹ️  dataset_coco_format/ neexistuje"
fi

echo
echo "🗂️ MAŽU NEPOTŘEBNÉ SOUBORY..."

# Bezpečné mazání souborů
if [ -f "pretrained_test_result.jpg" ]; then
    echo "❌ Mažu pretrained_test_result.jpg"
    rm -f pretrained_test_result.jpg
fi

if [ -f "train-00000-of-00001.parquet" ]; then
    echo "❌ Mažu train-00000-of-00001.parquet (109MB)"
    rm -f train-00000-of-00001.parquet
fi

if [ -f "validation-00000-of-00001.parquet" ]; then
    echo "❌ Mažu validation-00000-of-00001.parquet (5.1MB)"
    rm -f validation-00000-of-00001.parquet
fi

echo
echo "✅ BEZPEČNÝ CLEANUP DOKONČEN!"
echo
echo "📊 VÝSLEDEK:"
du -sh */ 2>/dev/null | sort -hr
echo
echo "🎯 DŮLEŽITÉ SOUBORY ZACHOVÁNY:"
echo "✅ pytcvatsrew/           - Python environment"
echo "✅ dataset_prepared/      - RF-DETR dataset" 
echo "✅ screwdriver_images_for_cvat/ - Zdrojové obrázky"
echo "✅ screw-test/            - Test CVAT export"
echo "✅ README.md + Python scripty - Nástroje"
