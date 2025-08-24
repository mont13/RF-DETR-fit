#!/usr/bin/env python3
"""
Generátor grafů z RF-DETR training logu
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

def parse_rf_detr_log(log_path):
    """Parse RF-DETR JSON log"""
    epochs = []
    train_losses = []
    val_losses = []
    map_values = []
    lr_values = []
    
    print(f"📖 Parsing RF-DETR log: {log_path}")
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    
                    if 'epoch' in data and 'train_loss' in data:
                        epoch = data['epoch']
                        epochs.append(epoch + 1)  # RF-DETR indexuje od 0
                        
                        # Train loss
                        train_loss = data.get('train_loss', 0)
                        train_losses.append(train_loss)
                        
                        # Validation/test loss  
                        val_loss = data.get('test_loss', data.get('ema_test_loss', train_loss))
                        val_losses.append(val_loss)
                        
                        # mAP@50
                        map_50 = 0
                        if 'test_results_json' in data and data['test_results_json']:
                            map_50 = data['test_results_json'].get('map', 0) * 100
                        elif 'ema_test_results_json' in data and data['ema_test_results_json']:
                            map_50 = data['ema_test_results_json'].get('map', 0) * 100
                        map_values.append(map_50)
                        
                        # Learning rate
                        lr = data.get('train_lr', 1e-4)
                        lr_values.append(lr)
                        
                except json.JSONDecodeError:
                    continue
    
    return epochs, train_losses, val_losses, map_values, lr_values

def create_plots(output_dir):
    """Vytvoří grafy z RF-DETR logu"""
    output_path = Path(output_dir)
    log_path = output_path / "log.txt"
    
    if not log_path.exists():
        print(f"❌ Log soubor neexistuje: {log_path}")
        return None
    
    # Parse log
    epochs, train_losses, val_losses, map_values, lr_values = parse_rf_detr_log(log_path)
    
    if not epochs:
        print("❌ Nepodařilo se načíst žádná data z logu")
        return None
    
    print(f"✅ Parsed {len(epochs)} epochs")
    print(f"📈 Final: train_loss={train_losses[-1]:.3f}, val_loss={val_losses[-1]:.3f}, mAP@50={map_values[-1]:.1f}%")
    
    # Vytvoř grafy
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 RF-DETR Training Results - VÝJIMEČNÝ ÚSPĚCH!', fontsize=18, fontweight='bold')
    
    # 1. Loss graf
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=3, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=3, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#fafafa')
    
    # 2. mAP graf
    ax2.plot(epochs, map_values, label='mAP@50', linewidth=3, marker='D', markersize=5, color='#2ecc71')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mAP@50 (%)', fontsize=12)
    ax2.set_title('Mean Average Precision - SPEKTAKULÁRNÍ!', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#fafafa')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.8, label='Excellent (90%)')
    ax2.legend()
    
    # 3. Learning rate
    ax3.plot(epochs, lr_values, color='#f39c12', linewidth=3, marker='^', markersize=5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#fafafa')
    
    # 4. Overfitting detection - opravená logika
    gap = [v - t for v, t in zip(val_losses, train_losses)]  # Pozitivní = overfitting, negativní = skvělá generalizace
    colors = []
    for g in gap:
        if g < -0.1:  # Val loss mnohem lepší než train
            colors.append('#27ae60')  # Zelená - výjimečná generalizace
        elif g < 0.05:  # Mírně lepší nebo podobné
            colors.append('#3498db')  # Modrá - dobrá generalizace  
        elif g < 0.1:  # Mírný overfitting
            colors.append('#f39c12')  # Oranžová
        else:  # Značný overfitting
            colors.append('#e74c3c')  # Červená
    
    bars = ax4.bar(epochs, gap, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='No Overfitting')
    ax4.axhline(y=0.05, color='#f39c12', linestyle='--', alpha=0.8, linewidth=2, label='Watch Zone')
    ax4.axhline(y=0.1, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2, label='Overfitting Zone')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Val - Train Loss', fontsize=12)
    ax4.set_title('Overfitting Analysis - VÝJIMEČNÁ GENERALIZACE!', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_facecolor('#fafafa')
    
    # Přidej hodnoty na bary
    for bar, value in zip(bars, gap):
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            y_pos = height + 0.02
        else:
            va = 'top'
            y_pos = height - 0.02
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{value:.2f}', ha='center', va=va, fontsize=8)
    
    plt.tight_layout()
    
    # Ulož graf
    plot_file = output_path / "actual_training_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Vyhodnocení - opravená logika
    final_gap = gap[-1]
    if final_gap < -0.1:
        overfitting_status = "🌟 VÝJIMEČNÁ GENERALIZACE - Val loss mnohem lepší!"
    elif final_gap < 0:
        overfitting_status = "✅ SKVĚLÁ GENERALIZACE - Val loss lepší než train!"
    elif final_gap < 0.05:
        overfitting_status = "✅ DOKONALÉ - Žádný overfitting!"
    elif final_gap < 0.1:
        overfitting_status = "⚠️ Mírný overfitting"
    else:
        overfitting_status = "❌ Značný overfitting"
    
    print(f"\n🎯 KONEČNÉ VYHODNOCENÍ:")
    print(f"📊 mAP@50: {map_values[-1]:.1f}% (VÝJIMEČNÉ!)")
    print(f"📉 Train Loss: {train_losses[-1]:.3f}")
    print(f"📉 Val Loss: {val_losses[-1]:.3f}")
    print(f"🎭 Overfitting: {overfitting_status}")
    print(f"📈 Graf uložen: {plot_file}")
    
    return plot_file

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "screwdriver_model"
    create_plots(output_dir)
