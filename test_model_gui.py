#!/usr/bin/env python3
"""
GUI aplikace pro testování RF-DETR Hard Hat Workers modelu
Umožňuje vybrat obrázek a zobrazit detekce s bounding boxy
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import os
import sys

# Import RF-DETR
try:
    from rfdetr.detr import RFDETRMedium
except ImportError:
    print("⚠️  Nelze importovat rfdetr. Ujisti se, že je nainstalovaný!")
    exit(1)


class ModelTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RF-DETR Hard Hat Workers - Model Tester 🎯")
        self.root.geometry("1400x900")
        
        # Config soubor
        self.config_file = Path("gui_config.json")
        self.config = self.load_config()
        
        # Podporované formáty obrázků (case-insensitive)
        self.supported_formats = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
            '.webp', '.ppm', '.pgm', '.pbm'
        }
        
        # Model a kategorie - OPRAVENÉ podle skutečných MODEL ID!
        # Model čísluje od 0: head=0, helmet=1, person=2
        self.model = None
        self.model_path = None
        self.categories = {
            1: "head",      # Model ID 0 = head
            2: "helmet",    # Model ID 1 = helmet (přilby!)
            3: "person"     # Model ID 2 = person
        }
        
        # Barvy pro kategorie (RGB formát pro PIL) - OPRAVENÉ
        self.colors = {
            0: (255, 165, 0),   # head - oranžová
            1: (0, 255, 0),     # helmet - zelená (nejdůležitější!)
            2: (0, 0, 255)      # person - modrá
        }
        
        # Současný obrázek
        self.current_image = None
        self.original_image = None
        self.display_scale = 1.0
        
        # Automatické hledání modelu při startu
        self.auto_find_model()
        
        self.setup_gui()
        
        # Aktualizace GUI statusu po načtení
        self.update_gui_status()
        
    def load_config(self):
        """Načte konfiguraci"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"last_model_path": None, "last_image_dir": None, "confidence": 0.5}
        
    def save_config(self):
        """Uloží konfiguraci"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Chyba při ukládání config: {e}")
        
    def auto_find_model(self):
        """Automaticky najde model při startu"""
        # Zkusíme napřed uložený model z config
        if self.config.get("last_model_path") and Path(self.config["last_model_path"]).exists():
            if self.load_model_from_path(self.config["last_model_path"]):
                print(f"✅ Načten model z config: {self.config['last_model_path']}")
                return
                
        # Pokud config neexistuje, hledáme automaticky
        possible_paths = [
            "hardhat_workers_model/checkpoint_best_total.pth",
            "hardhat-prepared/checkpoint_best_total.pth"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                try:
                    if self.load_model_from_path(path):
                        print(f"✅ Automaticky načten model: {path}")
                        return
                except:
                    continue
                    
    def update_gui_status(self):
        """Aktualizuje status v GUI"""
        if hasattr(self, 'model_status'):
            if self.model is not None and self.model_path:
                self.model_status.config(text=f"✅ Model načten: {Path(self.model_path).name}")
            else:
                self.model_status.config(text="❌ Model není načten")
                
    def is_supported_image(self, file_path):
        """Kontroluje, zda je soubor podporovaný obrázek"""
        if not Path(file_path).exists():
            return False
        
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_formats
        
    def load_model_from_path(self, model_path):
        """Načte model z cesty pomocí RF-DETR konstruktoru"""
        try:
            from rfdetr.detr import RFDETRMedium
            
            print(f"📦 Načítám model: {model_path}")
            
            # Nejprve zkontroluj existenci souboru
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model soubor neexistuje: {model_path}")
            
            # RF-DETR způsob - použij checkpoint jako pretrain_weights
            self.model = RFDETRMedium(pretrain_weights=str(model_path))
            self.model_path = str(model_path)  # ULOŽÍME CESTU!
            
            # ULOŽÍME DO CONFIG
            self.config["last_model_path"] = str(model_path)
            self.save_config()
            
            print(f"✅ Model načten z {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Chyba při načítání modelu: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_path = None
            return False
        
    def setup_gui(self):
        """Nastavení GUI"""
        # Hlavní frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Horní panel - načtení modelu
        top_frame = ttk.LabelFrame(main_frame, text="🤖 Model", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tlačítko pro načtení modelu
        ttk.Button(top_frame, text="📁 Načíst model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=(0, 10))
        
        # Status modelu
        self.model_status = ttk.Label(top_frame, text="❌ Model není načten")
        self.model_status.pack(side=tk.LEFT)
        
        # Střední panel - obrázek
        middle_frame = ttk.LabelFrame(main_frame, text="🖼️ Obrázek a detekce", padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Tlačítko pro načtení obrázku
        image_controls = ttk.Frame(middle_frame)
        image_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(image_controls, text="📸 Vybrat obrázek", 
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(image_controls, text="🔍 Detekovat objekty", 
                  command=self.detect_objects).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(image_controls, text="🔄 Reset obrázek", 
                  command=self.reset_image).pack(side=tk.LEFT)
        
        # Confidence threshold
        ttk.Label(image_controls, text="Confidence:").pack(side=tk.LEFT, padx=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(image_controls, from_=0.1, to=0.9, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL, length=150)
        confidence_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        self.confidence_label = ttk.Label(image_controls, text="0.5")
        self.confidence_label.pack(side=tk.LEFT)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Canvas pro obrázek
        canvas_frame = ttk.Frame(middle_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbary pro canvas
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(middle_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(fill=tk.X)
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Spodní panel - výsledky
        bottom_frame = ttk.LabelFrame(main_frame, text="📊 Výsledky detekce", padding=10)
        bottom_frame.pack(fill=tk.X)
        
        # Treeview pro výsledky
        columns = ('ID', 'Kategorie', 'Confidence', 'X', 'Y', 'Šířka', 'Výška')
        self.results_tree = ttk.Treeview(bottom_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        
        self.results_tree.pack(fill=tk.X, pady=(0, 5))
        
        # Statistiky
        self.stats_label = ttk.Label(bottom_frame, text="Žádné detekce")
        self.stats_label.pack()
        
    def update_confidence_label(self, value):
        """Aktualizuje label pro confidence threshold"""
        self.confidence_label.config(text=f"{float(value):.2f}")
        
    def load_model(self):
        """Načte RF-DETR model"""
        model_path = filedialog.askopenfilename(
            title="Vyberte model checkpoint",
            filetypes=[
                ("PyTorch checkpoints", "*.pth"),
                ("Všechny soubory", "*.*")
            ],
            initialdir="."
        )
        
        if not model_path:
            return
            
        self.model_status.config(text="⏳ Načítám model...")
        self.root.update()
        
        if self.load_model_from_path(model_path):
            self.model_status.config(text=f"✅ Model načten: {Path(model_path).name}")
            # ULOŽÍME CESTU DO CONFIG
            self.config["last_model_path"] = str(model_path)
            self.save_config()
            messagebox.showinfo("Úspěch", f"Model byl úspěšně načten!\n{Path(model_path).name}")
        else:
            self.model_status.config(text="❌ Chyba při načítání modelu")
            messagebox.showerror("Chyba", f"Nepodařilo se načíst model:\n{Path(model_path).name}")
            
    def load_image(self):
        """Načte obrázek s ošetřením všech formátů"""
        # Vytvoření filtru pro všechny podporované formáty
        formats_str = " ".join([f"*{fmt}" for fmt in self.supported_formats])
        formats_str += " " + " ".join([f"*{fmt.upper()}" for fmt in self.supported_formats])
        
        image_path = filedialog.askopenfilename(
            title="Vyberte obrázek",
            initialdir=self.config.get("last_image_dir", "."),
            filetypes=[
                ("Všechny obrázky", formats_str),
                ("JPEG", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("PNG", "*.png *.PNG"),
                ("BMP", "*.bmp *.BMP"),
                ("TIFF", "*.tiff *.tif *.TIFF *.TIF"),
                ("WebP", "*.webp *.WEBP"),
                ("Všechny soubory", "*.*")
            ]
        )
        
        if not image_path:
            return
            
        # Kontrola podporovaného formátu
        if not self.is_supported_image(image_path):
            messagebox.showwarning("Nepodporovaný formát", 
                                 f"Soubor {Path(image_path).suffix} není podporován!\n"
                                 f"Podporované formáty: {', '.join(sorted(self.supported_formats))}")
            return
            
        try:
            # Načtení obrázku s ošetřením různých formátů
            self.original_image = Image.open(image_path)
            
            # Konverze na RGB pokud je potřeba (např. RGBA, P, L)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            
            self.current_image = self.original_image.copy()
            
            # Výpočet scale pro zobrazení
            self.calculate_display_scale()
            
            # Zobrazení na canvasu
            self.display_image()
            
            # Vymazání předchozích výsledků
            self.clear_results()
            
            # ULOŽENÍ DO CONFIG
            self.config["last_image_dir"] = str(Path(image_path).parent)
            self.config["last_image_path"] = image_path
            self.save_config()
            
            # Informace o obrázku
            size_info = f"📏 {self.original_image.width}×{self.original_image.height}px"
            if self.display_scale != 1.0:
                size_info += f" (zobrazeno {self.display_scale:.1%})"
            print(f"✅ Obrázek načten: {Path(image_path).name} {size_info}")
            
        except Exception as e:
            messagebox.showerror("Chyba", f"Nepodařilo se načíst obrázek:\n{str(e)}")
            
    def calculate_display_scale(self):
        """Vypočte scale pro zobrazení na obrazovce"""
        if self.original_image is None:
            return
            
        # Maximální velikost pro zobrazení
        max_display_width = 1000
        max_display_height = 600
        
        width_scale = max_display_width / self.original_image.width
        height_scale = max_display_height / self.original_image.height
        
        self.display_scale = min(1.0, width_scale, height_scale)
            
    def display_image(self):
        """Zobrazí obrázek na canvasu s automatickým škálováním"""
        if self.current_image is None:
            return
            
        # Vytvoření kopie pro zobrazení
        display_image = self.current_image.copy()
        
        # Škálování podle vypočítaného poměru
        if self.display_scale != 1.0:
            new_width = int(display_image.width * self.display_scale)
            new_height = int(display_image.height * self.display_scale)
            display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Konverze pro tkinter
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Vymazání canvasu a zobrazení obrázku
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Nastavení scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def detect_objects(self):
        """Provede detekci objektů"""
        if self.model is None:
            messagebox.showwarning("Upozornění", "Nejprve načtěte model!")
            return
            
        if self.original_image is None:
            messagebox.showwarning("Upozornění", "Nejprve načtěte obrázek!")
            return
            
        try:
            # Konverze PIL na numpy array
            image_array = np.array(self.original_image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # RGB -> BGR pro OpenCV
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # SPRÁVNÁ PREDIKCE - použijeme načtený model!
            confidence_threshold = self.confidence_var.get()
            
            # Použijeme model který už máme načtený
            results = self.model.predict(
                image_array, 
                confidence=confidence_threshold
                # ODSTRANĚNO: checkpoint=self.model_path - model už je načtený!
            )
            
            # Zpracování výsledků
            self.process_results(results, image_array)
            
        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba při detekci:\n{str(e)}")
            
    def process_results(self, results, image_array):
        """Zpracuje výsledky detekce a zobrazí je"""
        if self.original_image is None:
            return
            
        # Pracovní kopie původního obrázku
        result_image = self.original_image.copy()
        
        # Vymazání předchozích výsledků
        self.clear_results()
        
        detections = []
        
        # Zpracování detekcí podle typu results
        try:
            if hasattr(results, 'xyxy') and hasattr(results, 'confidence') and hasattr(results, 'class_id'):
                # supervision.detection.core.Detections formát (RF-DETR 1.2.1)
                boxes = results.xyxy  # již numpy array [N, 4] - x1, y1, x2, y2
                scores = results.confidence  # numpy array [N]
                classes = results.class_id  # numpy array [N]
                
                print(f"🔍 Supervision.Detections: {len(boxes)} objektů")
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    cat_name = self.categories.get(cls, f"Unknown_{cls}")
                    print(f"   {i+1}. CLASS_ID={cls} -> {cat_name}: {score:.3f} - bbox: {box}")
                
            elif isinstance(results, list) and len(results) > 0:
                # RF-DETR tuple formát - každý výsledek je tuple (bbox, None, score, class, None, {})
                boxes_list = []
                scores_list = []
                classes_list = []
                
                for result in results:
                    if isinstance(result, tuple) and len(result) >= 4:
                        bbox, _, score, cls, _, _ = result
                        if bbox is not None and score is not None and cls is not None:
                            boxes_list.append(bbox)
                            scores_list.append(float(score))
                            classes_list.append(int(cls))
                
                boxes = np.array(boxes_list) if boxes_list else np.array([])
                scores = np.array(scores_list) if scores_list else np.array([])
                classes = np.array(classes_list) if classes_list else np.array([])
                
                print(f"🔍 RF-DETR tuple detekce: {len(boxes)} objektů")
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    cat_name = self.categories.get(cls, f"Unknown_{cls}")
                    print(f"   {i+1}. CLASS_ID={cls} -> {cat_name}: {score:.3f} - bbox: {box}")
                
            elif hasattr(results, 'boxes') and results.boxes is not None:
                # YOLOv8/Ultralytics formát
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                
            elif hasattr(results, 'pred') and len(results.pred) > 0:
                # Jiný formát
                pred = results.pred[0]
                boxes = pred[:, :4].cpu().numpy()  # x1, y1, x2, y2
                scores = pred[:, 4].cpu().numpy()
                classes = pred[:, 5].cpu().numpy().astype(int)
                
            else:
                # Žádné detekce nebo neznámý formát
                print(f"⚠️  Neznámý formát výsledků: {type(results)}")
                if hasattr(results, '__len__'):
                    print(f"    Délka: {len(results)}")
                if isinstance(results, list) and len(results) > 0:
                    print(f"    První prvek: {type(results[0])}")
                boxes = np.array([])
                scores = np.array([])
                classes = np.array([])
                
        except Exception as e:
            print(f"Chyba při zpracování výsledků: {e}")
            boxes = np.array([])
            scores = np.array([])
            classes = np.array([])
        
        # Kreslení bounding boxů na PIL obrázek
        if len(boxes) > 0:
            draw = ImageDraw.Draw(result_image)
            
            # Pokus o načtení fontu
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Získání názvu kategorie a barvy
                category_name = self.categories.get(cls, f"Unknown_{cls}")
                color = self.colors.get(cls, (128, 128, 128))
                
                # Kreslení obdélníku
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Text s kategorií a confidence
                label = f"{category_name} {score:.2f}"
                
                # Pozadí pro text
                bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(bbox, fill=color)
                
                # Text
                draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)
                
                # Uložení detekce
                detections.append({
                    'id': i + 1,
                    'category': category_name,
                    'confidence': score,
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                })
        
        # Aktualizace současného obrázku
        self.current_image = result_image
        
        # Zobrazení výsledku
        self.display_image()
        
        # Zobrazení výsledků v tabulce
        self.display_results(detections)
        
    def display_results(self, detections):
        """Zobrazí výsledky v tabulce"""
        # Vymazání předchozích výsledků
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Přidání nových výsledků
        for detection in detections:
            self.results_tree.insert('', 'end', values=(
                detection['id'],
                detection['category'],
                f"{detection['confidence']:.3f}",
                detection['x'],
                detection['y'],
                detection['width'],
                detection['height']
            ))
        
        # Statistiky
        if detections:
            stats_by_category = {}
            for det in detections:
                cat = det['category']
                if cat not in stats_by_category:
                    stats_by_category[cat] = 0
                stats_by_category[cat] += 1
            
            stats_text = f"Celkem: {len(detections)} detekcí | "
            stats_text += " | ".join([f"{cat}: {count}" for cat, count in stats_by_category.items()])
            self.stats_label.config(text=stats_text)
        else:
            self.stats_label.config(text="Žádné detekce")
            
    def clear_results(self):
        """Vymaže výsledky"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.stats_label.config(text="Žádné detekce")
        
    def reset_image(self):
        """Resetuje obrázek na původní stav"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image()
            self.clear_results()


def main():
    """Hlavní funkce aplikace"""
    try:
        import tkinter as tk
        root = tk.Tk()
        
        print("🚀 RF-DETR Hard Hat Workers GUI Tester")
        print("=" * 50)
        print(f"✅ CUDA dostupná: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "⚠️  CUDA nedostupná")
        print(f"📁 Pracovní adresář: {Path.cwd()}")
        
        # Vytvoříme GUI jednou a spustíme
        gui = ModelTesterGUI(root)
        
        print("\n📋 GUI je připraveno!")
        print("💡 Tip: Model a poslední obrázek se ukládají automaticky")
        
        # Spustíme GUI
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Chyba při spuštění GUI: {e}")
        print("💡 Možná chybí tkinter. Nainstaluj: sudo apt-get install python3-tk")


if __name__ == "__main__":
    main()
