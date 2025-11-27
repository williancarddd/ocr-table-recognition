import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from glob import glob

import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple

import random


class ICDARYOLOConverter:
    # Deixei a classe como estática 
    class_id = 0
    class_label = 'cell'

    @staticmethod
    def create_folders(root_dataset_path : str | Path, 
                       include_test : bool = False):

        split_paths = ['train', 'val']
        if include_test:
            split_paths+= ['test']

        if not os.path.exists(root_dataset_path):
            os.makedirs(root_dataset_path)

        for split_path in split_paths:  
            os.makedirs(os.path.join(root_dataset_path, split_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(root_dataset_path, split_path, 'labels'), exist_ok=True)
    
    @staticmethod
    def normalize_mask_points( 
                         mask : Iterable[Tuple[float]], 
                         image_width : float, 
                         image_height : float):
        '''
        As coordenadas dos pontos dos polígonos devem ser normalizados pela respectiva 
        dimensão da imagem. Dessa forma, x_norm = x / image_width e y_norm = y / image_height.
        No arquivo TXT final, deve constar class_id x1_norm y2_norm x2_norm y2_norm ... xn_norm yn_norm 
        '''

        new_xy_points = []
        for xy_point in mask:
            x, y = xy_point

            x_norm = x / image_width
            y_norm = y / image_height
            
            # lista para facilitar a conversão pelo JSON ao ler a partir do CSV
            new_xy_points.append([x_norm, y_norm])
        
        return new_xy_points
    
    @staticmethod
    def normalize_masks(
            masks : Iterable[Iterable[Tuple[float]]],
            image_width : float, 
            image_height : float):
        
        '''
        Normaliza todas as máscaras contidas em uma iterável de máscaras.
        '''

        normalized_masks = []
    
        for mask in masks:        
            # normaliza os pontos da máscara
            normilized_mask = ICDARYOLOConverter.normalize_mask_points(mask, image_width, image_height)  
            normalized_masks.append(normilized_mask) 
        
        return normalized_masks

    @staticmethod
    def create_txt_file_content(normalized_masks : Iterable[Iterable[Tuple[float]]]):

        lines = []
        for mask in normalized_masks:
            # constroi a string com os pontos normalizados x1_norm y2_norm x2_norm y2_norm ... xn_norm yn_norm 
            poly_str = " ".join([f"{coord:.6f}" for xy_point in mask for coord in xy_point ])
            # define a string que representa a linha com class_id x1_norm y2_norm x2_norm y2_norm ... xn_norm yn_norm 
            line = f"{ICDARYOLOConverter.class_id} {poly_str}"
            # adiciona a linnha na lista de linhas
            lines.append(line)       

        # concatena todas as linhas do arquivo TXT em uma string separadas pela quebra de linha
        return "\n".join(lines)

    @staticmethod
    def create_txt_file(content : str,  path : str | Path):
        with open(path, 'w') as txt_file: 
            txt_file.write(content)


    def process_masks(masks : Iterable[Iterable[Tuple[float]]], 
                      image_width : int, 
                      image_height : int,
                      path_txt_file : str | Path):
        
        normalized_masks = ICDARYOLOConverter.normalize_masks(masks, image_width, image_height)
        txt_file_content =  ICDARYOLOConverter.create_txt_file_content(normalized_masks)
        ICDARYOLOConverter.create_txt_file(txt_file_content, path_txt_file)

        return normalized_masks

    
    @staticmethod
    def create_yaml( 
                    output_dir : str | Path = None, 
                    train_fold_path : str | Path = None, 
                    val_fold_path : str | Path = None):  #cria um yaml

        yaml_path = os.path.join(output_dir, "dataset.yaml")
        
        with open(yaml_path, "w") as f:
            f.write(
f"""path: {os.path.abspath(output_dir)}
train: {os.path.abspath(train_fold_path)}
val: {os.path.abspath(val_fold_path)}
task: segment

names:
  {ICDARYOLOConverter.class_id}: {ICDARYOLOConverter.class_label}
"""
            )
        
        print(f"\nYAML criado em: {yaml_path}")
        return yaml_path


class YOLOConverter:
    
    def __init__(self, base_dir="./TRACKB1", output_dir="./dataset_YOLO", train_ratio=0.8, class_id=0):
        self.base_dir = base_dir
        self.gt_dir = base_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        self.train_ratio = train_ratio
        self.class_id = class_id
        
        self.train_pairs = []
        self.val_pairs = []
    
    def create_folders(self):
        for d in [self.images_dir, self.labels_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)
    
    def find_files(self):
        image_files = sorted(
            glob(os.path.join(self.gt_dir, "*.jpg")) +
            glob(os.path.join(self.gt_dir, "*.png")) +
            glob(os.path.join(self.gt_dir, "*.TIFF"))
        )
        xml_files = sorted(glob(os.path.join(self.gt_dir, "*.xml")))
        print(f"XMLs: {len(xml_files)}")
        print(f"Imagens:{len(image_files)}")
        
        return xml_files, image_files
    
    def associate_pairs(self, xml_files):
        pairs = []
        for xml in xml_files:
            base = os.path.splitext(os.path.basename(xml))[0]
            
            for ext in [".jpg", ".png", ".TIFF"]:
                img_path = os.path.join(self.gt_dir, base + ext)
                if os.path.exists(img_path):
                    pairs.append((xml, img_path))
                    break
        
        return pairs
    
    def split_dataset(self, pairs):
        random.shuffle(pairs)
        train_split = int(len(pairs) * self.train_ratio)
        
        self.train_pairs = pairs[:train_split]
        self.val_pairs = pairs[train_split:]
        
        print(f"Treino: {len(self.train_pairs)} imagens")
        print(f"Val:    {len(self.val_pairs)} imagens")
    
    def normalize_points(self, points, w, h):
        pts = []
        for p in points.split():
            x, y = map(float, p.split(","))
            pts.append(x / w)
            pts.append(y / h)
        return pts
    
    def convert_annotation(self, xml_path, img_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img = Image.open(img_path)
        w, h = img.size       
        lines = []
        for table in root.findall("table"):
            for cell in table.findall("cell"):
                coords = cell.find("Coords")
                if coords is None:
                    continue               
                points = coords.attrib["points"]
                poly_norm = self.normalize_points(points, w, h)
                poly_str = " ".join([f"{p:.6f}" for p in poly_norm])
                line = f"{self.class_id} {poly_str}"
                lines.append(line)       
        return "\n".join(lines)
    
    def process_pair(self, xml_path, img_path, split):
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]
        
        img_out = os.path.join(self.images_dir, split, img_name)
        label_out = os.path.join(self.labels_dir, split, base + ".txt")
        
        shutil.copy(img_path, img_out)
        
        txt = self.convert_annotation(xml_path, img_path)
        with open(label_out, "w", encoding="utf-8") as f:
            f.write(txt)
        
        print(f"ok {img_name} → {split}")
    
    def process_all(self):
        for xml_path, img_path in self.train_pairs:
            self.process_pair(xml_path, img_path, "train")
        for xml_path, img_path in self.val_pairs:
            self.process_pair(xml_path, img_path, "val")
    
    def create_yaml(self): #cria um yaml
        yaml_path = os.path.join(self.output_dir, "trackb1-seg.yaml")
        
        with open(yaml_path, "w") as f:
            f.write(
f"""path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
task: segment

names:
  0: cell
"""
            )
        
        print(f"\nYAML criado em: {yaml_path}")
        return yaml_path
    
    def run(self):
        self.create_folders()
        xml_files, _ = self.find_files()
        pairs = self.associate_pairs(xml_files)
        self.split_dataset(pairs)
        self.process_all()
        yaml_path = self.create_yaml()

if __name__ == "__main__":
    converter = YOLOConverter(
        base_dir="./TRACKB1",
        output_dir="./dataset_YOLO",
        train_ratio=0.8,
        class_id=0
    )
    
    converter.run()