import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image
from glob import glob
from datetime import datetime


class MaskRCNNConverter:
    
    def __init__(self, base_dir="./TRACKB1", output_dir="./dataset_RCNN", train_ratio=0.8):
        self.base_dir = base_dir
        self.gt_dir = base_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        
        self.categories = [
            {"id": 1, "name": "cell", "supercategory": "table"}
        ]
        
        self.train_pairs = []
        self.val_pairs = []
    
    def create_folders(self):
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)
    
    def find_files(self):
        image_files = sorted(
            glob(os.path.join(self.gt_dir, "*.jpg")) +
            glob(os.path.join(self.gt_dir, "*.png")) +
            glob(os.path.join(self.gt_dir, "*.TIFF"))
        )
        xml_files = sorted(glob(os.path.join(self.gt_dir, "*.xml")))
        
        print(f"xmls encontrados: {len(xml_files)}")
        print(f"imagens encontradas: {len(image_files)}")
        
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
    
    def parse_points_to_polygon(self, points_str):
        polygon = []
        for point in points_str.split():
            x, y = map(float, point.split(","))
            polygon.extend([x, y])
        return polygon
    
    def polygon_to_bbox(self, polygon):
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]
        
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        
        width = x_max - x_min
        height = y_max - y_min
        
        return [x_min, y_min, width, height]
    
    def calculate_area(self, polygon):
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]
        
        area = 0.0
        n = len(xs)
        for i in range(n):
            j = (i + 1) % n
            area += xs[i] * ys[j]
            area -= xs[j] * ys[i]
        
        return abs(area) / 2.0
    
    def process_pair(self, xml_path, img_path, img_id, split):
        img = Image.open(img_path)
        width, height = img.size
        
        img_filename = os.path.basename(img_path)
        img_dest = os.path.join(self.output_dir, split, img_filename)
        shutil.copy(img_path, img_dest)
        
        image_data = {
            "id": img_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        }
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations_data = []
        cell_count = 0
        
        for table in root.findall("table"):
            for cell in table.findall("cell"):
                coords = cell.find("Coords")
                if coords is None:
                    continue
                
                points_str = coords.attrib["points"]
                segmentation = self.parse_points_to_polygon(points_str)
                bbox = self.polygon_to_bbox(segmentation)
                area = self.calculate_area(segmentation)
                
                annotations_data.append({
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": area
                })
                
                cell_count += 1
        
        print(f"ok {img_filename} → {split} . {cell_count} células")
        
        return image_data, annotations_data
    
    def process_split(self, pairs, split_name):   
        coco_data = {
            "info": {
                "description": "Table Cell Detection Dataset",
                "version": "1.0",
                "year": 2025,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self.categories
        }
        
        annotation_id = 1
        
        for img_id, (xml_path, img_path) in enumerate(pairs, start=1):
            image_data, annotations_list = self.process_pair(xml_path, img_path, img_id, split_name)
            
            coco_data["images"].append(image_data)
            
            for ann_data in annotations_list:
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [ann_data["segmentation"]],
                    "area": ann_data["area"],
                    "bbox": ann_data["bbox"],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
        
        json_path = os.path.join(self.output_dir, f"{split_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"JSON salvo: {json_path}")
    
    def run(self):
        
        self.create_folders()
        xml_files, _ = self.find_files()
        pairs = self.associate_pairs(xml_files)
        self.split_dataset(pairs)
        
        self.process_split(self.train_pairs, "train")
        self.process_split(self.val_pairs, "val")

if __name__ == "__main__":
    converter = MaskRCNNConverter(
        base_dir="./TRACKB1",
        output_dir="./dataset_RCNN",
        train_ratio=0.8
    )
    
    converter.run()