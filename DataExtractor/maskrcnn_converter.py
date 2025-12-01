import os
import json
import shutil
import yaml
from PIL import Image


class YOLO2MaskRCNN:
    
    def __init__(self, folds_dir="dataset_folds", output_dir="dataset_rcnn"):
        self.folds_dir = folds_dir
        self.output_dir = output_dir

    def load_classes(self, yml_path):
        with open(yml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data["names"]
        
        categories = []
        for i, name in enumerate(names):
            categories.append({"id": int(i), "name": name})
        return categories

    def yolo_to_coco_bbox(self, xc, yc, w, h, img_w, img_h):
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        return [
            x1,
            y1,
            w * img_w,
            h * img_h
        ]

    def process_split(self, split_path, split_name, categories, out_images_dir):
    
        images_dir = os.path.join(split_path, "images")
        labels_dir = os.path.join(split_path, "labels")

        coco = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        ann_id = 1

        for image_name in os.listdir(images_dir):
            if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(images_dir, image_name)
            lbl_path = os.path.join(labels_dir, image_name.rsplit(".", 1)[0] + ".txt")

            img = Image.open(img_path)
            w, h = img.size

            image_id = len(coco["images"]) + 1

            shutil.copy(img_path, os.path.join(out_images_dir, image_name))

            coco["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": w,
                "height": h
            })

            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = line.strip().split()

                cls = int(values[0]) # class id
                coords = list(map(float, values[1:])) # bounding box coords

                # YOLO segmentation: x1, y1, x2, y2, ..., xN, yN (normalized)
                polygon = []
                for i in range(0, len(coords), 2):
                    x = round(coords[i] * w, 2)
                    y = round(coords[i+1] * h, 2)
                    polygon.extend([x, y])

                # cálculo do bbox
                xs = polygon[0::2]
                ys = polygon[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, round(x_max - x_min, 2), round(y_max - y_min, 2)]

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "segmentation": [polygon],   
                    "iscrowd": 0
                })
                ann_id += 1

        return coco

    def run(self):

        for fold in sorted(os.listdir(self.folds_dir)):
            fold_path = os.path.join(self.folds_dir, fold)
            if not os.path.isdir(fold_path):
                continue

            print(f"\n Processando fold: {fold}")

            yml_path = os.path.join(fold_path, "dataset.yaml")
            categories = self.load_classes(yml_path)

            out_fold_dir = os.path.join(self.output_dir, fold)
            os.makedirs(out_fold_dir, exist_ok=True)

            # PROCESSAR train
            train_path = os.path.join(fold_path, "train")
            out_train_dir = os.path.join(out_fold_dir, "train")
            os.makedirs(out_train_dir, exist_ok=True)

            print(f" → Convertendo train...")
            coco_train = self.process_split(train_path, "train", categories, out_train_dir)
            with open(os.path.join(out_fold_dir, "train.json"), "w") as f:
                json.dump(coco_train, f, indent=2)

            # PROCESSAR val
            val_path = os.path.join(fold_path, "val")
            out_val_dir = os.path.join(out_fold_dir, "val")
            os.makedirs(out_val_dir, exist_ok=True)

            print(f" → Convertendo val...")
            coco_val = self.process_split(val_path, "val", categories, out_val_dir)
            with open(os.path.join(out_fold_dir, "val.json"), "w") as f:
                json.dump(coco_val, f, indent=2)

            print(f"Fold {fold} finalizado.")


if __name__ == "__main__":
    converter = YOLO2MaskRCNN(
        folds_dir="dataset_folds",
        output_dir="dataset_rcnn"
    )
    converter.run()
