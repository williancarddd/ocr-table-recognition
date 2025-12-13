##### DEPRECIADO #####

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
