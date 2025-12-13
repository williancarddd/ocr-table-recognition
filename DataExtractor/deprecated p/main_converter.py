##### DEPRECIADO #####

import os
from yolo_converter import YOLOConverter
from maskrcnn_converter import MaskRCNNConverter


def download_dataset():
    import urllib.request
    import zipfile
    
    zip_url = "https://github.com/cndplab-founder/ICDAR2019_cTDaR/archive/refs/heads/master.zip"
    target_dir = "dataset"
    zip_file = "dataset.zip"
    temp_dir = "ICDAR2019_cTDaR-master"
    
    if os.path.exists(target_dir):
        print(f"{target_dir} ja existe")
        return
    print("baixando o dataset.. espere")
    
    try:
        urllib.request.urlretrieve(zip_url, zip_file)
        print("extraindo dataset")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.rename(temp_dir, target_dir)
        os.remove(zip_file)
        print("download do dataset")
        
    except Exception as e:
        print(f"erro: {e}")
        raise



def main():
    download_dataset()
    print("YOLO Segmentation")

    
    yolo_converter_train = YOLOConverter(
        base_dir="./dataset/training/TRACKB1/ground_truth",
        output_dir="../yolo/dataset_YOLO_train",
        train_ratio=0.8,
        class_id=0
    )
    yolo_converter_train.run()

    yolo_converter_test = YOLOConverter(
        base_dir="./dataset/test/TRACKB1",
        output_dir="../yolo/dataset_YOLO_test",
        train_ratio=0.8,
        class_id=0
    )
    yolo_converter_test.run()

    print("\n" + "="*50)
    print("Mask R-CNN coco")
    print("="*50)
    
    rcnn_converter_train = MaskRCNNConverter(
        base_dir="./dataset/training/TRACKB1/ground_truth",
        output_dir="../rcnn/dataset_RCNN_train",
        train_ratio=0.8
    )
    rcnn_converter_train.run()

    rcnn_converter_test = MaskRCNNConverter(
        base_dir="./dataset/test/TRACKB1/",
        output_dir="../rcnn/dataset_RCNN_test",
        train_ratio=0.8
    )
    rcnn_converter_test.run()


    print("conversão concluid")



if __name__ == "__main__":
    main()