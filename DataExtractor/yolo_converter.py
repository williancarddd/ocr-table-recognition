import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from glob import glob

import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple, Literal

import random



class YOLOConverter:
    
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
    def convert_xy2yolo(xmin : int, ymin : int, xmax : int, ymax : int):
        '''
        DocString
        '''
        
        xcentral = (xmax + xmin) // 2
        ycentral = (ymax + ymin) // 2
        width = xmax - xmin
        height = ymax - ymin

        return xcentral, ycentral, width, height
    
    @staticmethod
    def convert_bouding_box_xy2yolo(bounding_box : Iterable[Tuple[int]],
                                      image_width : int, image_height : int):
        '''
        Converte uma bouding box dada por [[xmin, ymin], [xmax, ymax]] em uma bounding box
        com as coordenadas dos pontos normalizadas pelas respectivas dimensões das imagens
        retornando [ xcentral / image_width, ycentral / image_height, width / image_width, height / image_height ]
        '''
        
        xmin, ymin, xmax, ymax = sum(bounding_box, [])
        xcentral, ycentral, width, height = YOLOConverter.convert_xy2yolo(xmin, ymin, xmax, ymax)
        
        xcentral /= image_width
        ycentral /= image_height
        width /= image_width
        height /= image_height

        return [xcentral, ycentral, width, height] 


    @staticmethod
    def convert_bounding_boxes_xy2yolo(bounding_boxes : Iterable[Iterable[Tuple[int]]],
                                       image_width : int, image_height : int):
        '''  
        Converte umas lista de bounding boxes dadas por [[[x0min, y0min], [x0max, y0max]], ...[[xnmin, ynmin], [xnmax, ynmax]]]
        em uma lista de bounding boxes com as coordenadas dos pontos normalizadas pelas respectivas dimensões das imagens dadas
        por [ xcentral / image_width, ycentral / image_height, width / image_width, height / image_height ]
        '''
        
        converted_bouding_boxes = [YOLOConverter.convert_bouding_box_xy2yolo(boudning_box, image_width, image_height)
                                   for boudning_box in bounding_boxes]
        
        return converted_bouding_boxes
    
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
            normilized_mask = YOLOConverter.normalize_mask_points(mask, image_width, image_height)  
            normalized_masks.append(normilized_mask) 
        
        return normalized_masks

    @staticmethod
    def create_mask_txt_file_content(normalized_masks : Iterable[Iterable[Tuple[float]]],
                                     class_ids : Iterable[int]):

        lines = []
        for class_id, mask in zip(class_ids, normalized_masks):
            # constroi a string com os pontos normalizados x1_norm y2_norm x2_norm y2_norm ... xn_norm yn_norm 
            poly_str = " ".join([f"{coord:.6f}" for xy_point in mask for coord in xy_point ])
            # define a string que representa a linha com class_id x1_norm y2_norm x2_norm y2_norm ... xn_norm yn_norm 
            line = f"{class_id} {poly_str}"
            # adiciona a linnha na lista de linhas
            lines.append(line)       

        # concatena todas as linhas do arquivo TXT em uma string separadas pela quebra de linha
        return "\n".join(lines)
    
    @staticmethod
    def create_bbox_txt_file_content(normalized_bouding_boxes : Iterable[Iterable[Tuple[float]]],
                                     class_ids : Iterable[int]):

        lines = []
        for class_id, bbox in zip(class_ids, normalized_bouding_boxes):
            xcentral, ycentral, width, height = bbox
            line = f"{class_id} {xcentral} {ycentral} {width} {height}"
            lines.append(line)

        # concatena todas as linhas do arquivo TXT em uma string separadas pela quebra de linha
        return "\n".join(lines)


   
    @staticmethod
    def create_yaml_content(output_dir : str | Path, 
                            train_fold_path : str | Path, 
                            class_ids : Iterable[int], 
                            class_labels : Iterable[str],
                            task : Literal['segment', 'detect'],
                            val_fold_path : str | Path = None, 
                            test_fold_path : str | Path = None):
        
        header_content = f"path: {output_dir}\ntrain: {train_fold_path}\n"
        header_content += f"val: {val_fold_path}\n" if val_fold_path is not None else ""
        header_content += f"test: {val_fold_path}\n" if test_fold_path is not None else ""
        header_content += f"task : {task}\n\nnames:\n"
        classes_content = "".join([f"   {class_id}: {class_label}\n" for class_id, class_label in zip(class_ids, class_labels)])

        return header_content + classes_content
    
    @staticmethod
    def save_file(content : str,  path : str | Path):
        with open(path, 'w') as file: 
            file.write(content)


class ICDARYOLOConverter:
    # Deixei a classe como estática 
    class_id = 0
    class_label = 'cell'

    @staticmethod
    def process_masks(masks : Iterable[Iterable[Tuple[float]]], 
                      image_width : int, 
                      image_height : int,
                      path_txt_file : str | Path):
        
        normalized_masks = YOLOConverter.normalize_masks(masks, image_width, image_height)
        class_ids = len(normalized_masks) * [ICDARYOLOConverter.class_id]
        txt_file_content = YOLOConverter.create_mask_txt_file_content(normalized_masks, class_ids)
        YOLOConverter.save_file(txt_file_content, path_txt_file)

        return normalized_masks

    @staticmethod
    def create_yaml(output_dir : str | Path, 
                    train_fold_path : str | Path, 
                    val_fold_path : str | Path):
    

        yaml_path = os.path.join(output_dir, "dataset.yaml")
        yaml_content = YOLOConverter.create_yaml_content(
            output_dir = output_dir, 
            train_fold_path = train_fold_path,
            val_fold_path = val_fold_path,
            class_ids = [ICDARYOLOConverter.class_id],
            class_labels = [ICDARYOLOConverter.class_label],
            task = 'segment'
        )

        YOLOConverter.save_file(yaml_content, yaml_path)

        print(f"\nYAML criado em: {yaml_path}")
        return yaml_path



