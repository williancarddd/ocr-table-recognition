import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from glob import glob

import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple
from tqdm import tqdm

from .file_finder import FileFinder


class ConvertICDARDatasetToDataframe:

    def __init__(self, 
                 images_path : str | Path, 
                 labels_path : str | Path,
                 class_id : int = 0, 
                 class_label : str = 'cell'):
        
        self.images_path = images_path
        self.labels_path = labels_path

        self.class_id = class_id
        self.class_label = class_label
            
        self.image_files = self.find_files(images_path, format_list=['png', 'jpg', 'TIFF'])
        self.label_files = self.find_files(labels_path, format_list=['xml', 'txt'])

        self.pairs_image_label = self.associate_pairs()

    def get_image_shape(self, image_path):
        image = Image.open(image_path)
        image_width, image_height = image.size

        return image_width, image_height
    
    def find_files(self, dir_path, format_list):
        return FileFinder.find_files(dir_path=dir_path, format_list=format_list)

    
    def associate_pairs(self):
        return FileFinder.associate_files_by_name(self.image_files, self.label_files)
    
    
    def get_xy_annotations_from_xml(self, xml_path : str | Path = None):
        '''
        Acessa os arquivos XML, extraindo as anotações no formato XY.
        '''
        
        tree = ET.parse(xml_path) # abre o arquivo XML
        root = tree.getroot() # encontra a raiz
        
        lines = []
        # para cada tag table
        for table in root.findall("table"):
            # para cada tag cell
            for cell in table.findall("cell"):
                # encontra as coordenadas da célula
                coords = cell.find("Coords")
                if coords is None:
                    continue             
                # recupera os pontos individuais 
                points = coords.attrib["points"]
                # converte os pontos para uma lista de tuplas de pares x, y
                points = [list(map(float, p.split(",")))  for p in  points.split()]
                # adiciona a linnha na lista de linhas
                lines.append(points)  
                                 
        return lines
    
    def generate_dataframe(self):

        print('Gerando DataFrame do conjnuto de dados...')
        df_pairs = pd.DataFrame(self.pairs_image_label, columns=['image_path', 'label_path'])

        images_shape = []
        for image_path in tqdm(df_pairs['image_path'], desc= 'Extraindo dimensão das imagens...'):
            images_shape.append(self.get_image_shape(image_path))
    
        temp_df = pd.DataFrame(images_shape, columns=['image_width', 'image_height'])
        df_pairs = pd.concat([df_pairs, temp_df], axis=1)
    
        xy_annotations = [] 
        for xml_path in tqdm(df_pairs['label_path'], desc='Extraindo as anotações...'):
            xy_annotations.append(self.get_xy_annotations_from_xml(xml_path)) 
        
        
        df_pairs['class_id'] = self.class_id
        df_pairs['class_label'] = self.class_label
        df_pairs['xy'] = xy_annotations

        return df_pairs
