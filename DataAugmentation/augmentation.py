import numpy as np
from PIL import Image
import albumentations as A

from pathlib import Path 
from typing import Iterable, Tuple

class Augmentation: 

    def resize_mask(mask, 
                    original_width, 
                    new_width, 
                    original_height, 
                    new_height):
        
        new_xy_points = []
        for xy_point in mask:
            x, y = xy_point
            x_new = float(np.round(x * new_width / original_width, 0))
            y_new = float(np.round(y * new_height / original_height, 0))
            
            # adicionado como lista para facilitar a conversão JSON ao ler o CSV
            new_xy_points.append([x_new, y_new])

        return new_xy_points
    
    @staticmethod
    def resize_masks(masks, 
                     original_width, 
                     new_width, 
                     original_height, 
                     new_height):
        
        new_masks = [] 

        for mask in masks:
            new_xy_points = Augmentation.resize_mask(
                mask,
                original_width, 
                new_width, 
                original_height, 
                new_height 
            )
            new_masks.append(new_xy_points)

        return new_masks
    
    
    @staticmethod
    def resize_image(image : np.array,
                     new_width : int,
                     new_height: int,
                     masks : Iterable[Iterable[Tuple[float]]] = None,
                     **kwargs) -> np.array:
        
        if isinstance(image, Image.Image):
            image = np.asarray(image)
        
        if masks is not None:
            image_shape = image.shape
            
            # Para contabilizar imagens com canais de cor
            if len(image_shape) > 2:
                height, width, _ = image_shape
            else:
                # Atentar para o fato que o numpy inverte
                height, width = image_shape
            
            masks = Augmentation.resize_masks(masks, 
                                              width, new_width,
                                              height, new_height)

        transform = A.Resize(width=new_width, height=new_height, **kwargs)
        new_image = transform(image=image)['image']
        
        return new_image, masks 
    
    @staticmethod
    def save_image(image : np.array, path : str | Path):
        image_pil = Image.fromarray(image)
        image_pil.save(path)
