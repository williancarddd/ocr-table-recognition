import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import albumentations as A

import shutil
from pathlib import Path 
from typing import Iterable, Tuple, Any

class Augmentation: 

    @staticmethod
    def is_pil_image(image : Any):
        return isinstance(image, Image.Image)
    
    
    @staticmethod
    def is_ndarray(image : Any):
        return isinstance(image, np.ndarray)
    
    @staticmethod
    def add_salt_and_pepper_noise(image : Image.Image | np.ndarray, 
                                  amount_percentage : float = 0.05, 
                                  salt_vs_pepper_percentage : float =0.5,
                                  return_ndarray : bool = False,
                                  seed : int = None):
        
        random_generator = np.random.default_rng(seed=seed)
        
        if Augmentation.is_pil_image(image):
            image = np.array(image)

        noisy_image = image.copy()
        height, width = image.shape[:2]

        noisy_pixel_amount =  int(amount_percentage * height * width)

        salt_amount = int( salt_vs_pepper_percentage * noisy_pixel_amount )
        pepper_amount = noisy_pixel_amount - salt_amount
        salt_peper_amounts = {'salt': salt_amount, 'peper':pepper_amount}

        for noise_type, noise_amount in salt_peper_amounts.items():
            for _ in range(noise_amount):
                x = random_generator.integers(0, width-1)
                y = random_generator.integers(0, height-1)
                
                noisy_image[y,x] = 255 if noise_type == 'salt' else 0 
          
        if not return_ndarray:
            noisy_image = Image.fromarray(noisy_image)
        
        return noisy_image


    @staticmethod
    def add_gaussian_blur(image : Image.Image | np.ndarray, 
                          amount : int = 3,
                          return_ndarray = False,
                          **kwargs):
        
        if Augmentation.is_ndarray(image):
            image = Image.fromarray(image)


        new_image = image.filter(ImageFilter.GaussianBlur(radius=amount, **kwargs))

        if return_ndarray:
            new_image = np.array(new_image)

        return new_image
    
    @staticmethod
    def set_brightness(image : Image.Image | np.ndarray, 
                       factor : float = 1.5,
                       return_ndarray = False,
                       **kwargs):
        
        if Augmentation.is_ndarray(image):
            image = Image.fromarray(image)

        enhancer = ImageEnhance.Brightness(image)
        new_image = enhancer.enhance(factor, **kwargs)

        if return_ndarray:
            new_image = np.array(new_image)

        return new_image
    
    
    @staticmethod
    def resize_mask(mask, 
                    original_width, 
                    new_width, 
                    original_height, 
                    new_height):
        
        new_xy_points = []
        for xy_point in mask:
            x, y = xy_point
            x_new = int(x * new_width / original_width)
            y_new = int(y * new_height / original_height)
            
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
    def resize_image(image : Image.Image | np.ndarray,
                 new_width : int,
                 new_height: int,
                 masks : Iterable[Iterable[Tuple[float]]] = None,
                 return_ndarray : bool = False,
                 **kwargs) -> np.array:
    
        if isinstance(image, np.ndarray): 
            image = Image.fromarray(image)
        
        if masks is not None:
            width, height = image.size
            
            masks = Augmentation.resize_masks(
                masks, 
                width, new_width,
                height, new_height
            )

        new_image = image.resize((new_width, new_height))

        if return_ndarray:
            new_image = np.array(new_image)
        
        return new_image, masks
    
    
    @staticmethod
    def apply_albumentation_tranform(image : Image.Image | np.ndarray, 
                                     transform : A.BasicTransform, 
                                     **kwargs) -> dict:
        
        if Augmentation.is_pil_image(image):
            image = np.array(image)

        transformed_output = transform(image=image, **kwargs)

        return transformed_output


    @staticmethod
    def save_image(image : Image.Image | np.ndarray, path : str | Path):
        
        if Augmentation.is_ndarray(image):
            image = Image.fromarray(image)
        
        image.save(path)