import cv2
from typing import Literal

icdar_color_class_map = {
    'cell': (255, 0, 0)
}

fintab_color_class_map = {
    'table': (214, 39, 40),
    'table column': (31, 119, 180),
    'table row': (140, 86, 75),
    'table column header': (255, 127, 14),
    'table projected row header' : (227, 119, 194),
    'table spanning cell': (127, 127, 127)
}

def draw_bouding_box(image, 
                     xmin, ymin, xmax, ymax, 
                     bbox_color = (255,0,0), 
                     bbox_alpha = None,
                     bbox_thickness = 2,
                     bbox_line_type = cv2.LINE_AA, #anti-alising
                     bbox_class = None,
                     font_origin = None,
                     font = cv2.FONT_HERSHEY_SIMPLEX, 
                     font_scale = 0.7,
                     font_color = (0,0,0),                      
                     font_thickness = 2,
                     font_line_type = cv2.LINE_AA):
    
    copied_image = image.copy()
    top_left_corner = (xmin, ymin)
    bottom_right_corner = (xmax, ymax)

    if bbox_alpha is not None: 
        bbox_thickness = -1
        copied_image = cv2.rectangle(copied_image, top_left_corner, bottom_right_corner, bbox_color, bbox_thickness)
        copied_image = cv2.addWeighted(copied_image, bbox_alpha, image, 1 - bbox_alpha, 0)

    
    copied_image = cv2.rectangle(copied_image, top_left_corner, bottom_right_corner, bbox_color, bbox_thickness, bbox_line_type)
    
    if bbox_class is not None:
        bottom_left_font_corner = (xmin, ymin-2) if font_origin is None else font_origin
        copied_image = cv2.putText(copied_image, bbox_class, bottom_left_font_corner,
                                   font, font_scale, font_color, font_thickness, font_line_type)

    return copied_image

