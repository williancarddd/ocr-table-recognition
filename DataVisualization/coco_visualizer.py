import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io

# Initialize COCO API for annotations
annFile = '/home/williancarddd/Projects/workspace-usp/ocr-table-recognition/dataset_rcnn/fold_1/train.json'
coco = COCO(annFile)

# Load an image
img_id = coco.getImgIds()[0]
img_info = coco.loadImgs(img_id)[0]
I = io.imread('/home/williancarddd/Projects/workspace-usp/ocr-table-recognition/dataset_rcnn/fold_1/train/' + img_info['file_name'])

# Display image
plt.imshow(I)
plt.axis('off')

# Load and display annotations
annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

plt.show()