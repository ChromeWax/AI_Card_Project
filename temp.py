from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np

# Used to create unique colors for each class
from distinctipy import distinctipy

# Import PIL for image manipulation
from PIL import Image, ImageDraw

# Import pytorch dependencies
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2  as transforms
from torchvision.transforms.v2 import functional as TF

# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def create_polygon_mask(image_size, vertices):
    mask_image = Image.new('L', image_size, 0)
    ImageDraw.Draw(mask_image, 'L').polygon(vertices, fill=(255))
    return mask_image

def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def tensor_to_pil(tensor):
    transform = transforms.ToPILImage()
    img = transform(tensor)
    return img

# Path for dataset
dataset_path = Path("./Images")
image_file_paths = list(dataset_path.glob("*.jpg"))
annotation_file_paths = list(dataset_path.glob("*.json"))

# Name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

# Gets device for training
device = get_torch_device()
dtype = torch.float32

# Dictionary to map each file name to path
image_dict = {file.stem : file for file in image_file_paths}

# Dataframe for annotations
cls_dataframes = (pd.read_json(file_path, orient='index').transpose() for file_path in annotation_file_paths)
annotation_df = pd.concat(cls_dataframes, ignore_index=False)
annotation_df['index'] = annotation_df.apply(lambda row: row['imagePath'].split('.')[0], axis=1)
annotation_df = annotation_df.set_index('index')

# Dataframe for segmentations of annotations
shapes_df = annotation_df['shapes'].explode().to_frame().shapes.apply(pd.Series)

# Gets unique list of classes, in this case just one
class_names = shapes_df['label'].unique().tolist()

# The Mask R-CNN model expects datasets to have a background class
class_names = ['background'] + class_names

# Generate a list of colors with a length equal to the number of labels
colors = distinctipy.get_colors(len(class_names))
int_colors = [tuple(int(c*255) for c in color) for color in colors]

draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2, font=font_file, font_size=25)

# Get the file ID of the first image file
file_id = list(image_dict.keys())[56]
# Open the associated image file as a RGB image
sample_img = Image.open(image_dict[file_id]).convert('RGB')
# Extract the labels for the sample
labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
# Extract the polygon points for segmentation mask
shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
# Format polygon points for PIL
xy_coords = [[tuple(p) for p in points] for points in shape_points]
# Generate mask images from polygons
mask_imgs = [create_polygon_mask(sample_img.size, xy) for xy in xy_coords]
# Convert mask images to tensors
masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
# Generate bounding box annotations from segmentation masks
bboxes = torchvision.ops.masks_to_boxes(masks)

# Annotate the sample image with segmentation masks
annotated_tensor = draw_segmentation_masks(
    image=transforms.PILToTensor()(sample_img), 
    masks=masks, 
    alpha=0.3, 
    colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
)

# Annotate the sample image with labels and bounding boxes
annotated_tensor = draw_bboxes(
    image=annotated_tensor, 
    boxes=bboxes, 
    labels=labels, 
    colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
)

img = tensor_to_pil(annotated_tensor)
img.show()

# Initialize a Mask R-CNN model with pretrained weights
model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

# Get the number of input features for the classifier
in_features_box = model.roi_heads.box_predictor.cls_score.in_features
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

# Get the numbner of output channels for the Mask Predictor
dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

# Replace the box predictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=len(class_names))

# Replace the mask predictor
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=len(class_names))

# Set the model's device and data type
model.to(device=device, dtype=dtype);

# Add attributes to store the device and model name for later reference
model.device = device
model.name = 'maskrcnn_resnet50_fpn_v2'