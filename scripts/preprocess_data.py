import os
import cv2
from lxml import etree
import shutil
import random
from utility_scripts import read_xml_annotation, create_yolo_annotation, create_ssd_annotation, create_frcnn_annotation

def preprocess_ssd(dataset):
    # SSD preprocessing code
    # Resize and normalize images
    for img_name in os.listdir(input_images_dir):
        img_path = os.path.join(input_images_dir, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (image_size, image_size))
        img_normalized = img_resized / 255.0
        cv2.imwrite(os.path.join(output_images_dir, img_name), img_normalized * 255)

    # Convert annotations to SSD format
    if dataset == "Pascal_VOC":
        annotations_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/annotations'
        yolo_annotations_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/annotations_ssd'
    else:
        annotations_dir = f'../data/{dataset}/annotations'
        yolo_annotations_dir = f'../data/{dataset}/annotations_ssd'

    for ann_name in os.listdir(annotations_dir):
        ann_path = os.path.join(annotations_dir, ann_name)
        tree = etree.parse(ann_path)
        annotations = read_xml_annotation(tree)
        ssd_annotations = create_ssd_annotation(annotations, image_size)
        with open(os.path.join(ssd_annotations_dir, ann_name.replace('.xml', '.txt')), 'w') as f:
            for ann in ssd_annotations:
                f.write(ann + '\n')

def preprocess_yolo(dataset):
    # YOLO preprocessing code
    # Resize and normalize images
    for img_name in os.listdir(input_images_dir):
        img_path = os.path.join(input_images_dir, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (image_size, image_size))
        img_normalized = img_resized / 255.0
        cv2.imwrite(os.path.join(output_images_dir, img_name), img_normalized * 255)

    # Convert annotations to YOLO format
    if dataset == "Pascal_VOC":
        annotations_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/annotations'
        yolo_annotations_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/annotations_yolo'
    else:
        annotations_dir = f'../data/{dataset}/annotations'
        yolo_annotations_dir = f'../data/{dataset}/annotations_yolo'

    for ann_name in os.listdir(annotations_dir):
        ann_path = os.path.join(annotations_dir, ann_name)
        tree = etree.parse(ann_path)
        annotations = read_xml_annotation(tree)
        yolo_annotations = create_yolo_annotation(annotations, image_size)
        with open(os.path.join(yolo_annotations_dir, ann_name.replace('.xml', '.txt')), 'w') as f:
            for ann in yolo_annotations:
                f.write(ann + '\n')

def preprocess_frcnn(dataset):
    # Faster R-CNN preprocessing code
    # Resize and normalize images
    for img_name in os.listdir(input_images_dir):
        img_path = os.path.join(input_images_dir, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (image_size, image_size))
        img_normalized = img_resized / 255.0
        cv2.imwrite(os.path.join(output_images_dir, img_name), img_normalized * 255)

    # Convert annotations to Faster R-CNN format
    if dataset == "Pascal_VOC":
        annotations_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/annotations'
        yolo_annotations_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/annotations_frcnn'
    else:
        annotations_dir = f'../data/{dataset}/annotations'
        yolo_annotations_dir = f'../data/{dataset}/annotations_frcnn'

    for ann_name in os.listdir(annotations_dir):
        ann_path = os.path.join(annotations_dir, ann_name)
        tree = etree.parse(ann_path)
        annotations = read_xml_annotation(tree)
        frcnn_annotations = create_frcnn_annotation(annotations, image_size)
        with open(os.path.join(frcnn_annotations_dir, ann_name.replace('.xml', '.txt')), 'w') as f:
            for ann in frcnn_annotations:
                f.write(ann + '\n')

if __name__ == '__main__':
    # Define the dataset you want to preprocess, either 'COCO' or 'Pascal_VOC'
    dataset = 'Pascal_VOC'

    # Define the image_size depending on the chosen model
    image_size = 416  # for YOLO, for example

    # Set input_images_dir and output_images_dir based on the dataset
    if dataset == 'COCO':
        input_images_dir = '../data/COCO/train2017'
        output_images_dir = f'../data/COCO/train2017_preprocessed'
    elif dataset == 'Pascal_VOC':
        input_images_dir = '../data/Pascal_VOC/VOCdevkit/VOC2007/JPEGImages'
        output_images_dir = f'../data/Pascal_VOC/VOCdevkit/VOC2007/JPEGImages_preprocessed'
    else:
        raise ValueError("Invalid dataset name. Choose either 'COCO' or 'Pascal_VOC'")

    # Create output_images_dir if it doesn't exist
    os.makedirs(output_images_dir, exist_ok=True)

    # Call the appropriate preprocessing function based on your chosen model
    preprocess_yolo(dataset)
