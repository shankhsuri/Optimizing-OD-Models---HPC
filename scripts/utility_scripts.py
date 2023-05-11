from lxml import etree

def read_xml_annotation(tree):
    """
    Read the input XML file and parse the bounding box and class label information.

    Args:
        tree (etree.ElementTree): The parsed XML tree.

    Returns:
        list: A list of dictionaries containing bounding box and class label information.
    """
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    annotations = []
    for obj in root.findall('object'):
        class_label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append({'class_label': class_label, 'bbox': (xmin, ymin, xmax, ymax)})

    return annotations


def create_yolo_annotation(annotations, image_size):
    """
    Convert bounding box coordinates and class labels to the YOLO format.

    Args:
        annotations (list): A list of dictionaries containing bounding box and class label information.
        image_size (int): The size of the resized image (width and height are assumed to be equal).

    Returns:
        list: A list of strings with YOLO-formatted annotations.
    """
    yolo_annotations = []
    for ann in annotations:
        class_label = ann['class_label']
        bbox = ann['bbox']
        x_center = (bbox[0] + bbox[2]) / 2 / image_size
        y_center = (bbox[1] + bbox[3]) / 2 / image_size
        width = (bbox[2] - bbox[0]) / image_size
        height = (bbox[3] - bbox[1]) / image_size
        yolo_annotations.append(f'{class_label} {x_center} {y_center} {width} {height}')

    return yolo_annotations


def create_ssd_annotation(annotations, image_size):
    """
    Convert bounding box coordinates and class labels to the SSD format.

    Args:
        annotations (list): A list of dictionaries containing bounding box and class label information.
        image_size (int): The size of the resized image (width and height are assumed to be equal).

    Returns:
        list: A list of strings with SSD-formatted annotations.
    """
    ssd_annotations = []
    for ann in annotations:
        class_label = ann['class_label']
        bbox = ann['bbox']
        xmin = bbox[0] / image_size
        ymin = bbox[1] / image_size
        xmax = bbox[2] / image_size
        ymax = bbox[3] / image_size
        ssd_annotations.append(f'{class_label} {xmin} {ymin} {xmax} {ymax}')

    return ssd_annotations


def create_frcnn_annotation(annotations):
    """
    Convert bounding box coordinates and class labels to the Faster R-CNN format.

    Args:
        annotations (list): A list of dictionaries containing bounding box and class label information.

    Returns:
        list: A list of strings with Faster R-CNN-formatted annotations.
    """
    frcnn_annotations = []
    for ann in annotations:
        class_label = ann['class_label']
        bbox = ann['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        frcnn_annotations.append(f'{xmin} {ymin} {xmax} {ymax} {class_label}')

    return frcnn_annotations
