import torch
import gzip
import shutil
from models.yolo import YOLO
from models.ssd import SSD
from models.frcnn import FasterRCNN

def compress_model(model, model_name):
    # Ensure the model is in evaluation mode
    model.eval()

    # Save the model
    torch.save(model.state_dict(), f'trained_models/{model_name}_model.pt')

    # Compress the model
    with open(f'trained_models/{model_name}_model.pt', 'rb') as f_in:
        with gzip.open(f'optimized_models/{model_name}_model.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def main():
    # Load the original models
    yolo_model = YOLO()  # Initialize the model structure
    yolo_model.load_state_dict(torch.load('trained_models/yolo_model.pt'))  # Load the trained weights

    ssd_model = SSD()
    ssd_model.load_state_dict(torch.load('trained_models/ssd_model.pt'))

    frcnn_model = FasterRCNN()
    frcnn_model.load_state_dict(torch.load('trained_models/frcnn_model.pt'))

    # Compress the models
    compress_model(yolo_model, 'yolo')
    compress_model(ssd_model, 'ssd')
    compress_model(frcnn_model, 'frcnn')

if __name__ == '__main__':
    main()
