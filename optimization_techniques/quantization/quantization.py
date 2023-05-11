import torch
import torch.quantization
from models.yolo import YOLO
from models.ssd import SSD
from models.frcnn import FasterRCNN

def quantize_model(model, model_name):
    # Ensure the model is in evaluation mode
    model.eval()

    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Save the quantized model
    torch.save(quantized_model.state_dict(), f'optimized_models/{model_name}_quantized.pt')

def main():
    # Load the original models
    yolo_model = YOLO()  # Initialize the model structure
    yolo_model.load_state_dict(torch.load('trained_models/yolo_model.pt'))  # Load the trained weights

    ssd_model = SSD()
    ssd_model.load_state_dict(torch.load('trained_models/ssd_model.pt'))

    frcnn_model = FasterRCNN()
    frcnn_model.load_state_dict(torch.load('trained_models/frcnn_model.pt'))

    # Quantize the models
    quantize_model(yolo_model, 'yolo')
    quantize_model(ssd_model, 'ssd')
    quantize_model(frcnn_model, 'frcnn')

if __name__ == '__main__':
    main()
