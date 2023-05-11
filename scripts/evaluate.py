import time
import torch
from models.yolo import YOLO
from models.ssd import SSD
from models.frcnn import FasterRCNN
from data_loader.data_loaders import get_dataloader
from sklearn.metrics import classification_report
import numpy as np
from profiling_tools import GPUProfiler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize GPU Profiler
profiler = GPUProfiler(device)

def evaluate_model(model, dataloader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Measure GPU utilization before inference
            profiler.start()

            # Record inference start time
            start_time = time.time()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Record inference end time
            end_time = time.time()

            # Measure GPU utilization after inference
            profiler.stop()

            # Print inference time and GPU utilization
            print("Inference time: {:.2f}s".format(end_time - start_time))
            print("GPU utilization: {:.2f}%".format(profiler.get_utilization()))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))

def main():
    # Load the baseline models
    yolo_model = YOLO()  # Initialize the model structure
    yolo_model.load_state_dict(torch.load('trained_models/yolo_model.pt'))  # Load the trained weights
    yolo_model.to(device)

    ssd_model = SSD()
    ssd_model.load_state_dict(torch.load('trained_models/ssd_model.pt'))
    ssd_model.to(device)

    frcnn_model = FasterRCNN()
    frcnn_model.load_state_dict(torch.load('trained_models/frcnn_model.pt'))
    frcnn_model.to(device)

    # Load the optimized models
    yolo_model_quantized = torch.quantization.quantize_dynamic(
        yolo_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    yolo_model_quantized.load_state_dict(torch.load('optimized_models/yolo_model_quantized.pt'))
    yolo_model_quantized.to(device)

    ssd_model_quantized = torch.quantization.quantize_dynamic(
        ssd_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ssd_model_quantized.load_state_dict(torch.load('optimized_models/ssd_model_quantized.pt'))
    ssd_model_quantized.to(device)

    frcnn_model_quantized = torch.quantization.quantize_dynamic(
        frcnn_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    frcnn_model_quantized.load_state_dict(torch.load('optimized_models/frcnn_model_quantized.pt'))
    frcnn_model_quantized.to(device)

    # Load pruned models
    yolo_model_pruned = torch.load('optimized_models/yolo_model_pruned.pt')
    yolo_model_pruned.to(device)

    ssd_model_pruned = torch.load('optimized_models/ssd_model_pruned.pt')
    ssd_model_pruned.to(device)

    frcnn_model_pruned = torch.load('optimized_models/frcnn_model_pruned.pt')
    frcnn_model_pruned.to(device)

    # Load compressed models
    yolo_model_compressed = torch.load('optimized_models/yolo_model_compressed.pt')
    yolo_model_compressed.to(device)

    ssd_model_compressed = torch.load('optimized_models/ssd_model_compressed.pt')
    ssd_model_compressed.to(device)

    frcnn_model_compressed = torch.load('optimized_models/frcnn_model_compressed.pt')
    frcnn_model_compressed.to(device)

    # Load the test data
    test_dataloader_yolo = get_dataloader('test', 'yolo')
    test_dataloader_ssd = get_dataloader('test', 'ssd')
    test_dataloader_frcnn = get_dataloader('test', 'frcnn')

    # Evaluate the baseline models
    print("Evaluating YOLO model...")
    evaluate_model(yolo_model, test_dataloader_yolo)

    print("Evaluating SSD model...")
    evaluate_model(ssd_model, test_dataloader_ssd)

    print("Evaluating Faster R-CNN model...")
    evaluate_model(frcnn_model, test_dataloader_frcnn)

    # Evaluate the optimized models
    print("Evaluating quantized YOLO model...")
    evaluate_model(yolo_model_quantized, test_dataloader_yolo)

    print("Evaluating quantized SSD model...")
    evaluate_model(ssd_model_quantized, test_dataloader_ssd)

    print("Evaluating quantized Faster R-CNN model...")
    evaluate_model(frcnn_model_quantized, test_dataloader_frcnn)

    # Evaluate the pruned models
    print("Evaluating pruned YOLO model...")
    evaluate_model(yolo_model_pruned, test_dataloader_yolo)

    print("Evaluating pruned SSD model...")
    evaluate_model(ssd_model_pruned, test_dataloader_ssd)

    print("Evaluating pruned Faster R-CNN model...")
    evaluate_model(frcnn_model_pruned, test_dataloader_frcnn)

    # Evaluate the compressed models
    print("Evaluating compressed YOLO model...")
    evaluate_model(yolo_model_compressed, test_dataloader_yolo)

    print("Evaluating compressed SSD model...")
    evaluate_model(ssd_model_compressed, test_dataloader_ssd)

    print("Evaluating compressed Faster R-CNN model...")
    evaluate_model(frcnn_model_compressed, test_dataloader_frcnn)

if __name__ == '__main__':
    main()
