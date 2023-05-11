import torch
import torch.nn.utils.prune as prune
from models.yolo import YOLO
from models.ssd import SSD
from models.frcnn import FasterRCNN

def prune_model(model, model_name):
    # Ensure the model is in evaluation mode
    model.eval()

    # Prune the model
    parameters_to_prune = (
        (module, 'weight') for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    # Remove the pruned parameters and make the remaining ones permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    # Save the pruned model
    torch.save(model.state_dict(), f'optimized_models/{model_name}_pruned.pt')

def main():
    # Load the original models
    yolo_model = YOLO()  # Initialize the model structure
    yolo_model.load_state_dict(torch.load('trained_models/yolo_model.pt'))  # Load the trained weights

    ssd_model = SSD()
    ssd_model.load_state_dict(torch.load('trained_models/ssd_model.pt'))

    frcnn_model = FasterRCNN()
    frcnn_model.load_state_dict(torch.load('trained_models/frcnn_model.pt'))

    # Prune the models
    prune_model(yolo_model, 'yolo')
    prune_model(ssd_model, 'ssd')
    prune_model(frcnn_model, 'frcnn')

if __name__ == '__main__':
    main()
