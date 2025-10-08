from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import argparse
import torch
import cv2
import os
import time
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from unet import getModel, getFeatures
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_num_threads(3)

# Flask App
app = Flask(__name__)

def create_app(deviceId):

    # Device Selection
    if deviceId != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(deviceId)
    else:
        device = torch.device("cpu")
    
    if device.type == "cuda" and torch.cuda.get_device_properties(deviceId).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
       
    # Load model and checkpoint 
    sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, 
                      sam2_checkpoint, 
                      device=device, 
                      apply_postprocessing=False)
    
    mask_generator = SAM2AutomaticMaskGenerator(sam2, 
                                                stability_score_thresh=0.75, 
                                                multimask_output=False, 
                                                points_per_side=24, 
                                                pred_iou_thresh=0.9,
                                                box_nms_thresh=0.7,
                                                min_mask_region_area=10)

    # Load Unet Model
    modelUnet = getModel("../output/pigsUNetDistill3/checkpointI525.pth", device)
    
    @app.route('/inferenceSegment', methods=['POST'])
    def processSAM():
        framePath = request.json['framePath']
        image = Image.open(framePath)
        image = image.resize((1024, 1024), Image.Resampling.NEAREST)
        image = np.array(image.convert("RGB"))
        
        with torch.autocast(device.type, dtype=torch.bfloat16): 
            with torch.no_grad():
                masks = mask_generator.generate(image)

        for i in range(len(masks)):
            #del masks[i]['segmentation']
            masks[i]['segmentation'] = masks[i]['segmentation'].tolist()
            del masks[i]['point_coords']
            del masks[i]['crop_box']
            
        return jsonify({'masks': masks})

    @app.route('/inferenceUnet', methods=['POST'])
    def processUnet():
        framePath = request.json['framePath']
        pointMask = getFeatures(modelUnet, framePath, device)
        return jsonify({'unet': pointMask})

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Flask server for model inference.")
    parser.add_argument('--gpu', type=int, default='cpu', help="GPU device number to use (default: cpu)")
    parser.add_argument('--port', type=int, default=5000, help="Port number for the Flask server (default: 5000)")
    args = parser.parse_args()

    # Create the Flask app with the specified GPU device
    app = create_app(args.gpu)

    # Run the server
    app.run(port=args.port, threaded=False)


