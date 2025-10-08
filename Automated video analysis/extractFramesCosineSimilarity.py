import cv2
import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, model):
        self.model = model
        loaded = np.load('blank.npz')
        self.embeddings = [loaded[key] for key in loaded.files]
        embeddings_array = np.stack(self.embeddings, axis=0) 
        self.embeddingsNorm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8 
        
    def add_embedding(self, frame):
        image_tensor = preprocess_image(frame)
        embedding = extract_embedding(self.model, image_tensor)
        self.embeddings.append(embedding.numpy().astype(np.float32))  
        self.embeddings = self.embeddings[-1000:].copy()
        embeddings_array = np.stack(self.embeddings, axis=0) 
        self.embeddingsNorm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8 
        
    def compute_max_cosine_similarity(self, frame):
        if len(self.embeddings) > 0:
            query_tensor = preprocess_image(frame)
            query_embedding = extract_embedding(self.model, query_tensor).numpy().astype(np.float32)
            query_norm = query_embedding / np.linalg.norm(query_embedding + 1e-8)
            similarities = np.dot(self.embeddingsNorm, query_norm)
            max_similarity = np.max(similarities)
            return max_similarity
        else:
            return 0


def load_SqueezeNet():
    model = models.squeezenet1_1(weights="SqueezeNet1_1_Weights.IMAGENET1K_V1")    
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval() 
    return model

def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

def extract_embedding(model, image_tensor):
    with torch.no_grad():
        embedding = model(image_tensor)  
    return embedding.squeeze().flatten() 

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim=0).item()


def extract_frames(video_path, output_folder, embedManager, processedVideos, num_frames=100):
    filename = os.path.basename(video_path).rsplit(".", 1)[0]

    # Check processed files
    if filename in processedVideos:
        print(filename, "processed!")
        return 0
        
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
	
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if video has enough frames
    if total_frames < num_frames:
        print(f"Video {video_path} has fewer frames ({total_frames}) than {num_frames}. Skipping...")
        return
    
    # Calculate the interval between frames to sample
    frame_interval = total_frames // (num_frames)
    
    saved_count = 0
    iteration_count = 0
    while cap.isOpened():
        cap.set(1,  int(round(iteration_count * frame_interval, 0))-10)
        ret, frame = cap.read()
        if not ret:
            #print(total_frames, int(round(iteration_count * frame_interval, 0))-10)
            break  # Exit if video ends
        s = embedManager.compute_max_cosine_similarity(frame)
        print(iteration_count, len(embedManager.embeddings), s)
        if s < 0.6667:
            embedManager.add_embedding(frame)
            frame_filename = os.path.join(output_folder, f"{filename}_{saved_count:03d}.png")
            frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite(frame_filename, frame_resized)  # Save the frame
            saved_count += 1
        iteration_count += 1

        if iteration_count >= num_frames:
            break

    cap.release()
    print(f"Saved {saved_count} frames from {video_path} to {output_folder}.")

# Extract frames
videoFolder = ["/mnt/ilvoPigs/Experiment1/", "/mnt/ilvoPigs/Experiment2/", "/mnt/ilvoPigs/Experiment3/", 
               "/mnt/ilvoPigs/Experiment4/", "/mnt/ilvoPigs/Experiment5/"]
videoList = []
for vF in videoFolder:
        tempVideoList = [vF + f for f in os.listdir(vF) if f.endswith(".mp4") and 9 <= int(f.split('_')[1][:2]) <= 19]
        videoList += tempVideoList

output_dir = "../data/rawFrames"
model = load_SqueezeNet()
embedding_manager = EmbeddingManager(model)
videoProcessSet = set([i[18:-8]for i in sorted(glob.glob("../data/rawFrames/*png"))])

for i in videoList:
    extract_frames(i, output_dir, embedding_manager, videoProcessSet, num_frames=100)

# Checkpoint complete!


