import streamlit as st
import os
import glob
import shutil
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch 

torch.cuda.empty_cache()
torch.cuda.set_device(1)
torch.cuda.current_device()

# Directories
IMAGE_DIR = "./pigsCutSegment" #"./pigsAnnotatedSegment/None" 
TARGET_BASE_DIR = "./pigsAnnotatedSegment"
FOLDERS = [str(i) for i in range(1, 23)] + ["None"]

# Load the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()
model_rn = torch.load("max_val_accuracyV3.pth", map_location=device, weights_only=False)
model_rn.to(device)
model_rn.eval()

# Ensure folders 1-21 exist
for folder in FOLDERS:
    os.makedirs(os.path.join(TARGET_BASE_DIR, folder), exist_ok=True)

# Load images
def get_images():
    return sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png"))) + \
           sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))) + \
           sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))

# Move image to selected folder
def move_image(image_path, target_folder):
    target_path = os.path.join(TARGET_BASE_DIR, target_folder, os.path.basename(image_path))
    shutil.move(image_path, target_path)

# Get count of files in numbered folders
def get_folder_counts():
    counts = {folder: len([i for i in os.listdir(os.path.join(TARGET_BASE_DIR, folder)) if i.endswith('png')]) for folder in FOLDERS}
    return counts, sum(counts.values())

# Model prediction function
def predict(image_path):
    transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image)
    image = image.unsqueeze(0) 
    image = image.cuda()
    output = model_rn(image)
    output = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(output, 1)
    probabilities = [round(float(output[0][i]) ,3) for i in range(1, 22)]  + [0.0, round(float(output[0][0]) ,2)]
    predictedChoice = None
    if 0 < int(pred):
        predictedChoice = int(pred) - 1
    else:
        predictedChoice = 22
    return probabilities, predictedChoice

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Pig Stripe Annotation App")

# Load images
images = get_images()
if images:
    image_path = images[0]  # Get the first image in the sorted list
    st.image(image_path, caption=os.path.basename(image_path), width=300)  # Fixed image size
    probabilities, predicted_choice = predict(image_path)
    print(list(enumerate(probabilities)))
    folderProb = [FOLDERS[idx] + " - " + str(key) for idx, key in enumerate(probabilities)]

    # Sidebar for classification
    choice = st.sidebar.radio("Choose a folder to classify:", folderProb, index=predicted_choice)
    choice = choice.split(" ")[0]
    #occluded = st.sidebar.checkbox("Occluded")

    if st.sidebar.button("Classify"):
        #base, ext = os.path.splitext(os.path.basename(image_path))
        #if occluded:
        #    base += "_occluded"
        #new_filename = base + ext
        move_image(image_path, choice)
        st.rerun()  # Reload the app to update the image list

else:
    st.write("### No images left to classify.")

# Show folder counters at the bottom
st.subheader("Classification Summary")
counts, total = get_folder_counts()
st.write("  |  ".join([f"{folder}: {count}" for folder, count in counts.items()]))
st.write(f"**Total Classified Images:** {total}")

