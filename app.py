import streamlit as st
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load the pre-trained VGG16 model
@st.cache_resource
def load_model():
    return VGG16(weights="imagenet", include_top=False)

model = load_model()

# Function to extract features from an image
def extract_features(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    features = model.predict(image_array)
    return features.flatten()

# Function to find the top similar images
def find_top_similar_images(reference_image_path, dataset_image_paths, model, top_k=5, similarity_threshold=0.2):
    # Extract features for the reference image
    reference_features = extract_features(reference_image_path, model)
    
    # Extract features for the dataset images
    dataset_features = [extract_features(img_path, model) for img_path in dataset_image_paths]
    
    # Compute cosine similarity
    similarities = cosine_similarity([reference_features], dataset_features)[0]
    
    # Get the indices of the top K similar images, only including those with similarity > threshold
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_similar_images = [(dataset_image_paths[i], similarities[i]) for i in top_indices if similarities[i] >= similarity_threshold]
    
    return top_similar_images

# Streamlit App UI
st.title("Image Similarity Finder")
st.write("Upload an image and a dataset folder to find the most similar images.")

# Upload reference image
reference_image = st.file_uploader("Upload a reference image:", type=["jpg", "png", "jpeg"])

# Check if reference image is uploaded
if reference_image:
    st.image(reference_image, caption="Reference Image", width=200)  # Reduced input image size

    # Save the uploaded reference image temporarily
    with open("temp_reference.jpg", "wb") as f:
        f.write(reference_image.getbuffer())
    
    # Enter dataset directory path
    dataset_dir = st.text_input("Enter the dataset folder path:", r"C:\Users\PMLS\Downloads\New folder\downloaded_images")
    
    # Check if dataset directory exists
    if dataset_dir and os.path.exists(dataset_dir):
        dataset_image_paths = [
            os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir) if img.lower().endswith(("jpg", "png", "jpeg"))
        ]
        
        # Check if dataset contains images
        if dataset_image_paths:
            with st.spinner("Processing..."):
                # Find the top 5 similar images, excluding those with similarity below 0.2
                top_similar_images = find_top_similar_images("temp_reference.jpg", dataset_image_paths, model, top_k=5, similarity_threshold=0.2)
            
            # Display the results in a row
            if top_similar_images:
                st.success("Top similar images found!")
                columns = st.columns(5)  # Create 5 columns for the images
                for col, (img_path, score) in zip(columns, top_similar_images):
                    with col:
                        st.image(img_path, caption=f"Score: {score:.4f}", width=150)  # Display image with a smaller width
            else:
                st.warning("No similar images found with a similarity score above 0.2.")
        else:
            st.error("No images found in the specified dataset directory.")
    elif dataset_dir:
        st.error("The specified dataset folder does not exist.")
