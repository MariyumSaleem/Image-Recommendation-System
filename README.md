# Image-Based Recommendation System

This project implements an **Image-Based Recommendation System** using **Streamlit** and **Keras** with the VGG16 pre-trained model. The application recommends the most visually similar images from a dataset based on a user-uploaded reference image. 

---

## Features

1. **Feature-Based Recommendations**:
   - Utilizes the **VGG16** deep learning model (pre-trained on ImageNet) to extract feature embeddings from images.
   - Recommends images based on cosine similarity between the reference image's features and dataset images' features.

2. **Dynamic Filtering**:
   - Displays the most similar images with similarity scores above a configurable threshold (default: 0.2).

3. **Interactive User Interface**:
   - Upload a reference image.
   - Specify the path to a dataset of images.
   - Visualize the top `k` recommended images directly within the app.

---

## Prerequisites

### Libraries
Ensure the following libraries are installed:

- **Streamlit**: For creating the web app interface.
- **Keras/TensorFlow**: For using the pre-trained VGG16 model.
- **NumPy**: For numerical computations.
- **scikit-learn**: For calculating cosine similarity.
- **Pillow (PIL)**: For image loading and processing.

Install these libraries using:
```bash
pip install streamlit tensorflow numpy scikit-learn pillow
```

---

## How It Works

1. **Feature Extraction**:
   - The pre-trained **VGG16** model extracts features from images.
   - Images are resized to `224x224`, as required by VGG16.
   - Extracted features are flattened into a vector representation.

2. **Recommendation Algorithm**:
   - **Cosine Similarity** is computed between the reference image's feature vector and those of the dataset images.
   - Top `k` images with the highest similarity scores above the threshold are recommended.

3. **Results Display**:
   - Recommended images are displayed in a grid with their similarity scores.

---

## Usage

1. **Clone or Save the Code**:
   - Save the code to a file named, for example, `image_recommendation_app.py`.

2. **Run the Application**:
   - Open a terminal and execute:
     ```bash
     streamlit run image_recommendation_app.py
     ```

3. **Use the App**:
   - **Step 1**: Upload a reference image in `.jpg`, `.png`, or `.jpeg` format.
   - **Step 2**: Enter the path to a folder containing dataset images.
   - **Step 3**: View the top recommended images, complete with similarity scores.

---

## Directory Structure

```plaintext
project/
├── image_recommendation_app.py  # Main application code
├── temp_reference.jpg           # Temporarily saved uploaded image (auto-created)
└── dataset_folder/              # Your dataset of images
```

---

## Customization

- **Top `k` Recommendations**:
   - Default: Top 5 recommendations.
   - Adjust by modifying the `top_k` parameter in the `find_top_similar_images` function.

- **Similarity Threshold**:
   - Default: 0.2 (only images with a similarity score above this are recommended).
   - Adjust by modifying the `similarity_threshold` parameter.

---

## Applications

- **E-Commerce**: Recommend visually similar products (e.g., clothes, furniture).
- **Photo Libraries**: Help users organize or find related images.
- **Art Galleries**: Suggest artworks that are visually similar.
- **Content Creation**: Aid in finding inspiration based on uploaded images.

---

## Screenshots of the App

 <img width="727" alt="1" src="https://github.com/user-attachments/assets/dc98e9c5-65ba-4ed9-aa9b-7a401ae6b466">
 <img width="758" alt="3" src="https://github.com/user-attachments/assets/e72a73b2-cdbf-41c0-9c06-4dc5a23244ad">



---

## Example Dataset

Place your dataset images in a folder, e.g., `dataset_folder`, with supported formats (`.jpg`, `.png`, `.jpeg`). Specify the path to this folder in the app.

---

## Contact

For questions, suggestions, or contributions, feel free to get in touch.  

---
 
