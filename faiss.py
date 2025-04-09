import torch
from torchvision import transforms
import PIL.Image
import os
import faiss
import numpy as np
from collections import Counter

from utils.utils import build_model, convert_to_rgb, save_predictions_to_csv
from fathomnet_classification import load_data

def extract_embedding(model, image_path, device='cpu'):
    """
    Extracts an embedding from a given image using the provided ViT model.
    
    Args:
        model: The trained ViT model.
        image_path (str): Path to the image file.
        device (str): Device to run the model on ('cpu' or 'cuda').
    
    Returns:
        np.ndarray: The extracted embedding as a 1D NumPy array.
    """
    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(convert_to_rgb),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Open and process the image
    img = PIL.Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Set model to evaluation mode and compute the embedding
    model.eval()
    with torch.no_grad():
        embedding = model(img_tensor)
    
    # Convert embedding to a NumPy array and squeeze to remove extra dimensions
    embedding = embedding.squeeze().cpu().numpy()
    return embedding

def build_faiss_index(model, train_df, device='cpu'):
    """
    Extracts embeddings for all images in the training dataset, fetches their labels,
    and builds a FAISS index.
    
    Returns:
    index: FAISS index built from the training embeddings.
    labels: List of labels corresponding to the embeddings (in the same order).
    """
    
    embeddings = []
    labels = []
    for _, row in train_df.iterrows():
        img_path = row["path"]
        label = row["label"]
        print(f"Processing image: {img_path}, label: {label}")
        emb = extract_embedding(model, img_path, device=device)
        embeddings.append(emb)
        labels.append(label)
    
    # Convert embeddings to a float32 NumPy array (FAISS requires float32)
    embeddings = np.vstack(embeddings).astype('float32')
    
    # Build a FAISS index using L2 distance (IndexFlatL2)
    print("Building FAISS index...")
    d = embeddings.shape[1]  # dimensionality of the embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print("Finished building FAISS index.")
    
    return index, labels

def predict_labels(model, faiss_index, labels, test_df, k=10, device='cpu'):
    """
    Predicts labels for a new test image by computing its embedding, querying the FAISS index,
    and returning the labels of the k nearest neighbors.
    
    Returns:
        List[str]: The predicted labels (from the k nearest neighbors).
    """
    images = []
    predictions = []
    for _, row in test_df.iterrows():
        image_path = row["path"]

        # Compute embedding for the test image
        emb = extract_embedding(model, image_path, device=device)
        emb = np.expand_dims(emb, axis=0).astype('float32')
        
        # Search the FAISS index for the k nearest neighbors
        distances, indices = faiss_index.search(emb, k)
        
        # Retrieve the labels for the k nearest neighbors
        neighbor_labels = [labels[i] for i in indices[0]]
        print(f"Distances: {distances}, Neighbor labels: {neighbor_labels}")
        
        # Determine the most common label among the neighbors
        most_common_label, _ = Counter(neighbor_labels).most_common(1)[0]
        images.append(image_path.split("/")[-1].split("_")[-1].split(".")[0])
        predictions.append(most_common_label)

    return images, predictions

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/vit_b_32_pre-None_cls-one_hot_seed-0"
    ckpt_path = os.path.join(model_path, "model.ckpt")
    model = build_model(
        encoder_arch="vit_b_32",
        encoder_path=None,
        classifier_type="one_hot",
        requires_grad=False,
    )
    
    # Load the checkpoint state dict
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    df, test_df = load_data()
    
    # Build the FAISS index from training data
    faiss_index, faiss_train_labels = build_faiss_index(model, df, device=device)
    
    #Predict
    image_names, predictions = predict_labels(model, faiss_index, faiss_train_labels, test_df, device=device)
    print("Predicted labels for the test image:", predictions)
    save_predictions_to_csv(image_names, predictions, os.path.join(model_path, "fssai_predictions.csv"))
