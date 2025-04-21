import torch
from torchvision import transforms
import PIL.Image
import os
import faiss
import json
import numpy as np
from collections import Counter

from utils.utils import build_model, convert_to_rgb, save_predictions_to_csv
from fathomnet_classification import load_data

def get_hierarchy(img_path, index_taxtree_map):
    image_ = img_path.split("/")[-1]
    return index_taxtree_map[image_]

def get_index_taxtree_map():
    index_taxtree_map = {}
    filepath = "../data/taxa_dataset_train.json"
    """Read a JSON file and return its content."""
    with open(filepath, "r") as file:
        data = json.load(file)
    annotations = data.get("annotations", [])
    for annotation in annotations:
        ann_id = f"{annotation.get('image_id')}_{annotation.get('id')}.png"
        index_taxtree_map[ann_id] = annotation.get("taxonomy_tree", {})
    return index_taxtree_map

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

def compute_embeddings(model, data_df, device='cpu', max_rows=None):
    """
    Compute embeddings, labels, and paths for training images.
    """
    embeddings = []
    labels = []
    paths = []
    df = data_df.iloc[:max_rows] if max_rows is not None else data_df
    for _, row in df.iterrows():
        img_path = row["path"]
        label = row["label"]
        print(f"Processing image: {img_path}, label: {label}")
        emb = extract_embedding(model, img_path, device=device)
        embeddings.append(emb)
        labels.append(label)
        paths.append(img_path)
    embeddings = np.vstack(embeddings).astype('float32')
    return embeddings, labels, paths

def build_linear_faiss_index(embeddings):
    """
    Returns:
        index: FAISS linear index.
    """
    d = embeddings.shape[1]  # dimensionality of the embeddings
    print("Building linear FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print("Finished building linear FAISS index.")
    return index

def build_hierarchical_faiss_index(embeddings, nprobe=10):
    """
    
    Returns:
        index: FAISS IVF index.
    """
    d = embeddings.shape[1]
    # nlist = int(np.sqrt(len(embeddings)))  # number of clusters
    nlist = 51 # number of genus
    print(f"Using {nlist} clusters for the IVF index.")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.nprobe = nprobe  # number of clusters to search
    print("Training hierarchical FAISS index...")
    index.train(embeddings)
    index.add(embeddings)
    print(f"Finished building hierarchical FAISS index with {index.ntotal} vectors.")
    return index

def compute_consensus(neighbors, imageid_taxtree_map):
    """
    Compute the consensus dictionary and return it along with the number of neighbor taxonomies.
    """
    neighbor_taxonomies = [get_hierarchy(neighbor, imageid_taxtree_map) for neighbor in neighbors]
    consensus = {}
    if neighbor_taxonomies:
        for level in neighbor_taxonomies[0].keys():
            votes = Counter(tax[level] for tax in neighbor_taxonomies)
            most_common_label, count = votes.most_common(1)[0]
            consensus[level] = {"prediction": most_common_label, "votes": count}
    return consensus, len(neighbor_taxonomies)

def consensus_protocol(neighbors, imageid_taxtree_map):
    """
    Compute consensus by majority vote over all taxonomic levels.
    Returns the consensus dictionary and the first non-"None" prediction from the
    most specific to the most general level.
    """
    consensus, _ = compute_consensus(neighbors, imageid_taxtree_map)
    taxonomic_levels = ["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    final_prediction = None
    # Iterate from the most specific (Species) to the most general (Domain)
    for level in reversed(taxonomic_levels):
        if level in consensus and consensus[level]["prediction"] != "None":
            final_prediction = consensus[level]["prediction"]
            break
    return consensus, final_prediction

def consensus_protocol_majority(neighbors, imageid_taxtree_map):
    """
    Compute consensus with a strict majority requirement.
    Returns the consensus dictionary and a final prediction only if there is a strict majority
    (more than half the neighbors vote for the label) and the label is not "None".
    """
    consensus, total_neighbors = compute_consensus(neighbors, imageid_taxtree_map)
    taxonomic_levels = ["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    final_prediction = None
    # Iterate from the most specific (Species) to the most general (Domain)
    for level in reversed(taxonomic_levels):
        if level in consensus:
            pred = consensus[level]["prediction"]
            vote_count = consensus[level]["votes"]
            if pred != "None" and vote_count > (total_neighbors / 2):
                final_prediction = pred
                break
    return consensus, final_prediction

def consensus_protocol_most_commom(neighbor_labels):
    return Counter(neighbor_labels).most_common(1)[0]
    
def predict_labels(faiss_index, labels, paths, test_embeddings, test_df, k=20):
    """
    Predicts labels for test images by querying the FAISS index with precomputed
    test embeddings and applying three consensus protocols.

    Args:
        faiss_index: The FAISS index (linear or hierarchical).
        labels (List[str]): Training labels corresponding to the FAISS index.
        paths (List[str]): Training image paths corresponding to the FAISS index.
        test_embeddings (np.ndarray): Precomputed test embeddings.
        test_df (pandas.DataFrame): DataFrame for test images (used here to extract image paths for naming).
        k (int): Number of nearest neighbors.

    Returns:
        Tuple: (image_names, predictions_consensus, predictions_majority, predictions_most_common)
    """
    imageid_taxtree_map = get_index_taxtree_map()
    image_names = []
    predictions_consensus = []
    predictions_majority = []
    predictions_most_common = []
    
    # Iterate over test embeddings (assume order matches test_df)
    for idx, row in test_df.iterrows():
        image_path = row["path"]
        emb = np.expand_dims(test_embeddings[idx], axis=0).astype('float32')
        
        distances, indices = faiss_index.search(emb, k)
        
        # Retrieve the labels and paths for the k nearest neighbors
        neighbor_labels = [labels[i] for i in indices[0]]
        neighbor_paths = [paths[i] for i in indices[0]]
        
        # Apply consensus protocols
        _, final_pred1 = consensus_protocol(neighbor_paths, imageid_taxtree_map)
        _, final_pred2 = consensus_protocol_majority(neighbor_paths, imageid_taxtree_map)
        most_common_label, _ = consensus_protocol_most_commom(neighbor_labels)
        
        # Logging (optional)
        print(f"Image: {image_path}")
        print(f"  consensus_protocol prediction: {final_pred1}")
        print(f"  consensus_protocol_majority prediction: {final_pred2}")
        print(f"  consensus_protocol_most_commom prediction: {most_common_label}")
        
        # Extract image name (customize as needed)
        img_name = image_path.split("/")[-1].split("_")[-1].split(".")[0]
        image_names.append(img_name)
        predictions_consensus.append(final_pred1)
        predictions_majority.append(final_pred2)
        predictions_most_common.append(most_common_label)
        
    return image_names, predictions_consensus, predictions_majority, predictions_most_common
        

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/vit_l_16_pre-None_cls-one_hot_seed-42_e-50"
    ckpt_path = os.path.join(model_path, "model.ckpt")
    model = build_model(
        encoder_arch="vit_l_16",
        encoder_path=None,
        classifier_type="one_hot",
        requires_grad=False,
    )
    
    # Load the checkpoint state dict
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    train_df, test_df = load_data()
    embeddings, train_labels, train_paths = compute_embeddings(model, train_df, device=device, max_rows=None)
    # Build the FAISS index from training data
    faiss_linear_index = build_linear_faiss_index(embeddings)
    faiss_hierarchical_index = build_hierarchical_faiss_index(embeddings, nprobe=10)
    
    test_embeddings, _, _ = compute_embeddings(model, test_df, device=device, max_rows=None)
    #Predict - linear index
    num_nearest_neighbors = 20

    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Get predictions from all three protocols
    (image_names,
     preds_consensus,
     preds_majority,
     preds_most_common) = predict_labels(faiss_linear_index, train_labels, train_paths, test_embeddings, test_df, num_nearest_neighbors)
    
    # Save predictions to separate CSV files
    save_predictions_to_csv(image_names, preds_consensus, os.path.join(model_path, "fssai_linear_index_predictions_{model_name}.csv"))
    save_predictions_to_csv(image_names, preds_majority, os.path.join(model_path, "fssai_linear_index_majority_{model_name}.csv"))
    save_predictions_to_csv(image_names, preds_most_common, os.path.join(model_path, "fssai_linear_index_most_common_{model_name}.csv"))

    #Predict - hierarchical index
    # Get predictions from all three protocols
    (image_names,
     preds_consensus,
     preds_majority,
     preds_most_common) = predict_labels(faiss_hierarchical_index, train_labels, train_paths, test_embeddings, test_df, num_nearest_neighbors)
    
    # Save predictions to separate CSV files
    save_predictions_to_csv(image_names, preds_consensus, os.path.join(model_path, "fssai_hierarchical_index_predictions_{model_name}.csv"))
    save_predictions_to_csv(image_names, preds_majority, os.path.join(model_path, "fssai_hierarchical_index_majority_{model_name}.csv"))
    save_predictions_to_csv(image_names, preds_most_common, os.path.join(model_path, "fssai_hierarchical_index_most_common_{model_name}.csv"))
