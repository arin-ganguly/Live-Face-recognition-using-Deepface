import os
import pickle
from deepface import DeepFace
import numpy as np

embedding_path = "E:\Face Recognition\Embedded files\embeddings24.pkl"

def get_image_paths_flat(dataset_path):
    """Get all image paths and use filename (without extension) as label."""
    image_paths = []
    labels = []

    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dataset_path, img_name)
            label = os.path.splitext(img_name)[0]  # use filename as label
            image_paths.append(img_path)
            labels.append(label)

    return image_paths, labels

def compute_embedding(img_path, model_name="ArcFace", detector_backend="retinaface"):
    """Extract a face embedding using DeepFace."""
    try:
        representation = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        return representation[0]["embedding"]
    except Exception as e:
        print(f"Failed on {img_path} - {str(e)}")
        return None

def generate_embeddings(image_paths, labels):
    embeddings = []
    valid_labels = []

    for img_path, label in zip(image_paths, labels):
        print(f"Processing: {img_path}")
        embedding = compute_embedding(img_path)
        if embedding is not None:
            embeddings.append(embedding)
            valid_labels.append(label)
            print(f"Embedded: {img_path}")
        else:
            print(f"No embedding for: {img_path}")

    return embeddings, valid_labels

def save_embeddings(embeddings, labels, output_path=embedding_path):
    with open(output_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"\nSaved {len(embeddings)} embeddings to {output_path}")

def main():
    dataset_path = r"E:\Fortuna_Video_Analytics_OnFace_&_FAISS\dataset\Data"
    
    print("Loading flat image dataset...")
    image_paths, labels = get_image_paths_flat(dataset_path)

    print("Generating embeddings...")
    embeddings, valid_labels = generate_embeddings(image_paths, labels)

    print("Saving embeddings...")
    save_embeddings(embeddings, valid_labels)

if __name__ == "__main__":
    main()

