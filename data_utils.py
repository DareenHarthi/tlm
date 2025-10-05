import numpy as np
import pickle
from pathlib import Path

def shuffle(X, y):
    
    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
        
    return X, y 

def load_metadata(metadata_file):
    """Load metadata from file"""
    with open(metadata_file, 'r') as f:
        lines = f.read().split('\n')
    
    metadata = []
    for line in lines:
        if line.strip():
            parts = line.split('|')
            if len(parts) >= 3:
                file_path, speaker_id, age = parts[:3]
                if int(age) > 0:
                    metadata.append((file_path, speaker_id, int(age)))
    
    return metadata


def load_data(metadata_train, metadata_eval, data_folder, embedding_dim=192):
    """Load embeddings and labels from metadata"""
    X_train = np.zeros((len(metadata_train), embedding_dim))
    y_train = np.zeros(len(metadata_train))
    
    X_test = np.zeros((len(metadata_eval), embedding_dim))
    y_test = np.zeros(len(metadata_eval))

    for i, (file_path, speaker_id, age) in enumerate(metadata_train):
        # Convert file path to embedding path
        embedding_path = f"{data_folder}/{file_path}"
        embedding_path = embedding_path.replace(".WAV", ".npy")
        X_train[i] = np.load(embedding_path)
        y_train[i] = age     

    for i, (file_path, speaker_id, age) in enumerate(metadata_eval):
        embedding_path = f"{data_folder}/{file_path}"
        embedding_path = embedding_path.replace(".WAV", ".npy")
        X_test[i] = np.load(embedding_path)
        y_test[i] = age     
    
    return X_train, X_test, y_train, y_test


def age_based_mixup(X, y, alpha=0.5, num_augmentations=3):
    """
    Age-based mixup augmentation
    
    Args:
        X: Input features
        y: Age labels  
        alpha: Mixup coefficient
        num_augmentations: Number of augmentation rounds
    
    Returns:
        Augmented X and y
    """
    X_original = X.copy()
    y_original = y.copy()
    
    X_augmented = []
    y_augmented = []
    
    for seed in range(1, num_augmentations + 1):
        # Sort by age
        sorted_indices = np.argsort(y_original)
        X_sorted = X_original[sorted_indices]
        y_sorted = y_original[sorted_indices]

        # Shuffle within age windows
        window_size = 4
        
        shuffle_indices = np.arange(len(X_sorted))
        for i in range(0, len(shuffle_indices), window_size):
            
            window = shuffle_indices[i:i+window_size]
            np.random.shuffle(window)
            shuffle_indices[i:i+window_size] = window
        
        X_shuffled = X_sorted[shuffle_indices]
        y_shuffled = y_sorted[shuffle_indices]
            
        # Mixup
        X_mixed = alpha * X_sorted + (1 - alpha) *  X_shuffled
        y_mixed = (alpha * y_sorted + (1 - alpha) * y_shuffled).astype(int)

        X_augmented.append(X_mixed)
        y_augmented.append(y_mixed)
    
    # Combine all augmented data
    X_all = [X_original] + X_augmented
    y_all = [y_original] + y_augmented
    
    X_final = np.concatenate(X_all, axis=0)
    y_final = np.concatenate(y_all, axis=0)
    
    # Shuffle the final dataset
    X_final, y_final = shuffle(X_final, y_final)
    
    return X_final, y_final

def uniform_mixup(X, y, alpha=0.5):
       
    for _ in range(3):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        try:
            X_mix = np.append(X_mix, alpha * X + (1 - alpha) * X_shuffled,  axis=0) 

            y_mix = np.append(y_mix, (alpha * y + (1 - alpha) * y_shuffled).astype(int),  axis=0) 
            
        except:
            X_mix = alpha * X + (1 - alpha) * X_shuffled

            y_mix = (alpha * y + (1 - alpha) * y_shuffled).astype(int)



    # Append to the orginal data
    
    X = np.append(X, X_mix, axis=0)
    y = np.append(y, y_mix, axis=0)
    
    X, y   = shuffle(X, y)
    return X, y

def load_speaker_info(speaker_info_file):
    """Load speaker gender information"""
    with open(speaker_info_file, "r") as f:
        info = f.read().split("\n")

    # Parse speaker info (skip header lines)
    info = [i.split("  ")[:2] for i in info[39:-1] if len(i.split("  ")) >= 2]
    return {k: v for k, v in info}


def filter_by_gender(metadata, speaker_info, gender):
    """Filter metadata by gender (M/F)"""
    return [(file_path, sid, age) for file_path, sid, age in metadata 
            if speaker_info.get(file_path.split("/")[-2][1:]) == gender]


def get_age_thresholds(y_train):
    """Generate age thresholds for tree splitting"""
    unique_ages = sorted(set(y_train))

    return unique_ages[2:-2]


def save_tree(tree, file):
    """Save tree model to file using pickle"""
    with open(file, 'wb') as f:
        pickle.dump(tree, f)
        

def load_tree(file):
    """Load tree model from file using pickle"""
    with open(file, 'rb') as f:
        tree = pickle.load(f)
    return tree