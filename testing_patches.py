import os
import json
import numpy as np


def extract_and_save_patches(array, patch_size=64, stride=64, save_dir="cnn_patches"):
    os.makedirs(save_dir, exist_ok=True)
    #remove batch dim

    H, W, C = array.shape
    metadata = []

    patch_id = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = array[i:i+patch_size, j:j+patch_size, :]
            filename = f"patch_{patch_id:04d}.npy"
            np.save(os.path.join(save_dir, filename), patch)

            # record metadata,, so matching labels easier later...
            metadata.append({
                "id": patch_id,
                "filename": filename,
                "origin": [i, j],
                "shape": patch.shape
            })
            patch_id += 1

    # save metadata as JSON
    with open(os.path.join(save_dir, "patch_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"save {patch_id} patches to {save_dir}")



def main():

    elev = np.load("elev_resized3.npy")
    lst = np.load("lst_resized2.npy")
    ndvi = np.load("ndvi_resized2.npy")
    landuse = np.load("ura_cnn_ready2.npy")

    cnn_input = np.concatenate([
        lst[..., np.newaxis],
        ndvi[..., np.newaxis],
        elev[..., np.newaxis],
        landuse[..., np.newaxis]
    ], axis=-1)
    
    #print(f"cnn_input shape: {cnn_input.shape}")

    #extract and save patches
    
    extract_and_save_patches(cnn_input, patch_size=64, stride=64)

if __name__ == "__main__":
    main()



