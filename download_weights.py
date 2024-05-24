import os
import requests
import argparse

URLS = {
    "small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "big": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "giant": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
}


def download_weights(weights_dir, model_size):
    url = URLS[model_size]
    filename = os.path.join(weights_dir, os.path.basename(url))

    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping download.")
        return

    print(f"Downloading {url} to {filename}")
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    description = "Download DINOv2 weights"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--weights-dir",
        type=str,
        help="Directory to save the weights",
        default="weights"
    )
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        type=str,
        choices=URLS.keys(),
        help="Model size to download",
        default=["small", "big", "large"],
    )
    args = parser.parse_args()

    assert len([x for x in args.model_sizes if x not in URLS.keys()]) == 0, "Invalid model size"
    
    
    os.makedirs("weights", exist_ok=True)

    for model_size in set(args.model_sizes):
        download_weights(args.weights_dir, model_size)
