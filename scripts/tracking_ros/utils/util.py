#!/usr/bin/env python3
import os
import requests
import gdown
from tqdm import tqdm

SAM_CHECKPOINT_DICT = {
    "sam_vit_h": "sam_vit_h_4b8939.pth",
    "sam_vit_l": "sam_vit_l_0b3195.pth",
    "sam_vit_b": "sam_vit_b_01ec64.pth",
    "sam_hq_vit_b": "sam_hq_vit_b.pth",
    "sam_hq_vit_l": "sam_hq_vit_l.pth",
    "sam_hq_vit_h": "sam_hq_vit_h.pth",
}
SAM_CHECKPOINT_URL_DICT = {
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_hq_vit_b": "https://drive.google.com/uc?id=11yExZLOve38kRZPfRx_MRxfIAKmfMY47",
    "sam_hq_vit_l": "https://drive.google.com/uc?id=1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G",
    "sam_hq_vit_h": "https://drive.google.com/uc?id=1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8",
}
XMEM_CHECKPOINT = "XMem-s012.pth"
XMEM_CHECKPOINT_URL = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)
DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
DINO_CHECKPOINT_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

def download_checkpoint(model, folder):
    os.makedirs(folder, exist_ok=True)
    if model in SAM_CHECKPOINT_DICT.keys():
        if "hq" in model:
            filepath = download_checkpoint_from_google_drive(
                SAM_CHECKPOINT_URL_DICT[model], folder, SAM_CHECKPOINT_DICT[model]
            )
            return filepath
    elif model == "xmem":
        filepath = os.path.join(folder, XMEM_CHECKPOINT)
        url = XMEM_CHECKPOINT_URL
    elif model == "dino":
        filepath = os.path.join(folder, DINO_CHECKPOINT)
        url = DINO_CHECKPOINT_URL
    else:
        raise ValueError(f"Unknown model {model}")

    if not os.path.exists(filepath):
        print("Download checkpoints of {} from {} ...".format(model, url))
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filepath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        print("Downloaded successfully!")
    return filepath


def download_checkpoint_from_google_drive(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(
            "Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)"
        )
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")
    return filepath
