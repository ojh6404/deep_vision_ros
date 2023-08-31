#!/usr/bin/env python3

import numpy as np
import torch
from torchvision.ops import box_convert
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict, annotate


def load_image(cv2_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(cv2_image)
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def get_grounded_bbox(model, image, text_prompt, box_threshold=0.5, text_threshold=0.5):
    """
    image: numpy ndarray [H, W, 3]
    text_prompt: string
    output: bboxes [N, 4] numpy array
    """
    image_source, image = load_image(image)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )


    h, w = image_source.shape[:2]

    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    bboxes = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    return bboxes, phrases
