# paint masks, contours, or points on images, with specified colors
import cv2
import numpy as np


def colormap(rgb=True):
	color_list = np.array(
		[
			1.000, 0.498, 0.313,
			0.392, 0.581, 0.929,
			0.000, 0.447, 0.741,
			0.850, 0.325, 0.098,
			0.929, 0.694, 0.125,
			0.494, 0.184, 0.556,
			0.466, 0.674, 0.188,
			0.301, 0.745, 0.933,
			0.635, 0.078, 0.184,
			0.300, 0.300, 0.300,
			0.600, 0.600, 0.600,
			1.000, 0.000, 0.000,
			1.000, 0.500, 0.000,
			0.749, 0.749, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 1.000,
			0.667, 0.000, 1.000,
			0.333, 0.333, 0.000,
			0.333, 0.667, 0.000,
			0.333, 1.000, 0.000,
			0.667, 0.333, 0.000,
			0.667, 0.667, 0.000,
			0.667, 1.000, 0.000,
			1.000, 0.333, 0.000,
			1.000, 0.667, 0.000,
			1.000, 1.000, 0.000,
			0.000, 0.333, 0.500,
			0.000, 0.667, 0.500,
			0.000, 1.000, 0.500,
			0.333, 0.000, 0.500,
			0.333, 0.333, 0.500,
			0.333, 0.667, 0.500,
			0.333, 1.000, 0.500,
			0.667, 0.000, 0.500,
			0.667, 0.333, 0.500,
			0.667, 0.667, 0.500,
			0.667, 1.000, 0.500,
			1.000, 0.000, 0.500,
			1.000, 0.333, 0.500,
			1.000, 0.667, 0.500,
			1.000, 1.000, 0.500,
			0.000, 0.333, 1.000,
			0.000, 0.667, 1.000,
			0.000, 1.000, 1.000,
			0.333, 0.000, 1.000,
			0.333, 0.333, 1.000,
			0.333, 0.667, 1.000,
			0.333, 1.000, 1.000,
			0.667, 0.000, 1.000,
			0.667, 0.333, 1.000,
			0.667, 0.667, 1.000,
			0.667, 1.000, 1.000,
			1.000, 0.000, 1.000,
			1.000, 0.333, 1.000,
			1.000, 0.667, 1.000,
			0.167, 0.000, 0.000,
			0.333, 0.000, 0.000,
			0.500, 0.000, 0.000,
			0.667, 0.000, 0.000,
			0.833, 0.000, 0.000,
			1.000, 0.000, 0.000,
			0.000, 0.167, 0.000,
			0.000, 0.333, 0.000,
			0.000, 0.500, 0.000,
			0.000, 0.667, 0.000,
			0.000, 0.833, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 0.167,
			0.000, 0.000, 0.333,
			0.000, 0.000, 0.500,
			0.000, 0.000, 0.667,
			0.000, 0.000, 0.833,
			0.000, 0.000, 1.000,
			0.143, 0.143, 0.143,
			0.286, 0.286, 0.286,
			0.429, 0.429, 0.429,
			0.571, 0.571, 0.571,
			0.714, 0.714, 0.714,
			0.857, 0.857, 0.857
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3)) * 255
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list


color_list = colormap()

def mask_painter(image, mask, color_index, alpha=0.5):
    # image : [H, W, C] numpy array
    # masks : mask [H, W] True/False numpy array
    # colors : color [C] numpy array
    # alpha : float
    # return : [H, W, C] numpy array of image
    if mask is None:
        return image
    color = color_list[color_index+1]
    image[mask] = image[mask] * (1 - alpha) + color * alpha
    return image.astype(np.uint8)

def point_drawer(image, points, labels, radius=5):
    # image : [H, W, C] numpy array
    # points : list of point [2] numpy array
    # labels : list of label like [1, 0]
    # colors : list of color [C] numpy array
    # radius : int
    # alpha : float
    # return : [H, W, C] numpy array of image
    if points == []:
        return image
    for i, point in enumerate(points):
        if labels[i] == 1:
            color = [0, 0, 255]
        else:
            color = [255, 0, 0]
        image = cv2.circle(image, tuple(point), radius, color, -1)
    return image.astype(np.uint8)

def bbox_drawer(image, bbox, color=None):
    # image : [H, W, C] numpy array
    # bbox : bbox [4] numpy array
    # colors : color [C] numpy array
    # return : [H, W, C] numpy array of image
    if bbox is None:
        return image
    if color is None:
        color = [0, 255, 0]
    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]) ), color, 2)
    return image.astype(np.uint8)

def bbox_drawer_with_text(image, bbox, text, color_index):
    # image: [H, W, C]
    # bbox: [x1, y1, x2, y2]
    # text: string
    # output: [H, W, C]
    color = color_list[color_index+1].astype(np.int32).tolist()
    image = bbox_drawer(image, bbox, color=color)
    image = cv2.putText(image, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # image = cv2.putText(image, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return image
