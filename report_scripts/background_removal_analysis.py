import cv2
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import sys
import torch

def make_deeplab(device):
    deeplab = deeplabv3_resnet101(pretrained=True).to(device)
    deeplab.eval()
    return deeplab


def apply_deeplab(deeplab, img, device):
    deeplab_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return (output_predictions == 15)



def remove_bg_transformer(image):
    device = torch.device("cpu")
    deeplab = make_deeplab(device)
    img_orig = image.copy()

    mask = apply_deeplab(deeplab, img_orig, device)

    masked = cv2.bitwise_and(img_orig, img_orig, mask=mask.astype(np.uint8))

    return masked


def remove_bg_opencv(img):
    original = img.copy()

    edges = cv2.GaussianBlur(img, (21, 51), 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(edges, 6, 6)

    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    data = mask.tolist()
    sys.setrecursionlimit(10 ** 8)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] != 255:
                data[i][j] = -1
            else:
                break
        for j in range(len(data[i]) - 1, -1, -1):
            if data[i][j] != 255:
                data[i][j] = -1
            else:
                break
    image = np.array(data)
    image[image != -1] = 255
    image[image == -1] = 0

    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask == 0] = 255
    return result


def remove_bg_opencv_2(image):
    blur = 21
    canny_low = 6
    canny_high = 6
    min_area = 0.0005
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    image_area = image.shape[0] * image.shape[1]

    max_area = max_area * image_area
    min_area = min_area * image_area
    mask = np.zeros(edges.shape, dtype=np.uint8)
    for contour in contour_info:
        if contour[1] > min_area and contour[1] < max_area:
            mask = cv2.fillConvexPoly(mask, contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask == 0] = 255
    return result

def remove_bg_custom(image):
    indices = np.argwhere(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 140)
    indices = [(int(i), int(j)) for i, j in zip(indices[:, 0], indices[:, 1])]
    for i in indices: image[i] = 0
    return image

image = cv2.imread(r"C:\Users\Filippo\OneDrive - Universiteit Utrecht\Documents\Programming\Python\thesisProject\sample_cropped_face.jpg")
copy_img = image.copy()
pass