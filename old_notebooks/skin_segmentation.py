import cv2
import numpy as np
import glob
from converter import RGB_TO_HSI

def load_rgb_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

def import_image_folder(img_folder):
    images = glob.glob(img_folder + "*.jpg")
    return images

def get_skin(input_image, min_h=5, max_h=25):
    #hsi_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HLS)
    hsi_image = RGB_TO_HSI(input_image, BGR=False)
    output_shape = list(hsi_image.shape[0:2])
    empty_image = np.zeros(tuple(output_shape), dtype='uint16')
    for row in range(0,hsi_image.shape[0]):
        for col in range(0,hsi_image.shape[1]):
            if hsi_image[row,col,0] > min_h and hsi_image[row,col,0] < max_h:
                empty_image[row,col] = 1
    return empty_image

def erode_and_dilate(img, polyFill=True):
    img_erosion = cv2.erode(img,
                            np.ones((5,5), np.uint8), iterations=4)
    blurred = cv2.GaussianBlur(img_erosion,(7,7),cv2.BORDER_DEFAULT).astype('uint8')

    img_dilation = cv2.dilate(blurred, np.ones((5,5), np.uint8), iterations=4)
    if polyFill:
        (thresh, im_bw) = cv2.threshold(img_dilation, 125, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        (cnts, _) = cv2.findContours(im_bw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(img.shape)
        if len(cnts) > 0:
            c = max(cnts, key = cv2.contourArea)
            cv2.fillPoly(mask, pts =[c], color=(1))
        return mask
    else:
        return img_dilation

def apply_mask(image, mask):
    empty = np.zeros(image.shape)
    for row in range(0,mask.shape[0]):
        for col in range(0,mask.shape[1]):
            if mask[row,col] > 0:
                empty[row,col] = image[row,col]
    #masked = cv2.bitwise_and(image, image, mask=mask)
    return empty