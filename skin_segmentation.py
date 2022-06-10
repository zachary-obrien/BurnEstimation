import cv2
import numpy as np

def get_skin(input_image, min_h=0, max_h=50):
    hsi_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HLS)
    output_shape = list(hsi_image.shape[0:2])
    empty_image = np.zeros(tuple(output_shape), dtype='uint16')
    for row in range(0,hsi_image.shape[0]):
        for col in range(0,hsi_image.shape[1]):
            if hsi_image[row,col,0] > min_h and hsi_image[row,col,0] < max_h:
                empty_image[row,col] = 1

    #restored_img = cv2.cvtColor(empty_image, cv2.COLOR_HLS2RGB)
    return empty_image


def erode_and_dilate(img):
    img_erosion = cv2.erode(img,
                            np.ones((5,5), np.uint8), iterations=4)
    blurred = cv2.GaussianBlur(img_erosion,(7,7),cv2.BORDER_DEFAULT).astype('uint8')

    img_dilation = cv2.dilate(blurred, np.ones((5,5), np.uint8), iterations=4)
    (thresh, im_bw) = cv2.threshold(img_dilation, 125, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #(cnts, _) = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cnts, _) = cv2.findContours(im_bw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape)
    if len(cnts) > 0:
        c = max(cnts, key = cv2.contourArea)
        #cv2.drawContours(mask, cnts, -1, (255,255,255), 3)
        cv2.fillPoly(mask, pts =[c], color=(1))
    return mask

def apply_mask(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked