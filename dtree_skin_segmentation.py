import pickle
import numpy as np
import cv2
import math

model_loc = "./Dataset/dtree_all_images.sav"

def load_rgb_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

def RGB_TO_HSI(img, BGR=False):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        if BGR:
            #Separate color channels
            blue = bgr[:,:,0]
            green = bgr[:,:,1]
            red = bgr[:,:,2]
        else:
            blue = bgr[:,:,2]
            green = bgr[:,:,1]
            red = bgr[:,:,0]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return hsi

def image_masks(rgb_image, aggression=0.9):
    loaded_model = pickle.load(open(model_loc, 'rb'))
    print("Loaded Model is:", model_loc)
    background = np.zeros(rgb_image.shape[:2])
    skin = np.zeros(rgb_image.shape[:2])
    burn = np.zeros(rgb_image.shape[:2])

    test_image_hsi = RGB_TO_HSI(rgb_image)
    for row in range(0, test_image_hsi.shape[1]):
        for col in range(0, test_image_hsi.shape[0]):
            if np.isnan(test_image_hsi[row,col][0]):
                background[row,col] = 1
            else:
                prediction = loaded_model.predict([test_image_hsi[row,col]])
                score = loaded_model.predict_proba([test_image_hsi[row,col]])
                if prediction == 0:
                    background[row,col] = 1
                elif prediction == 1 and score[0][1] > aggression:
                    skin[row,col] = 1
                    #print("Skin Score:", score)
                elif prediction == 2 and score[0][2] > aggression:
                    burn[row,col] = 1
                    #print("Burn Score:", score)
                else:
                    background[row,col] = 1
    return background, skin, burn

def dilate_and_erode(mask, erosion=4, dilation=2):
    img_dilation = cv2.dilate(mask,
                              np.ones((3,3), np.uint8),iterations=dilation)
    # blurred = cv2.GaussianBlur(img_dilation,(7,7),
    #                            cv2.BORDER_DEFAULT).astype('uint8')
    img_erosion = cv2.erode(img_dilation,
                            np.ones((3,3), np.uint8), iterations=erosion)
    return img_erosion

def erode_and_dilation(mask, erosion=2, dilation=2):
    img_erosion = cv2.erode(mask,
                            np.ones((3,3), np.uint8), iterations=erosion)
    img_dilation = cv2.dilate(img_erosion,
                              np.ones((3,3), np.uint8),iterations=dilation)
    # blurred = cv2.GaussianBlur(img_dilation,(7,7),
    #                            cv2.BORDER_DEFAULT).astype('uint8')
    return img_dilation


def skin_overlay(rgb_image, skin_only=False, return_mask=False, aggression=0.5, skin_dilation=2, skin_erosion=4, burn_dilation=2, burn_erosion=4):
    background, skin, burn = image_masks(rgb_image, aggression)
    background_processed = erode_and_dilation(background)
    skin_processed = dilate_and_erode(skin, dilation=skin_dilation, erosion=skin_erosion)
    burn_processed = dilate_and_erode(burn, dilation=burn_dilation, erosion=burn_erosion)
    class_mask = np.zeros(rgb_image.shape[:2])
    img_copy = np.zeros(rgb_image.shape)

    for row in range(0,img_copy.shape[1]):
        for col in range(0,img_copy.shape[0]):
            if burn_processed[row,col] > 0:
                img_copy[row,col] = (255, 0, 0)
                class_mask[row,col] = 2
            elif skin_processed[row,col] > 0:
                img_copy[row,col] = (0, 255, 0)
                class_mask[row,col] = 1
            elif background_processed[row,col] > 0:
                img_copy[row,col] = (0, 0, 0)
                class_mask[row,col] = 0
    if skin_only:
        print("remove background")
    else:
        img_copy_converted = img_copy.astype('uint8')
        fused_img = cv2.addWeighted(rgb_image, 0.8, img_copy_converted, 0.2, 0)

    if return_mask:
        return fused_img, class_mask
    else:
        return fused_img
