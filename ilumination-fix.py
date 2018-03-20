import cv2

#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os

def useCLAHE(img):
    #cv2.imshow("img",img)

    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #cv2.imshow("lab",lab)

    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #cv2.imshow('final', final)
    return cl


# Path of the files that you want to resize
path = "ilumination/s3/"
path_content = os.listdir(path)

for image_name in path_content:
    print (image_name)
    image_path  = path + '/' + image_name
    image = cv2.imread(image_path)
    final = useCLAHE(image)
    # resized_image = cv2.resize(image, (800, 600))
    # cv2.waitKey(0)

    cv2.imwrite(image_path, final)
