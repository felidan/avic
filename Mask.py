import cv2 as CV
import numpy as NP
import Firefly as F
import AVIC as AV
import math
import os

def get_mask(url, img, save, y, x, h, w, numero, morf_open, morf_kernel_1, morf_kernel_2):

    print('processing ' + url)
    # Init
    if(url == ''):
        image = NP.copy(img)
    else:
        image = CV.imread(url)

    if(y == 0 and x == 0 and h == 0 and w == 0):
        pass
    else:
        copy = image.copy()
        image = copy[y:y+h, x:x+w]
    _base = '_result/mask/'

    ### 
    print('Applying CLAHE..')
    cinza = CV.cvtColor(image, CV.COLOR_BGR2GRAY)
    cinza = CV.equalizeHist(cinza)
    clahe = CV.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(cinza)

    if(save == "S"):
        CV.imwrite(_base + '1_{!s}_CLAHE.png'.format(numero), cl)

    ###
    print('Applying BLUR..')
    blur = CV.GaussianBlur(cl,(5,5),0)
    if(save == "S"):
        CV.imwrite(_base + '2_{!s}_BLUR.png'.format(numero), blur)

    ###
    print("Applying Firefly..")
    limiares = 1
    vagalumes = 80
    geracoes = 100
    firefly = F.segFirefly(vagalumes, limiares, 1, 0.50, 1, geracoes, blur)
    if(save == "S"):
        CV.imwrite(_base + '3_{!s}_FIREFLY.png'.format(numero), firefly)

    ###
    print("Applying THRESH..")
    (thresh, im_bw) = CV.threshold(firefly, 0, 255, CV.THRESH_BINARY_INV | CV.THRESH_OTSU)
    if(save == "S"):
        CV.imwrite(_base + '4_{!s}_THRESH.png'.format(numero), im_bw)

    ###
    print("Applying Connected Components..")
    nb_components, output, stats, centroids = CV.connectedComponentsWithStats(im_bw, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 50
    connected = NP.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            connected[output == i + 1] = 255
    
    if(save == "S"):
        CV.imwrite(_base + '5_{!s}_CONNECTED.png'.format(numero), connected)

    ###
    print("Applying Morphology..")

    if(morf_open == "S"):
        morfBefore=CV.morphologyEx(connected,
                                   CV.MORPH_OPEN,
                                   CV.getStructuringElement(CV.MORPH_RECT,
                                                            (18, 10))
                                   )
    
    kernel = NP.ones((morf_kernel_1,morf_kernel_2),NP.uint8)
    morfB = CV.dilate(connected, kernel)
    if(save == "S"):
        CV.imwrite(_base + '6_{!s}_MORF.png'.format(numero), morfB)




    ###
    print("Applying Contour..")
    morfB = NP.uint8(morfB)
    im2, contours, hierarchy = CV.findContours(morfB,CV.RETR_TREE,CV.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=CV.contourArea)
    for cnt in contours:
        CV.drawContours(morfB,[cnt],-1,255,-1)
    '''
    if(existe == 1):
        morfB = CV.add(thresh_mask, morfB)
    else:
        pass
    '''
    if(save == "S"):
        CV.imwrite(_base + '7_{!s}_CONTOUR.png'.format(numero), morfB)  

    ###
    print('Generating Mask..')

    gray = morfB
    ret, gray = CV.threshold(gray, 10,255,0)
    image_src = image

    image, contours, hierarchy = CV.findContours(gray, CV.RETR_LIST, CV.CHAIN_APPROX_SIMPLE)
    largest_area = sorted(contours, key=CV.contourArea)[-1]
    mask = NP.zeros(image_src.shape, NP.uint8)
    CV.drawContours(mask, [largest_area], 0, (255,255,255,255), -1)
    CV.imwrite(_base + '8_{!s}_CONTOUR_2.png'.format(numero), mask)
    dst = CV.bitwise_and(image_src, mask)

    if(save == "S"):
        CV.imwrite(_base + '9_{!s}_MASK.png'.format(numero), dst)

    print('')

    return dst

y, x, h, w = 150, 140, 470, 300
numero_1 = 10
numero_2 = 1
'''
im = get_mask('_base/crop_video21_media_10.png',
              "S",
              y,
              x,
              h,
              w,
              numero_1,
              morf_open="N",
              morf_kernel_1=15,
              morf_kernel_2=40)
'''
'''
im2 = get_mask('_base/crop_video12_media_2.png',
               '',
              "S",
              0,
              0,
              0,
              0,
              numero_2,
              morf_open="N",
              morf_kernel_1=2,
              morf_kernel_2=5)

im3 = get_mask('_base/crop_video12_media_4.png',
               '',
              "S",
              0,
              0,
              0,
              0,
              numero_2,
              morf_open="N",
              morf_kernel_1=2,
              morf_kernel_2=5)
#CV.imshow('1', im)
CV.imshow('2', im2)
CV.imshow('3', im3)
'''
