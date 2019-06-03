import numpy as np
import cv2 as cv

def height_building(img, im_orig):
    img2 = np.copy(im_orig)
    center = int(img.shape[1]/2)
    minY = 0
    maxY = 0
    continua = True

    try:    
        for x in range(0, img.shape[0]):
            if(img[x, center][0] > 0 and continua):
                minY = x
                continua = False
            if(img[x, center][0] > 0 and x >= maxY):
                maxY = x
    except:
        for x in range(0, img.shape[0]):
            if(img[x, center] > 0 and continua):
                minY = x
                continua = False
            if(img[x, center] > 0 and x >= maxY):
                maxY = x
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    cv.line(img2, (center, maxY), (center, minY), (0,0,255), 3, cv.LINE_4)
    hgt = abs(minY - maxY)

    return [img2, hgt]

# '_MASC.png'
def get_height(name_base):
    
    ant = 0
    arr_h = []
    arr_url = []
    url_1 = name_base
    
    w = True
    i = 1

    while(w):
        url = ''
        url = '_base\\' + str(i) + url_1
        im1 = cv.imread(url)
        try:
            if(im1.size > 0):
                arr_url.append(url)
                i += 1
            else:
                w = False
        except:
            break
    
    for y in range(0, len(arr_url)):
    
        im1 = cv.imread(arr_url[y])
        im2 = np.copy(im1)

        im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    
        [im, a] = height_building(im1, im2)
        
        cv.imwrite('_result/' + str(y) + url_1, im)
        print('Height in ' + str(arr_url[y]) + ': ' + str(a))
