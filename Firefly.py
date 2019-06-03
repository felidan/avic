import numpy as np
import cv2 as cv
import lplFirefly as fire

def segFirefly(n, d, gamma, alpha, beta, maxGenerarion, img):

    # escala de cinza
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_cinza = np.copy(img)
    # Equalização histogramica
    img_equalizada = fire.lplHisteq(img_cinza)
    # Histograma
    H = fire.psrGrayHistogram(img)
    
    intensidades = fire.lplFirefly(n, d, gamma, alpha, beta, maxGenerarion, H)
    altura, largura = img_cinza.shape

    img_segmentada = np.copy(img_cinza)

    for y in range(altura):
        for x in range(largura):
            valor = fire.comparaIntensidade(img_equalizada[y, x], intensidades)
            img_segmentada[y, x] = valor
            img_segmentada[y, x] = valor
            img_segmentada[y, x] = valor

    return img_segmentada
