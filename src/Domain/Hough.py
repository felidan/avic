import numpy as np
import cv2 as cv
import math

def HoughLinesPadrao(img, cor = (0,255,0), espes_linha = 1, limiares = 50):

    # Detectar as bordas da imagem usando um detector Canny
    dst = cv.Canny(img, 50, 200, None, 3)

    img_saida = img
    lines = cv.HoughLines(dst, 1, np.pi / 180, limiares, None, 0, 0)

    # Transformação de linha Hough padrão
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            
            # Imagem, Ponto1(x,y), Ponto2(x,y), cor, espessura, tipo de linha
            cv.line(img_saida, pt1, pt2, cor, espes_linha, cv.LINE_4)

    return [img_saida, lines]

def HoughLinesProbabilistico(img, cor = (0,255,0), espes_linha = 1, limiares = 50, minLineGap = 50, maxLineGap = 10):

    # Detectar as bordas da imagem usando um detector Canny
    dst = cv.Canny(img, 50, 200, None, 3)

    img_saida = img
    theta = []

    lines = cv.HoughLines(dst, 1, np.pi / 180, limiares, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            theta.append(math.degrees(lines[i][0][1]))
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, limiares, None, minLineGap, maxLineGap)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # Imagem, Ponto1(x,y), Ponto2(x,y), cor, espessura, Número de bits, tipo de linha
            cv.line(img_saida, (l[0], l[1]), (l[2], l[3]), cor, espes_linha, cv.LINE_AA)

    return [img_saida, theta]
