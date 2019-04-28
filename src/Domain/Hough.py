import numpy as np
import cv2 as cv
import math

def HoughLinesPadrao(img, cor = (0,255,0), espes_linha = 1, limiares = 50):

    # Detectar as bordas da imagem usando um detector Canny
    dst = cv.Canny(img, 50, 200, None, 3)

    img_saida = img
    #lines = cv.HoughLines(dst, 1, np.pi / 180, limiares, None, 0, 0)
    lines = cv.HoughLines(dst, 1, np.pi/180, 30, 250)

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
            theta.append(math.degrees(lines[i][0][1]) + 90)
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, limiares, None, minLineGap, maxLineGap)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # Imagem, Ponto1(x,y), Ponto2(x,y), cor, espessura, Número de bits, tipo de linha
            cv.line(img_saida, (l[0], l[1]), (l[2], l[3]), cor, espes_linha, cv.LINE_AA)

    return [img_saida, theta]

def HoughLinesProbabilistico2(img, cor = (0,255,0), espes_linha = 1, limiar=30, maxGap = 250, cannyMinVal=75, cannyMaxVal=150):
    count = 0
    # Tenta converter para escala de cinza
    try:
        img_saida = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except:
        img_saida = img

    # Detecta as bordas da imagem
    bordas = cv.Canny(img_saida, cannyMinVal, cannyMaxVal)
    
    lines = cv.HoughLinesP(bordas, 1, np.pi/180, limiar, maxLineGap=maxGap)

    # Aplica pitagoras para descobrir o tamanho da diagonal da imagem
    # [DIMEN]x[ANGULO]
    t = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
    matrizBinaria = np.zeros((t,91), dtype=int)
    matrizLinhas = np.zeros((len(lines), 2))

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Identifica o centro da linha para encontrar o rho e theta
        modX = np.abs(x1 - x2)/2
        modY = np.abs(y1 - y2)/2
        centroX = modX + min(x1, x2)
        centroY = modY + min(y1, y2)
        # Calcula Rho e Theta do centro da reta
        rho = np.sqrt(centroX**2 + centroY**2)
        theta = math.degrees(np.arctan2(centroY, centroX))
        # Popula matriz binária e matriz de retas
        matrizBinaria[int(round(rho))][int(round(theta))] += 1
        matrizLinhas[count] = (rho, theta)
        cv.line(img, (x1, y1), (x2, y2), cor, espes_linha)
        count += 1

    return [img, matrizLinhas, matrizBinaria]
