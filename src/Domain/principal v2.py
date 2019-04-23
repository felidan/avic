import cv2 as cv
import numpy as np
import Firefly as f
import Hough as ho
import AVIC as av
import matplotlib.pyplot as plt

###################
#   AVIC - TCC2   #
###################

url_imagem = "frame_0.jpg"
url_video = "_base/video4.mp4"
url_destino = "fotos_salvas/frames/"

# av.salva_frames(url_video, url_destino, 0.2, 78, 90)

######
result = av.media_frames(url_destino, 'frame_', '.jpg', 0, 20)
######

# 8 - Hough
cor = (0, 0, 255)
# img, cor = (0,255,0), espes_linha = 1, limiares = 50, minLineGap = 50, maxLineGap = 10
# implementar validação para ver se a imagem está em escala de cinza
[imagem_linhas, theta] = ho.HoughLinesProbabilistico(result, cor, 1, 50, 100, 10)

cv.imshow("Transformada de Hough", imagem_linhas)

plt.hist(theta, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

cv.waitKey(0)
