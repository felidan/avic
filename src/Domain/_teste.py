import cv2
import AVIC as av
import numpy as np
import Hough as r
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

url = "frame_3.jpg"

im0 = cv2.imread(url)

# 0 - Porcentagem de quando irá cortar de cada lado da iamgem
porcent_vertical = 0.00 # %
porcent_horizontal = 0.30 # %

# 2 - Corta a imagem para deixar apenas o prédio
vertical = int(im0.shape[0])
horizontal = int(im0.shape[1])
v = int(vertical*porcent_vertical)
h = int(horizontal*porcent_horizontal)
im0 = im0[v:vertical-v, h:horizontal-h]

im1 = np.copy(im0)

[imagem, retas, matriz] = r.HoughLinesProbabilistico2(im1, cor=(0,255,0), espes_linha=1, limiar=30, maxGap=10, cannyMinVal=75, cannyMaxVal=150)

cv2.imshow("", imagem)
cv2.waitKey(0)
