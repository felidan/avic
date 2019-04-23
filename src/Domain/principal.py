import cv2
import numpy as np
import Firefly as f
import Hough as ho

###################
#   AVIC - TCC2   #
###################

url_imagem = "_base/predio.jpg"
url_video = "_base/video.mp4"
largura, altura = 500, 500

# 0 - Porcentagem de quando irá cortar de cada lado da iamgem
porcent_vertical = 0.15 # %
porcent_horizontal = 0.2 # %

# 1 - Ler imagem
imagem_original = cv2.imread(url_imagem)
cv2.imshow("Predio", imagem_original)

# 2 - Corta a imagem para deixar apenas o prédio
vertical = int(imagem_original.shape[0])
horizontal = int(imagem_original.shape[1])
v = int(vertical*porcent_vertical)
h = int(horizontal*porcent_horizontal)
imagem_cortada = imagem_original[v:vertical-v, h:horizontal-h]
cv2.imshow("Fatia da imagem", imagem_original)

# 3 - Redimenciona a imagem
imagem_original_redimen = cv2.resize(imagem_cortada, (largura, altura))
cv2.imshow("Predio 500x500", imagem_original_redimen)

# 4 - Transforma em escala de cinza
imagem_cinza = cv2.cvtColor(imagem_original_redimen, cv2.COLOR_BGR2GRAY)
cv2.imshow("Predio Cinza", imagem_cinza)

# 5 - Equalização histogramica Global
imagem_equalizada = cv2.equalizeHist(imagem_cinza) 
cv2.imshow("Equalização", imagem_equalizada)

# 6 - Firefly
limiares = 1
vagalumes = 80
geracoes = 100
# img = cv2.cvtColor(imagem_original_redimen, cv2.COLOR_BGR2GRAY)
imagem_segmentada = f.segFirefly(vagalumes, limiares, 1, 0.97, 1, geracoes, imagem_original_redimen)
cv2.imshow("Segmentação", imagem_segmentada)

# 7 - Morfologia Matemática
kernel = np.ones((5,5),np.uint8)
imagem_morf = cv2.erode(imagem_segmentada,kernel,iterations = 1)  # Erosão binária
imagem_morf = cv2.dilate(imagem_segmentada,kernel,iterations = 1) # Dilatação binária
cv2.imshow("Morfologia matemática", imagem_morf)

# 8 - Hough
cor = (0, 255, 0)
# img, cor = (0,255,0), espes_linha = 1, limiares = 50, minLineGap = 50, maxLineGap = 10
# implementar validação para ver se a imagem está em escala de cinza
imagem_linhas = ho.HoughLinesProbabilistico(imagem_cinza, cor, 1, 50, 50, 10)
cv2.imshow("Transformada de Hough", imagem_linhas)

# 9 - TODO - Pegar a maior região de interesse

# 10 - TODO - Equalização somente na região de interesse

'''
# [OPCIONAL] TODO - level-set somente na região de interesse
[imagem_level_set,tcost,imgrec]=levelset_ivc2013(imagem_equalizada);
cv2.imshow(imagem_level_set);
'''

cv2.waitKey(0)
