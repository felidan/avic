import cv2 as cv
import numpy as np
import Firefly as f
import Hough as ho
import AVIC as av
import matplotlib.pyplot as plt

###################
#   AVIC - TCC2   #
###################

print("Processando..")

url_video = "_base/video4.mp4"
url_frames = "pipeline_2/frames/"
url_crops = "pipeline_2/crops/"
nome_frame = "frame_video_4_"
grus_inclinacao = 2

####################
#   Salva frames   #
####################
# av.salva_frames(url_video, url_frames, nome_frame, 0.1, 77, 89)

################################
#  Media dos frames da imagem  #
################################
img = av.media_frames(url_frames, nome_frame, '.jpg', 3, 21)
cv.imwrite("pipeline_2/1_media_frames_3_21.jpg", img)

###############
#   Rotação   #
###############
img_teste_1 = np.copy(img)
img_teste_2 = np.copy(img)
img_teste_2 = av.rotacionaImagem(img_teste_2, grus_inclinacao)

#############################
#   Limita área do predio   #
#############################
img_cortada_1 = av.cortaImagem(imagem=img_teste_1, pc_vertical=0.02, pc_horizontal=0.23)
cv.imwrite("pipeline_2/2_img_cortada_teste_1.jpg", img_cortada_1)

img_cortada_2 = av.cortaImagem(imagem=img_teste_2, pc_vertical=0.02, pc_horizontal=0.23)
cv.imwrite("pipeline_2/2_img_cortada_teste_2.jpg", img_cortada_2)

############
#   Crop   #
############
arr_crops_teste_1 = av.crop_imagem(img_cortada_1, url_crops, 'crop_teste_1_', '.jpg', 4, 2, salvar="S")
arr_crops_teste_2 = av.crop_imagem(img_cortada_2, url_crops, 'crop_teste_2_', '.jpg', 4, 2, salvar="S")

############
#   Hough  #
############

espes_linha=1
limiar=30
maxGap=10
cannyMinVal=75
cannyMaxVal=150
cor_teste_1 = (255,0,0)
cor_teste_2 = (0,255,0)

img_cortada_1 = cv.cvtColor(img_cortada_1, cv.COLOR_GRAY2BGR)
img_cortada_2 = cv.cvtColor(img_cortada_2, cv.COLOR_GRAY2BGR)
[im_1, r_1, m_1] = ho.HoughLinesProbabilistico2(img_cortada_1, cor=cor_teste_1, espes_linha=espes_linha, limiar=limiar, maxGap=maxGap, cannyMinVal=cannyMinVal, cannyMaxVal=cannyMaxVal)
cv.imwrite("pipeline_2/3.1_img_hough_teste_1.jpg", im_1)
[im_2, r_2, m_2] = ho.HoughLinesProbabilistico2(img_cortada_2, cor=cor_teste_2, espes_linha=espes_linha, limiar=limiar, maxGap=maxGap, cannyMinVal=cannyMinVal, cannyMaxVal=cannyMaxVal)
cv.imwrite("pipeline_2/3.2_img_hough_teste_2.jpg", im_2)

arr_crop_img_hough_1 = []
arr_crop_img_hough_2 = []
arr_crop_retas_hough_1 = []
arr_crop_retas_hough_2 = []

for i in range(0, len(arr_crops_teste_1)):
    [imagem_1, retas_1, matriz_1] = ho.HoughLinesProbabilistico2(cv.cvtColor(arr_crops_teste_1[i], cv.COLOR_GRAY2BGR), cor=cor_teste_1, espes_linha=espes_linha, limiar=limiar, maxGap=maxGap, cannyMinVal=cannyMinVal, cannyMaxVal=cannyMaxVal)
    arr_crop_img_hough_1.append(imagem_1)
    arr_crop_retas_hough_1.append(retas_1)
    cv.imwrite(url_crops + "hough/img_crop_teste_1_" + str(i) + ".jpg", imagem_1)

    [imagem_2, retas_2, matriz_2] = ho.HoughLinesProbabilistico2(cv.cvtColor(arr_crops_teste_2[i], cv.COLOR_GRAY2BGR), cor=cor_teste_2, espes_linha=espes_linha, limiar=limiar, maxGap=maxGap, cannyMinVal=cannyMinVal, cannyMaxVal=cannyMaxVal)
    arr_crop_img_hough_2.append(imagem_2)
    arr_crop_retas_hough_2.append(retas_2)
    cv.imwrite(url_crops + "hough/img_crop_teste_2_" + str(i) + ".jpg", imagem_2)

######################################
#   Separa o Theta de de cada crop   #
######################################
arr_theta_1 = []
arr_theta_2 = []
temp = []

for crop in arr_crop_retas_hough_1:
    for reta in crop:
        temp.append(reta[1])
    arr_theta_1.append(temp)
    temp = []

for crop in arr_crop_retas_hough_2:
    for reta in crop:
        temp.append(reta[1])
    arr_theta_2.append(temp)
    temp = []

###################################
#   Compara histograma por crop   #
###################################

bins=180

plt.xlabel("Ângulo")
plt.ylabel("Valor")
plt.hist(arr_theta_1[i], bins, label='Teste 1 - reto')
plt.hist(arr_theta_2[i], bins, label='Teste 2 - ' + str(grus_inclinacao) + 'º de inclinação')
plt.legend(loc='upper right')
plt.savefig("pipeline_2/4_histograma_total.png")
plt.clf()
    
for i in range(0, len(arr_theta_1)):
    plt.xlabel("Ângulo")
    plt.ylabel("Valor")
    plt.hist(arr_theta_1[i], bins, label='Teste 1 - reto')
    plt.hist(arr_theta_2[i], bins, label='Teste 2 - ' + str(grus_inclinacao) + 'º de inclinação')
    plt.legend(loc='upper right')
    plt.savefig(url_crops + "hist/hist_crop_" + str(i) + ".png")
    plt.clf()
    plt.title("Histograma crop " + str(i))
    #plt.show()

    
print("FIM")

cv.waitKey(0)
