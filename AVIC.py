import numpy as np
import cv2

# Salva frames de um video
def salva_frames(url_video, url_destino, nome_frame, intervalo, i, f):
    video_cap = cv2.VideoCapture(url_video)
    count = 0
    i_temp = i
    success = True

    while success:
        if(i_temp <= f):
            # Configura intervalo entre frames
            video_cap.set(cv2.CAP_PROP_POS_MSEC,(i_temp*1000)) 
            # Lê o frame
            success, image = video_cap.read()
            if(success):
                # Salva o frame
                cv2.imwrite(url_destino + nome_frame + "%d.jpg" % count, image)
            count += 1
            i_temp += intervalo
        else:
            success = False
    print('Frame: ', count)

# Empilha duas imagens         
def empilha_imagem(img1, img2):
    imagem_comparacao = np.hstack((img1, img2))
    return imagem_comparacao

# Realiza a média de frames
def media_frames(url_frames, nome_imagem, formato_img, frame_ini, frame_fin):
    # Pega o primeiro frame para obter as dimensões
    url_img = url_frames + nome_imagem + str(frame_ini) + formato_img
    im0 = cv2.imread(url_img)
    
    [lin, col, c] = im0.shape
    im_final = np.zeros((lin, col))

    # Lê frames e soma imagens
    while(frame_ini <= frame_fin):
        url = url_frames + nome_imagem + str(frame_ini) + formato_img
        im_temp = cv2.imread(url)
        im_temp = cv2.cvtColor(im_temp, cv2.COLOR_BGR2GRAY)
        im_final += im_temp
        frame_ini += 1

    im_final = (im_final/frame_fin)
    
    return np.uint8(im_final)

# Realiza crop da imagem
def crop_imagem(img, url_destino, nome, formato, cropV, cropH, salvar = "N"):
    arr = []
    cont = 0
    vertical_ant = 0
    horizontal_ant = 0
    deltaV = int(img.shape[0]/cropV)
    deltaH = int(img.shape[1]/cropH)

    for i in range(0, cropV):
        for j in range(0, cropH):
            x = img[vertical_ant:(vertical_ant+deltaV), horizontal_ant:(horizontal_ant+deltaH)]
            if(salvar == "S"):
                cv2.imwrite(url_destino + nome + str(cont) + formato, x)
            arr.append( img[vertical_ant:(vertical_ant+deltaV), horizontal_ant:(horizontal_ant+deltaH)]    )
            cont += 1
            horizontal_ant += deltaH
        vertical_ant += deltaV
        horizontal_ant = 0
    return arr

def rotacionaImagem(img, angle):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def cortaImagem(imagem, pc_vertical, pc_horizontal):
    vertical = int(imagem.shape[0])
    horizontal = int(imagem.shape[1])
    v = int(vertical*pc_vertical)
    h = int(horizontal*pc_horizontal)
    imagem_cortada = imagem[v:vertical-v, h:horizontal-h]
    return imagem_cortada

def bordas_maior_regiao(im, borda = 3):

    mask = np.zeros_like(im)
    try:
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    except:
        imgray = np.copy(im)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    
    # detecta bordas
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # pega apenas a maior
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cv2.drawContours(mask, contours, max_index, (255, 0, 0), borda)

    return mask

def mask_imagem(img, url, salvar="S"):
    _imgBefore = np.copy(img)
    # ---------------CLAHE-----------------------------------------------------------------------------------------#
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clBefore = clahe.apply(img)
    cv2.imwrite(url + '1_1_CLAHE.png', clBefore)
    # ----------------Blur-----------------------------------------------------------------------------------------#
    blurBefore = cv2.GaussianBlur(clBefore,(5,5),0)
    cv2.imwrite(url + '2_1_BLUR.png', blurBefore)

    # ----------------Morfologia-----------------------------------------------------------------------------------#
    kernel = np.ones((7,2), np.uint8)
    morfBefore=cv2.morphologyEx(blurBefore, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10)))
    cv2.imwrite(url + '3_1_MORF.png', morfBefore)
    # --------------- Thresh --------------------------------------------------------------------------------------#
    threshBefore = cv2.adaptiveThreshold(morfBefore,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,35,20)
    cv2.imwrite(url + '4_1_THRESH.png', threshBefore)

    # ---------------- Canny ------------------------------------------------------------------------------------- #
    sigma=0.33

    # construct the argument parse and parse the arguments
    blurredBefore = threshBefore

    # apply Canny edge detection using a wide threshold, tight threshold, and automatically determined threshold
    wideBefore = cv2.Canny(blurredBefore, 100, 200) #10, 200
    tightBefore = cv2.Canny(blurredBefore, 225, 250) #225, 250

    # compute the median of the single channel pixel intensities
    vBefore = np.median(blurredBefore)

    # apply automatic Canny edge detection using the computed median
    lowerB = int(max(0, (1.0 - sigma) * vBefore))
    upperB = int(min(255, (1.0 + sigma) * vBefore))
    cannyBefore = cv2.Canny(blurredBefore, lowerB, upperB)

    cv2.imwrite(url + '5_1_CANNY.png', cannyBefore)

    
    # ------------------ MORFOLOGIA ------------------------------------------------------------------------------#
    kernel = np.ones((23,10),np.uint8) #23,10
    morfB = cv2.morphologyEx(cannyBefore, cv2.MORPH_CLOSE, kernel)
    
    kernel = np.ones((8,2),np.uint8) #8,1
    morfB = cv2.morphologyEx(morfB, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(url + '6_2_MORF.png', morfB)
    # ----------------------- CONTORNO E PREENCHIMENTO-----------------------------------------------------------#
    im2, contours, hierarchy = cv2.findContours(morfB,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(morfB,[cnt],0,255,-1)
    cv2.imwrite(url + '7_1_CONTORNO.png', morfB)
    # ---------------------- MASCARA ----------------------------------------------------------------------------#
    imgFinalBefore = cv2.bitwise_and(_imgBefore,_imgBefore,mask = morfB)
    cv2.imwrite(url + '_MASCARA.png', imgFinalBefore)
    
    return imgFinalBefore






























    
