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
