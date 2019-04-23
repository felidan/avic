import numpy as np
import cv2

def salva_frames(url_video, url_destino, intervalo, i, f):
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
                cv2.imwrite(url_destino + "frame_%d.jpg" % count, image)
            count += 1
            i_temp += intervalo
        else:
            success = False
    print('Frame: ', count)
            
def empilha_imagem(img1, img2):
    imagem_comparacao = np.hstack((img1, img2))
    return imagem_comparacao

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
