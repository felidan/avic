import cv2 as cv
import numpy as np
import Hough as ho
import AVIC as av

def check_slope(nm_im1, nm_im2, min_gap, max_gap, lim_1, lim_2, variation, save, name, limited):
    im1 = cv.imread(nm_im1)
    im2 = cv.imread(nm_im2)

    [img_1, matrizLinhas_1, matrizBinaria_1, variacaoMedia_1, xMedio_1, k_1, t_1,tamanhoMedio_1] = ho.HoughLinesProbabilisticoVariavel(im1,
                                                                                                cor = (255,0,0),
                                                                                                espes_linha = 1,
                                                                                                limiar=lim_1,
                                                                                                minLineGap=min_gap,
                                                                                                maxGap = max_gap,
                                                                                                cannyMinVal=75,
                                                                                                cannyMaxVal=150,
                                                                                                variacao=variation,
                                                                                                angulo=90,
                                                                                                limited=limited)
    
    print('Processing image ' + nm_im1 + '..')
    print('Average theta: ' + str(round(t_1, 5)))

    [img_2, matrizLinhas_2, matrizBinaria_2, variacaoMedia_2, xMedio_2, k_2, t_2, tamanhoMedio_2] = ho.HoughLinesProbabilisticoVariavel(im2,
                                                                                                cor = (0,0,255),
                                                                                                espes_linha = 1,
                                                                                                limiar=lim_2,
                                                                                                minLineGap=min_gap,
                                                                                                maxGap = max_gap,
                                                                                                cannyMinVal=75,
                                                                                                cannyMaxVal=150,
                                                                                                variacao=variation,
                                                                                                angulo=90,
                                                                                                limited=limited)
    
    print('\nProcessing image ' + nm_im2 + '..')
    print('Average theta: ' + str(round(t_2, 5)))
    
    print('\nvariation detected: ' + str(round(abs(t_1 - t_2), 5)))

    if(save == "S"):
        cv.imwrite('_result\_t1_' + name + '.jpg', img_1)
        cv.imwrite('_result\_t2_' + name + '.jpg', img_2)
        print("Saved")
        
    return [img_1, img_2]



#[im1, im2] = check_slope('_base\T1_building.png', '_base\T2_building.png', 80, 80, 100, 120, 5, "S", "predio", "N")
[im1, im2] = check_slope('_base\T1_building_santos.jpg', '_base\T2_building_santos.jpg', 10, 30, 50, 50, 5, "S", "santos", "S")
