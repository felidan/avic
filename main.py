import cv2 as cv
import AVIC as av
import Mask as ms
import GetHeight as ht
import Slope as sl

###################
#       AVIC      #
###################

url = '_base/video/video12.mp4'
url_dest = '_base/crop/'
x, y, h, w = 230, 320, 600, 900
img_t1 = ''
img_t2 = ''
print('-- Begin')

print('\nGet frames..')
av.salva_frames(url, url_dest, 'frame_video_12_', 0.2, 30, 50)

print("Frame average..")
img_t1 = av.media_frames(url_dest, 'frame_video_12_', '.jpg', 24, 32)
img_t2 = av.media_frames(url_dest, 'frame_video_12_', '.jpg', 64, 80)

copy_t1 = img_t1.copy()
img_t1 = copy_t1[y:y+h, x:x+w]

copy_t2 = img_t2.copy()
img_t2 = copy_t2[y:y+h, x:x+w]

print("Equalization..\n")
img_t1 = cv.equalizeHist(img_t1)
img_t2 = cv.equalizeHist(img_t2)

cv.imwrite('_base/crop/_crop_video12_media_1.png', img_t1)
cv.imwrite('_base/crop/_crop_video12_media_2.png', img_t2)

print('\n------------> Mask')
im1 = ms.get_mask(url='_base/crop/crop_video12_media_2.png',
                  img=img_t1,
                  save="S",
                  y=0,
                  x=0,
                  h=0,
                  w=0,
                  numero=1,
                  morf_open="N",
                  morf_kernel_1=2,
                  morf_kernel_2=5)

im2 = ms.get_mask(url='_base/crop/crop_video12_media_4.png',
                  img=img_t2,
                  save="S",
                  y=0,
                  x=0,
                  h=0,
                  w=0,
                  numero=2,
                  morf_open="N",
                  morf_kernel_1=2,
                  morf_kernel_2=5)

print('\n------------> Height')
ht.get_height('_MASC.png')

print('\n------------> Slope')
[im1_1, im2_1] = sl.check_slope('_base\T1_building.png', '_base\T2_building.png', 80, 80, 100, 120, 5, "S", "predio", "N")
[im1_2, im2_2] = sl.check_slope('_base\T1_building_santos.jpg', '_base\T2_building_santos.jpg', 10, 30, 50, 50, 5, "S", "santos", "S")

print('\n------------> Finish')
print('\nResults in -> _base/')

print('\n-- End')
