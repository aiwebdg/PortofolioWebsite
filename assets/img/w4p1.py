
# program import citra ke python
import cv2 as cv # manipulasi citra
import numpy as np
from PIL import Image

# nama file citra: lena.jpg tersimpan di directory yang sama dengan program
# python menyimpan citra RGB dengan urutan BGR
bgr = cv.imread('lena.jpg')

# ubah ukuran citra agar mudah diamati pada layar dalam persentase
persen = 50 # persen
lebar = int(bgr.shape[1] * persen / 100)
tinggi = int(bgr.shape[0] * persen / 100)
bgr0 = cv.resize(bgr, (lebar, tinggi))

# menampilkan citra yang telah dibaca/import
cv.imshow('citra', bgr0) # nama window, data citra

# memisahkan komponen warna
bgr = bgr0.astype('float')
B = (bgr[:, :, 0])
G = (bgr[:, :, 1])
R = (bgr[:, :, 2])

# ubah ke dalam ruang warna HSV
H = np.zeros([bgr.shape[0], bgr.shape[1]], dtype = float)
S = np.zeros([bgr.shape[0], bgr.shape[1]], dtype = float)
V = np.zeros([bgr.shape[0], bgr.shape[1]], dtype = float)
HSV = np.zeros((bgr.shape[0], bgr.shape[1], 3), 'uint8')

for i in range(bgr.shape[0]):
    for j in range(bgr.shape[1]):
        
        Bnorm = B[i][j]/float(255) # normalisasi
        Gnorm = G[i][j]/float(255)
        Rnorm = R[i][j]/float(255)

        Cmax = max(Bnorm, Gnorm, Rnorm) # nilai maksimum dari semua komponen warna
        Cmin = min(Bnorm, Gnorm, Rnorm) # nilai minimum dari semua komponen warna
        delta = Cmax - Cmin

        # hitung H
        if (delta == 0):
            H[i][j] = 0
        elif (Rnorm == Cmax):
            H[i][j] = (60*((Gnorm-Bnorm)/delta)+0 % 6)
        elif (Gnorm == Cmax):
            H[i][j] = (60*((Bnorm-Rnorm)/delta)+2 )
        elif (Bnorm == Cmax):
            H[i][j] = (60*((Rnorm-Gnorm)/delta)+4 )

        # hitung S
        if (Cmax == 0):
            S[i][j] = 0
        else:
            S[i][j] = (delta / Cmax) * 100

        # hitung V
        V[i][j] = Cmax*100

# menampilkan citra yang telah dibaca/import
H = H.astype('uint8')
S = S.astype('uint8')
V = V.astype('uint8')
HSV[..., 0] = H
HSV[..., 1] = S
HSV[..., 2] = V
cv.imshow('HSV ori', HSV) # nama window, data citra

# RGB to HSV dengan library opencv
HSV2 = cv.cvtColor(bgr0, cv.COLOR_RGB2HSV)
cv.imshow('HSV Library', HSV2)

# simpan hasil
cv.imwrite('HSV ori.jpg', HSV)
cv.imwrite('HSV lib.jpg', HSV2)

# menutup semua window imshow ketika ada penekanan tombol di keyboard
cv.waitKey(0)
cv.destroyAllWindows()
