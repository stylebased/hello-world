import numpy as np
import cv2
from matplotlib import pyplot as plt

imgt = cv2.imread('test_0052_aligned.png',0)

rows, cols = imgt.shape[:2]
crow, ccol = int(rows/2) , int(cols/2) 
mask = np.ones((rows, cols), np.uint8)
mask[crow-5:crow+5, ccol-5:ccol+5] = 0

img = imgt
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fshift = fshift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.figure("Filtered Value Channel of the ppp", figsize=(10,10))
plt.imshow(img_back)
plt.savefig('abcd.png')

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()

out_img = np.zeros((1024,1024,3), np.uint8)
for i in range(3):
    img = imgt[:,:,i]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift = fshift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    out_img[:,:,i] = img_back


plt.subplot(121),plt.imshow(cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB))
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.image as mpimg
import cv2
from scipy import fftpack


def extractValueChannel(image):
    try:
        # Check if it has three channels or not 
        np.size(image, 2)
    except:
        return image
    hsvImage = col.rgb_to_hsv(image)
    return hsvImage[..., 2]

def generateFilter(image,w,h, filtType):
    if w > 0.5 or h > 0.5:
        print("w and h must be < 0.5")
        exit()
    m = np.size(image,0)
    n = np.size(image,1)
    LPF = np.zeros((m,n))
    HPF = np.ones((m,n))
    xi = np.round((0.5 - w/2) * m)
    xf = np.round((0.5 + w/2) * m)
    yi = np.round((0.5 - h/2) * n)
    yf = np.round((0.5 + h/2) * n)
    LPF[int(xi):int(xf),int(yi):int(yf)] = 1
    HPF[int(xi):int(xf),int(yi):int(yf)] = 0
    if filtType == "LPF":
        return LPF
    elif filtType == "HPF":
        return HPF
    else:
        print("Only Ideal LPF and HPF are supported")
        exit()

image = mpimg.imread("test_0052_aligned.png")
plt.figure("Original Image", figsize=(10,10))
plt.imshow(image)
plt.show()

valueChannel = extractValueChannel(image)
plt.figure("Value Channel of the Image", figsize=(10,10))
plt.imshow(valueChannel,cmap='gray')
plt.show()

valueChannel = extractValueChannel(image)
FT = fftpack.fft2(valueChannel)
plt.figure("Fourier Transform of the Image", figsize=(10,10))
plt.imshow(np.log(1+np.abs(FT)))
plt.set_cmap("gray")
plt.show()

ShiftedFT = fftpack.fftshift(FT)
plt.figure("Fourier Transform of the Image", figsize=(10,10))
plt.imshow(np.log(1+np.abs(ShiftedFT)))
plt.set_cmap("gray")
plt.show()

LPF = generateFilter(ShiftedFT,0.05, 0.05, "LPF")
plt.figure("Ideal Low Pass Filter in frequency domain", figsize=(10,10))
plt.imshow(ind,cmap='gray')
plt.show()

filteredVChannel = np.abs(fftpack.ifft2(LPF * ShiftedFT))
plt.figure("Filtered Value Channel of the Image", figsize=(10,10))
plt.imshow(filteredVChannel)
plt.savefig('abc.png')


import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray

img1 = imread('test_0052_aligned.png')
img1_gray = rgb2gray(img1)
img2 = imread('test_0053_aligned.png')
img2_gray = rgb2gray(img2)

img1_fft = np.fft.fftshift(np.fft.fft2(img1_gray))
img2_fft = np.fft.fftshift(np.fft.fft2(img2_gray))

fig, ax = plt.subplots(2,2,figsize=(15,15))
ax[0,0].imshow(img1_gray, cmap='gray')
ax[0,0].set_title('img1')
ax[0,1].imshow(np.log(abs(img1_fft)), cmap='gray')
ax[0,1].set_title('FFT of img1')

ax[1,0].imshow(img2_gray, cmap='gray')
ax[1,0].set_title('img2')
ax[1,1].imshow(np.log(abs(img2_fft)), cmap='gray')
ax[1,1].set_title('FFT of img2')

def rgb_transform(img, mask):
    res = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2(
                                  img[:, :, i]))
        rgb_fft *= mask
        rgb = abs(np.fft.ifft2(rgb_fft))
        res.append(rgb.clip(0, 255).astype(int))
    return np.dstack(res)

rows, cols = img1.shape[:2]
crow, ccol = int(rows/2) , int(cols/2) 
hmask = np.zeros((rows, cols), np.uint8)
hmask[crow-30:crow+30, ccol-30:ccol+30] = 1
lmask = 1-hmask

img1h = rgb_transform(img1,hmask)
img1l = rgb_transform(img1,lmask)
img2h = rgb_transform(img2,hmask)
img2l = rgb_transform(img2,lmask)

img2t = rgb_transform(img2,hmask)

figure, axis = plt.subplots(3,2, figsize=(20,20))
axis[0][0].imshow(hmask, cmap='gray')
axis[0][0].set_title('High Pass Filter')
axis[0][1].imshow(lmask, cmap='gray')
axis[0][1].set_title('Low Pass Filter')
axis[1][0].imshow(img1h)
axis[1][1].imshow(img1l)
axis[2][0].imshow(img2h)
axis[2][1].imshow(img2l)
plt.savefig('img3.png')

A = img2
B = np.mean(A, -1)
Bt = np.fft.fft2(B)
Btsort = np.sort(np.abs(Bt.reshape(-1)))
figure, axis = plt.subplots(4,2, figsize=(10,10))
for i,keep in enumerate([0.1, 0.05, 0.01, 0.002]):
    thresh = Btsort[int(np.floor((1-keep)*len(Btsort)))]
    ind = np.abs(Bt)>thresh          # Find small indices
    Atlow = Bt * ind                 # Threshold small indices
    Alow = np.fft.ifft2(Atlow).real  # Compressed image
    axis[i][0].imshow(ind,cmap='gray')
    axis[i][0].axis('off')
    axis[i][1].imshow(Alow,cmap='gray')
    axis[i][1].axis('off')
    axis[i][1].set_title('Compressed image: keep = ' + str(keep))

