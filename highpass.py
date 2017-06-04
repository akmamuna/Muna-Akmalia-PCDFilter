import numpy as np #memanggil library numpy
import cv2 #memanggil library opencv
import matplotlib.pyplot as plt #memanggil library matplotlib
from scipy import ndimage #memangil library ndimage dari scipy


im = cv2.imread('E:\kuliah\mawar.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
data = np.array(gray, dtype=float)

kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
highpass_5x5 = ndimage.convolve(data, kernel)

hist1,bins1 = np.histogram(highpass_5x5.flatten(),256,[0,256]) #
cdf1 = hist1.cumsum() # membuat histogram gambar dari hasil low pass filter pada lpf
norm1 = cdf1 * hist1.max()/ cdf1.max()

cv2.imshow('Grayscale',gray)
cv2.imshow('Highpass_5x5',highpass_5x5)
plt.plot(norm1, color = 'g') #memberi warna hijau pada tampilan histo
plt.hist(highpass_5x5.flatten(),256,[0,256], color = 'b') #memberi warna biru pada tampilan histo
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
