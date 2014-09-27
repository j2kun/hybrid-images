import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math

def scaleSpectrum(A):
   return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))


# make a spherical gaussian function from the center of the image
def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)

   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient

   return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])


def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   misc.imsave("dft.png", scaleSpectrum(shiftedDFT))

   filteredDFT = shiftedDFT * filterMatrix # assume filterMatrix is properly normalized
   misc.imsave("filtered-dft.png", scaleSpectrum(filteredDFT))
   return ifft2(ifftshift(filteredDFT))


def lowPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))


def highPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))


if __name__ == "__main__":
   marilyn = ndimage.imread("marilyn.png", flatten=True)
   einstein = ndimage.imread("einstein.png", flatten=True)

   highPassedMarilyn = highPass(marilyn, 20)
   lowPassedMarilyn = lowPass(marilyn, 20)

   misc.imsave("low-passed-marilyn.png", numpy.real(lowPassedMarilyn))
   misc.imsave("high-passed-marilyn.png", numpy.real(highPassedMarilyn))
   misc.imsave("sum-of-marilyns.png", numpy.real((highPassedMarilyn + lowPassedMarilyn)/2.0))

