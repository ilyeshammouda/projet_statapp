'''
Ce code définit les classes qui vont être utilisées par la suite dans le projet 
'''



import numpy as np
import pylops
from pylops.optimization.sparsity import ISTA
from pylops.optimization.sparsity import FISTA
import pandas as pd
import time
from google.colab.patches import cv2_imshow
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import time
import pywt
import cv2
from google.colab.patches import cv2_imshow
from skimage import io, color
import pywt
from PIL import Image



'''
Ce code définit les classes qui vont être utilisées par la suite dans le projet 
'''


#la Classe random contient les fonctions qui simulent les variables aléatoires qui seront utilisée pour étudier les données simulés.

print("test")
class random:  
    def matrix_normal(n,p,mu=0,sigma=1):  # n est le nombre de lignes et p le nombre des colonnes, mu est la moyenne et sigma est l'écart type
        return (np.random.randn(n,p)*(sigma**2))+mu
    def vect_normal(n,mu=0,sigma=1):
        return (np.random.randn(n)*(sigma**2))+mu
    def beta(a,s,n): # s et a sont à préciser tel que s= 0,1*p et n> 2*s*log(p/2) pour commencer on peut utilisr a=1
        return a*(np.random.binomial(1,s/n , size=(n,)))
    def outcome(n,p,a,s,mu=0,sigma=1):
        X=random.matrix_normal(n,p,mu,sigma)
        beta=random.beta(a,s,p)
        epsilon=random.vect_normal(n,mu,sigma)
        Y=X @ beta+epsilon
        return Y,X,beta,epsilon


#la classe algo contient les algorithmes qui seront utilisés nottament ISTA et IHT
class algo:


    def HardThreshold(x,lamda):
        return x*(np.abs(x)>=lamda)
    def SoftThreshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


    def IHT(X, Y,beta=np.zeros(1) ,C=0.9,step=0.0001,max_iterations=3000,lamda=0.1, tol=1e-6,sparse='False'):
        n,m=X.shape
        Z,Beta=np.zeros(m),np.ones(m)
        loss=[]
        cost=[]
        check_vect=np.zeros(m)
        test=np.zeros(1)
        if np.array_equal(beta, test, equal_nan=False):
          beta=Beta.copy()
          print('We are in the unknow beta case,the cost function is not significant')
        start_time = time.time()        
        for i in range(max_iterations):
            Z=Beta+(step*(X.T)@(Y-X@Beta))
            Beta=algo.HardThreshold(Z, lamda)
            if sparse=='True':
                Beta= np.where(np.isclose(Beta, 1, atol=0.8), 1, 0)
            cost.append(np.linalg.norm(-beta+Beta))
            lamda*=C
            loss.append(Beta[-1]-Beta[-2])
            if np.linalg.norm(Beta -check_vect ) < tol:
                break
        end_time = time.time()
        time_taken = end_time - start_time
        print("IHT execution time :", time_taken, "seconds")

        return Beta,cost,loss

    def ISTA(X, Y,beta=np.zeros(1) ,step=0.0001,max_iterations=3000,lamda=0.01, tol=1e-6,sparse='False'):
        n,m=X.shape
        Z,Beta=np.zeros(m),np.ones(m)
        check_vect=np.zeros(m)
        test=np.zeros(1)
        cost=[]
        loss=[]
        if np.array_equal(beta, test, equal_nan=False):
          beta=Beta.copy()
          print('We are in the unknow beta case,the cost function is not significant')
        start_time = time.time()
        for i in range(max_iterations):
            Z=Beta+(step*(X.T)@(Y-X@Beta))
            Beta=algo.SoftThreshold(Z, lamda)
            if sparse=='True':
              Beta= np.where(np.isclose(Beta, 1, atol=0.8), 1, 0)
            cost.append(np.linalg.norm(-beta+Beta))
            loss.append(Beta[-1]-Beta[-2])
            if np.linalg.norm(Beta -check_vect ) < tol:
                break
        end_time = time.time()
        time_taken = end_time - start_time
        print("ISTA execution time :", time_taken, "seconds")
        return Beta,cost,loss
class image_processing:
    def image_tansformation_to_spare_vector_BW(img_path, wavelet_type='haar', threshold=20):
    # telecharger l'image
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # on applique la transofomation en ondellettes de dimension 2
      coeffs = pywt.dwt2(img, wavelet_type)
      cA, (cH, cV, cD) = coeffs
    #on stock les dimention de ces vecteurs afin de pouvoir les récuperer dans un second lieu
      coeff_size=[cA.size,cH.size,cV.size]
      coeff_shape=[cA.shape,cH.shape,cV.shape,cD.shape]
    # construire notre vecteur sparse qui contient toutes les infomations
      coef_vec = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
      coef_vec[np.abs(coef_vec) < threshold] = 0
      return coef_vec,coeff_size,coeff_shape


    def image_from_sparse_vector_BW(coef_vec,coeff_size,coeff_shape, wavelet_type='haar'):
    # on récupere les vecteurs que la transofomation en ondelettes a sorti
      cA_size=coeff_size[0]
      cH_size=coeff_size[1]
      cV_size=coeff_size[2]
      cA_thresh = coef_vec[:cA_size].reshape(coeff_shape[0])
      cH_thresh = coef_vec[cA_size:cA_size+cH_size].reshape(coeff_shape[1])
      cV_thresh = coef_vec[cA_size+cH_size:cA_size+cH_size+cV_size].reshape(coeff_shape[2])
      cD_thresh = coef_vec[cA_size+cH_size+cV_size:].reshape(coeff_shape[2])
      coeffs_thresh = cA_thresh, (cH_thresh, cV_thresh, cD_thresh)
      # on reconstruit l'image
      img_thresh = pywt.idwt2(coeffs_thresh, wavelet_type)
      return img_thresh


    def compress_image_RGB(img_path, wt='haar',threshold=20 ,n_lev=2, compression_value=0):
    # Load image
      imgRGB = io.imread(img_path)

    # Convert to YCbCr
      imgYCbCR = color.rgb2ycbcr(imgRGB)
      Y = imgYCbCR[:, :, 0]
      Cb = imgYCbCR[:, :, 1]
      Cr = imgYCbCR[:, :, 2]

    # Perform wavelet transform
      Y_coeff_arr, Y_coeff_slices = pywt.coeffs_to_array(pywt.wavedec2(Y, wavelet=wt, level=n_lev))
      Cb_coeff_arr, Cb_coeff_slices = pywt.coeffs_to_array(pywt.wavedec2(Cb, wavelet=wt, level=n_lev))
      Cr_coeff_arr, Cr_coeff_slices = pywt.coeffs_to_array(pywt.wavedec2(Cr, wavelet=wt, level=n_lev))
      Y_coeff_arr,Cb_coeff_arr, Cr_coeff_arr, = np.array(Y_coeff_arr) ,np.array(Cb_coeff_arr), np.array(Cr_coeff_arr)


    # Compress wavelet coefficients
      Y_coeff_arr_filt = np.where(np.abs(Y_coeff_arr) > compression_value * np.max(np.abs(Y_coeff_arr)),Y_coeff_arr,0)
      Cb_coeff_arr_filt = np.where(np.abs(Cb_coeff_arr) > compression_value * np.max(np.abs(Cb_coeff_arr)), Cb_coeff_arr, 0)
      Cr_coeff_arr_filt = np.where(np.abs(Cr_coeff_arr) > compression_value * np.max(np.abs(Cr_coeff_arr)), Cr_coeff_arr, 0)
      coeff_size=[Y_coeff_arr_filt.size,Cb_coeff_arr_filt.size,Cr_coeff_arr_filt.size]

      coeff_shape=[Y_coeff_arr_filt.shape, Cb_coeff_arr_filt.shape, Cr_coeff_arr_filt.shape]
      coef_vec=np.concatenate((Y_coeff_arr_filt.flatten(), Cb_coeff_arr_filt.flatten(),Cr_coeff_arr_filt.flatten()))
      coef_vec[np.abs(coef_vec) < threshold] = 0
      slices=[Y_coeff_slices,Cb_coeff_slices,Cr_coeff_slices]
      return coef_vec, slices,coeff_size,coeff_shape



    def decompress_photo_RGB(coef_vec,slices,coeff_size,coeff_shape,wt='haar'):
      Y_shape = coeff_shape[0]
      Cb_shape = coeff_shape[1]
      Cr_shape = coeff_shape[2]
      #reconstruct matrices
      Y_coeff_slices = slices[0]
      Cb_coeff_slices = slices[1]
      Cr_coeff_slices = slices[2]
  
      Y_coeff_arr_filt = coef_vec[0:coeff_size[0]].reshape(Y_shape)
      Cb_coeff_arr_filt = coef_vec[coeff_size[0]:coeff_size[0] + coeff_size[1]].reshape(Cb_shape)
      Cr_coeff_arr_filt = coef_vec[coeff_size[0] + coeff_size[1]:].reshape(Cr_shape)
      # Reconstruct images from compressed wavelet coefficients
      Y_compressed = pywt.waverec2(pywt.array_to_coeffs(Y_coeff_arr_filt, Y_coeff_slices, output_format='wavedec2'), wavelet=wt)
      Cb_compressed = pywt.waverec2(pywt.array_to_coeffs(Cb_coeff_arr_filt, Cb_coeff_slices, output_format='wavedec2'), wavelet=wt)
      Cr_compressed = pywt.waverec2(pywt.array_to_coeffs(Cr_coeff_arr_filt, Cr_coeff_slices, output_format='wavedec2'), wavelet=wt)

      # Convert back to RGB and stack the channels
      img_compressed = color.ycbcr2rgb(np.dstack((Y_compressed, Cb_compressed, Cr_compressed)))
      img_compressed = Image.fromarray((img_compressed* 255).astype(np.uint8))

      return img_compressed