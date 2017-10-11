# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:01:34 2017

@author: hespo
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import time

from PIL import Image
import os
os.chdir("C:/Users/hespo/OneDrive/Documentos/GitHub/Trab2_DataScience/gs")
cwd = os.getcwd()
print (cwd)
img = Image.open('2.jpg')
imggray = img.convert('LA')
plt.figure(figsize=(9, 6))
plt.imshow(imggray);

imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap='gray');


print ("Matriz Original da Foto", imgmat.shape)
Frobenius_Ori= LA.norm(imgmat, 'fro')


U, sigma, V = np.linalg.svd(imgmat)

print ("Matriz U", U.shape)
print ("Matriz sigma", sigma.shape)  ### Matriz com Valores Singulares
print ("Matriz V", V.shape)
print ("")

l5=5*len(sigma)/100
l10=10*len(sigma)/100
l25=25*len(sigma)/100
l50=50*len(sigma)/100

print ("5%",l5)
print ("10%",l10)
print ("25%",l25)
print ("50%",l50)


reconstimg = np.matrix(U[:, :l5]) * np.diag(sigma[:l5]) * np.matrix(V[:l5, :])
Frobenius_5= LA.norm(reconstimg, 'fro')
title = "n = %s" % l5
plt.title(title)
plt.imshow(reconstimg, cmap='gray');
plt.show()

reconstimg = np.matrix(U[:, :l10]) * np.diag(sigma[:l10]) * np.matrix(V[:l10, :])
Frobenius_10= LA.norm(reconstimg, 'fro')
title = "n = %s" % l10
plt.title(title)
plt.imshow(reconstimg, cmap='gray');
plt.show()

reconstimg = np.matrix(U[:, :l25]) * np.diag(sigma[:l25]) * np.matrix(V[:l25, :])
Frobenius_25= LA.norm(reconstimg, 'fro')
title = "n = %s" % l25
plt.title(title)
plt.imshow(reconstimg, cmap='gray');
plt.show()

reconstimg = np.matrix(U[:, :l50]) * np.diag(sigma[:l50]) * np.matrix(V[:l50, :])
Frobenius_50= LA.norm(reconstimg, 'fro')
title = "n = %s" % l50
plt.title(title)
plt.imshow(reconstimg, cmap='gray');
plt.show()

print (Frobenius_Ori,Frobenius_5,Frobenius_10,Frobenius_25,Frobenius_50)
#reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
#plt.imshow(reconstimg, cmap='gray');

#for i in range(2, 4):
 #   reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
 #  plt.imshow(reconstimg, cmap='gray')
 #   title = "n = %s" % i
 #   plt.title(title)
  #  plt.show()

#for i in range(5, 51, 5):
 #   reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
  #  plt.imshow(reconstimg, cmap='gray')
   # title = "n = %s" % i
    #plt.title(title)
   # plt.show()
    
print ("FIM")
