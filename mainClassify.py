#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:29:24 2018

@author: mdsamad
"""




import numpy as np
import scipy.io as sio
import multi_modules as mclf 
from sklearn.model_selection import KFold


# Loading .*mat files from Matlab
matFile01 = sio.loadmat('hog_250by250_features_noOpenOrTongue.mat')
matFile02 = sio.loadmat('all_labels_noOpenOrTongue.mat')

feaMat = matFile01 ['hog_features']
labels = matFile02['labels']


kf = KFold(n_splits=10, random_state=42, shuffle= True)



#meanAUC, stdAUC, confusion_matrix = mclf.SVM_Linear(feaMat,labels,kf)  

#meanAUC01, stdAUC01, confusion_matrix01 = mclf.Rand_Forest (feaMat,labels,kf)    

#meanAUC02, stdAUC02, confusion_matrix02 = mclf.GradBoost(feaMat,labels,kf)    

meanAUC03, stdAUC03, confusion_matrix03 = mclf.SVM_RBF(feaMat,labels,kf) 