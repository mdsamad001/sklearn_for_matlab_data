#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:37:03 2018

@author: mdsamad
"""
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import ( pipeline, preprocessing)
from sklearn.svm import SVC

from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier

    
#from sklearn.cross_validation import StratifiedKFold

import numpy as np
from sklearn.preprocessing import label_binarize



def GradBoost (Xm, labels, kf):    
    cnmat = np.zeros((7,7))
    labels = labels -1 

    mauc=[]    
    for (train, test) in kf.split(Xm): #enumerate (kf):
        
        trfold = Xm[train,:] 
        tsfold =  Xm[test,:] 
        ytr = labels [train]
        yts = labels [test]
      
        
    
        # Random forest 
        forestX = GradientBoostingClassifier(n_estimators= 5,
                                learning_rate=0.1,
                                random_state=1)
    
    
                        
        param_grid = [{'n_estimators': list(range(10, 100,20)),
                       'learning_rate': [0.05,0.1,0.5]}]
    
        gs_i = GridSearchCV(estimator=forestX, param_grid = param_grid, 
                            n_jobs=1, cv = 2)
        
        
       
        gs = gs_i.fit(trfold, ytr.ravel())
        print(gs.best_params_)
    
    
        forest = gs.best_estimator_
       
        y_pred = forest.predict(tsfold)
        probas = forest.predict_proba(tsfold)
        
        cnf_matrix = metrics.confusion_matrix(yts, y_pred)
        
        cnmat = cnmat+cnf_matrix
        
       # print (cnf_matrix)
        roc_auc = []
    
        for m in range(7):
            
            fpr0, tpr0, _ = metrics.roc_curve(yts.ravel(), probas[:,m], pos_label=m)
            roc_auc.append (metrics.auc(fpr0,tpr0))
            
        print ('AUC at this fold',np.mean(roc_auc))
        mauc.append(np.mean(roc_auc)) # Mean AUCs over three groups
       
    print(cnmat)    
        
    print ('Mean AUC is:',np.mean(mauc))
    print ('Standard deviation of auc is:', np.std(mauc))
    
    return np.mean(mauc), np.std(mauc), cnmat


def SVM_RBF (Xm, labels,kf):
 
    cnmat = np.zeros((7,7))
    mauc=[]
    for (train, test) in kf.split(Xm): #enumerate (kf):
    
        trGs = Xm[train,:] 
        tsGs =  Xm[test,:] 
        ytr = labels [train]
        yts = labels [test]
      
    
        c_rng = [0.05, 0.1, 0.5, 1, 2, 5, 10]
    
        g_rng = [0.05, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
    
        
    
        pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                                     #('feat', top_feat),
        ('clf', SVC(kernel='rbf', class_weight='balanced', random_state=1))])
    
       
        param_grid = [ {
               'clf__C':c_rng,
                'clf__gamma': g_rng}]
         
        gs_i = GridSearchCV(estimator=pipe, 
                            param_grid = param_grid, n_jobs=1, cv = 2)
        
        
        gs = gs_i.fit(trGs, ytr.ravel())
        print(gs_i.best_params_)
        
        
        SVMRBF = gs.best_estimator_
    
        # Confusion matrix calculation
        
        ypred = SVMRBF.predict(tsGs)
    
        cnf_matrix = metrics.confusion_matrix(yts, ypred)
    
        cnmat = cnmat+cnf_matrix
        
        
        # ROC AUC calculation
        y_pred = SVMRBF.decision_function(tsGs)
    
        roc_auc = []
       
        yts= label_binarize(yts, classes=[1,2,3,4,5,6,7])
       
        for m in range(7):
        
            fpr0, tpr0, _ = metrics.roc_curve(yts[:,m], y_pred[:,m])
            roc_auc.append (metrics.auc(fpr0,tpr0))
     
        print ('AUC at this fold',np.mean(roc_auc))
        mauc.append(np.mean(roc_auc)) # Mean AUCs over three groups
       
        #print(mauc)
    
    print(cnmat)    
        
    print ('Mean AUC is:',np.mean(mauc))
    print ('Standard deviation of auc is:', np.std(mauc))
    
    return np.mean(mauc), np.std(mauc), cnmat

def SVM_Linear (Xm, labels,kf):
 
    cnmat = np.zeros((7,7))
    mauc=[]
    labels=labels-1
    for (train, test) in kf.split(Xm): #enumerate (kf):
    
        trGs = Xm[train,:] 
        tsGs =  Xm[test,:] 
        ytr = labels [train]
        yts = labels [test]
      
    
        c_rng = [0.05, 0.1, 0.5, 1, 2, 5, 10]
    
        #g_rng = [0.05, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
    
        
    
        pipe = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                                     #('feat', top_feat),
        ('clf', SVC(kernel='linear', probability=True, class_weight='balanced',
                random_state=1))])
    
       
        param_grid = [ {'clf__C':c_rng}]
         
        gs_i = GridSearchCV(estimator=pipe, 
                            param_grid = param_grid, n_jobs=-1, cv = 2)
        
        
        gs = gs_i.fit(trGs, ytr.ravel())
        print(gs_i.best_params_)
        
        
        SVMLinear = gs.best_estimator_
    
    

        # Confusion matrix calculation
        
        ypred = SVMLinear.predict(tsGs)
    
        cnf_matrix = metrics.confusion_matrix(yts, ypred)
    
        cnmat = cnmat+cnf_matrix
        
        
        # ROC AUC calculation
        
        probas = SVMLinear.predict_proba(tsGs)
        
    
        roc_auc = []
    
        for m in range(7):
            
            fpr0, tpr0, _ = metrics.roc_curve(yts.ravel(), probas[:,m], pos_label=m)
            roc_auc.append (metrics.auc(fpr0,tpr0))
            
        print ('AUC at this fold',np.mean(roc_auc))
        mauc.append(np.mean(roc_auc)) # Mean AUCs over three groups
       
    print(cnmat)    
        
    print ('Mean AUC is:',np.mean(mauc))
    print ('Standard deviation of auc is:', np.std(mauc))
     
    
    return np.mean(mauc), np.std(mauc), cnmat

def Rand_Forest (Xm, labels, kf):    
    cnmat = np.zeros((7,7))
    labels = labels -1 

    mauc=[]    
    for (train, test) in kf.split(Xm): #enumerate (kf):
        
        trfold = Xm[train,:] 
        tsfold =  Xm[test,:] 
        ytr = labels [train]
        yts = labels [test]
      
        
    
        # Random forest 
        forestX = RandomForestClassifier(n_estimators= 5,
                                class_weight='balanced',
                                random_state=1)
    
    
                        
        param_grid = [{'n_estimators': list(range(10, 100,20))}]
    
        gs_i = GridSearchCV(estimator=forestX, param_grid = param_grid, 
                            n_jobs=-1, cv = 2)
        
        
       
        gs = gs_i.fit(trfold, ytr.ravel())
        print(gs.best_params_)
    
    
        forest = gs.best_estimator_
       
        y_pred = forest.predict(tsfold)
        probas = forest.predict_proba(tsfold)
        
        cnf_matrix = metrics.confusion_matrix(yts, y_pred)
        
        cnmat = cnmat+cnf_matrix
        
       # print (cnf_matrix)
        roc_auc = []
    
        for m in range(7):
            
            fpr0, tpr0, _ = metrics.roc_curve(yts.ravel(), probas[:,m], pos_label=m)
            roc_auc.append (metrics.auc(fpr0,tpr0))
            
        print ('AUC at this fold',np.mean(roc_auc))
        mauc.append(np.mean(roc_auc)) # Mean AUCs over three groups
       
    print(cnmat)    
        
    print ('Mean AUC is:',np.mean(mauc))
    print ('Standard deviation of auc is:', np.std(mauc))
    
    return np.mean(mauc), np.std(mauc), cnmat
