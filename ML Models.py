# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:36:58 2021

@author: USER
"""
import time
import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.svm import SVC 
from sklearn import svm, datasets

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler as ss, MinMaxScaler as mm, RobustScaler as rs
from sklearn.preprocessing import MaxAbsScaler as ma, Normalizer as nz, QuantileTransformer as qt, PowerTransformer as pt
#add robust scaling
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from playsound import playsound

from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
import joblib

from config import path, trial_group, train_file, test_file,dataset

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )


# train = pd.read_csv('data/'+str(path)+'/train_1.csv')
# test = pd.read_csv('data/'+str(path)+'/test_1.csv')

x_train = train.drop( 'y', axis = 1 ).values
y_train = train.y.values

x_test = test.drop( 'y', axis = 1 ).values
y_test = test.y.values

x_train_ss = ss().fit_transform(x_train)
x_train_mm = mm().fit_transform(x_train)
x_train_rs = rs().fit_transform(x_train)
x_train_ma = ma().fit_transform(x_train)
x_train_nz = nz().fit_transform(x_train)
x_train_qt = qt().fit_transform(x_train)
x_train_pt = pt().fit_transform(x_train)

x_test_ss = ss().fit_transform(x_test)
x_test_mm = mm().fit_transform(x_test)
x_test_rs = rs().fit_transform(x_test)
x_test_ma = ma().fit_transform(x_test)
x_test_nz = nz().fit_transform(x_test)
x_test_qt = qt().fit_transform(x_test)
x_test_pt = pt().fit_transform(x_test)

no_sc_flag = True
ss_flag = True
mm_flag = True
rs_flag = False
ma_flag = False
nz_flag = True
qt_flag = False
pt_flag = False


model_params = {
    # 'svm': {
    #     'model': svm.SVC(gamma='auto'),
    #     'params' : {
    #         'C': [0.01,0.1,1,10,100],
    #         'kernel': ['rbf','linear','poly','sigmoid'],
    #         'degree': [2,3]
            
            
    #     }  
    # },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [10,20,30,40,50,60,70,80,90,100],
            'criterion': ['gini','entropy'],
            'max_features': [10,20,30,40,50,60,70,80,90,100],
            
            
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(multi_class='auto'),
        'params': {
            'C': [1,10,100],
            'solver': ['newton-cg','sag','saga','lbfgs']
            
            
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {
            
        }
        
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            'max_features': ['auto', 'sqrt','log2']
            
            
        }
    },

    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params' : {
            'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
            'weights': ['uniform','distance'],
            
        
        }
            
        
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=150),
        'params': {
            'hidden_layer_sizes': [(50,50,50),(50,100,50),(100,300,100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive']
        
        }
            
        
    }
          
}
print("******************************************************")
if no_sc_flag==True :
    
    # #Non Scaled X
    scores_no_sc = []
    
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_no_sc.pkl')
        
        
        best_clf = clf.best_estimator_
    
        test_predict = best_clf.predict(x_test)
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_no_sc.append({
            'model': model_name,
            'scaling' : 'Non scaled',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
    
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_no_sc_results.csv',header = True, index = None)
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_no_sc_ca_report.csv')
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))
    
        
        
        print("******************************************************")
        
    df_no_sc = pd.DataFrame(scores_no_sc,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])
   
    
if ss_flag==True :

    #Standard Scaled X - ss
    scores_ss = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_ss, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_ss.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_ss)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_ss.append({
            'model': model_name,
            'scaling' : 'Standard scaled',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_ss_results.csv',header = True, index = None)
        
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_ss_ca_report.csv')
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))

        
        
        
        print("******************************************************")
        
    df_ss = pd.DataFrame(scores_ss,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])


if mm_flag==True :
    #MinMax Scaled X - mm
    scores_mm = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_ss, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_mm.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_mm)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_mm.append({
            'model': model_name,
            'scaling' : 'MinMax scaled',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_mm_results.csv',header = True, index = None)
        
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_mm_ca_report.csv')
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))

        
        
        
        print("******************************************************")
        
    df_mm = pd.DataFrame(scores_mm,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])

if rs_flag==True :
    #Robust Scaled X - rs
    scores_rs = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_rs, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_rs.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_rs)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_rs.append({
            'model': model_name,
            'scaling' : 'Robust scaled',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_rs_results.csv',header = True, index = None)
        
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_rs_ca_report.csv')
        
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))

        
        
        
        print("******************************************************")
        
    df_rs = pd.DataFrame(scores_rs,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])


if ma_flag==True :
    #MaxAbs Scaled X - ma
    scores_ma = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_rs, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_ma.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_rs)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_ma.append({
            'model': model_name,
            'scaling' : 'MaxAbs scaled',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_ma_results.csv',header = True, index = None)
        
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_mm_ca_report.csv')
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))
  
        
        
        
        print("******************************************************")
        
    df_ma = pd.DataFrame(scores_ma,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])


if nz_flag==True :
    #Normalizer  X - nz
    scores_nz = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_nz, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_nz.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_nz)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_nz.append({
            'model': model_name,
            'scaling' : 'Normalized ',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_nz_results.csv',header = True, index = None)
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_nz_ca_report.csv')
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))

        
        
        
        print("******************************************************")
        
    df_nz = pd.DataFrame(scores_nz,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])


if qt_flag==True :
    #QuantileTransformer  X - qt
    scores_qt = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_qt, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_qt.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_qt)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_qt.append({
            'model': model_name,
            'scaling' : 'Quantile',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_qt_results.csv',header = True, index = None)
        
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_qt_ca_report.csv')
        
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))

        
        
        
        print("******************************************************")
        
    df_qt = pd.DataFrame(scores_qt,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])


if pt_flag==True :
    #PowerTransformer  X - pt
    scores_pt = []
    for model_name, mp in model_params.items():
        
        start=time.time()
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=True,refit=True)
        clf.fit(x_train_pt, y_train)
        
        joblib.dump(clf.best_estimator_, 'data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_pt.pkl')
    
    
        best_clf = clf.best_estimator_
        test_predict = best_clf.predict(x_test_pt)
    
        
        print(mp['model'])
        print('')
        
        skplt.metrics.plot_confusion_matrix(y_test,test_predict) 
        ts_acc = round(accuracy_score(y_test,test_predict), 3)
        
        stop = time.time()
        gs_time = stop - start
        
        scores_pt.append({
            'model': model_name,
            'scaling' : 'Power',
            'val_acc': clf.best_score_,
            'best_params': clf.best_params_,
            'total GS time' : gs_time,
            'time': clf.refit_time_,
            'test_acc': ts_acc
            })
        
        
        r=pd.DataFrame(clf.cv_results_)
        r.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_pt_results.csv',header = True, index = None)
        
        
        df_CAreport=classification_report(y_test,test_predict,output_dict=True)
        df_ca_report = pd.DataFrame(df_CAreport).transpose()
        
        df_ca_report.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(model_name)+'_'+str(trial_group)+'_pt_ca_report.csv')
        
        print(f"Gridsearch total time: {stop - start}s")
        print("Test Accuracy: "+str(ts_acc))

        
        
        print("******************************************************")
        
    df_pt = pd.DataFrame(scores_pt,columns=['model','scaling','val_acc','best_params','total GS time','time','test_acc'])

df_all = pd.concat([df_no_sc,
                    df_ss,
                    df_mm,
                    # df_ma,
                    df_nz,
                    # df_pt,
                    # df_qt,
                    # df_rs
                    ], axis=0, ignore_index=True)

df_all.to_csv('data'+str(dataset)+'/'+str(path)+'/'+str(trial_group)+'_all_results.csv',header = True, index = None)

print("*******************Done Compute***********************")
playsound('alarm.mp3')