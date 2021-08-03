# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:31:32 2021

@author: USER
"""
from config import input_file, features_file
import numpy as np
import pandas as pd
import tsfresh

#Read test set
df = pd.read_csv(input_file, header = 1 )

df.columns=['x','y','z','ElapsedTime','LoopTime']

df = df.drop(['LoopTime'], axis = 1)

# Manually select time series data classes
df_base = df.iloc[1500:19500]
df_xp1 = df.iloc[28000:45000]
df_xn1 = df.iloc[55000:73000]
df_yp1 = df.iloc[81000:99500]
df_yn1 = df.iloc[108500:126300]

# Append class
df_base.loc[:,'target']=1
df_xp1.loc[:,'target']=2
df_xn1.loc[:,'target']=3
df_yp1.loc[:,'target']=4
df_yn1.loc[:,'target']=5

df_all = pd.concat([df_base,df_xp1,df_xn1,df_yp1,df_yn1], axis=0, ignore_index=True)

# Append id based on sub_length (Number of instance per one cycle)
from tsfresh.utilities.dataframe_functions import add_sub_time_series_index
df_index = add_sub_time_series_index(df_all,sub_length = 180)
df_final = df_index[['ElapsedTime','x','y','z','id','target']]

# Remove duplicate id for y 
y1 = df_final.id
y2 = df_final.target
y12 = pd.concat([y1,y2],axis=1)
y_drop = y12.drop_duplicates()
y_reset = y_drop.reset_index()
y = y_reset.drop(['index','id'], axis = 1)

# Remove Target for feature extraction
df_final.drop( 'target', axis = 1, inplace = True )

# Extract Features
from tsfresh import extract_features
df_features = extract_features(df_final, column_id="id",column_sort="ElapsedTime")

from tsfresh.utilities.dataframe_functions import impute
impute( df_features )
assert df_features.isnull().sum().sum() == 0

# Append class back to extracted features 
df_features['y'] = y

# Save feature file
df_features.to_csv(features_file, header = True , index = None)