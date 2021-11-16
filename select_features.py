#!/usr/bin/env python

"select features, save train and test files"

import pandas as pd
from tsfresh import extract_features, select_features

from config import input_file, features_file
from config import validation_split_i, fdr_level, train_file, test_file
import numpy as np
from sklearn.model_selection import train_test_split
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test as ftest

# workaround for multiprocessing on windows
if __name__ == '__main__':
    print ("loading {}".format( features_file ))
    
    features = pd.read_csv( features_file )
    
    # train_x = features.iloc[:validation_split_i].drop( 'y', axis = 1 )
    # test_x = features.iloc[validation_split_i:].drop( 'y', axis = 1 )

    # train_y = features.iloc[:validation_split_i].y
    # test_y = features.iloc[validation_split_i:].y
    
    train_x, test_x, train_y, test_y = train_test_split(
            features.drop( 'y', axis = 1 ), features.y, 
            test_size=0.2, random_state=42, 
        )

    print ("selecting features...")
    train_features_selected = select_features( train_x, train_y, fdr_level = fdr_level, )

    print ("selected {} features.".format( len( train_features_selected.columns )))

    train = train_features_selected.copy()
    train['y'] = train_y

    test = test_x[ train_features_selected.columns ].copy()
    test['y'] = test_y
    
    #

    # print ("saving {}".format( train_file ))
    # train.to_csv( train_file, index = None )

    # print ("saving {}".format( test_file ))
    # test.to_csv( test_file, index = None )
    
    # Relevance Table
    
    RTable = ftest(features.drop( 'y', axis = 1 ), features.y)