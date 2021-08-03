dataset = '1'


# path = 'base'
# path = 'faulty_x+1'
# path = 'faulty_y+2'
path = '1'
# gyro = 'x'
# gyro = 'y'
# gyro = 'z'
gyro = '1'

classes='1'

trial_group = '2'

#Raw data
input_file = 'data1/1.csv'

# input_file = 'data1/1/'+str(path)+"/"+str(gyro)+'_class.csv'

#Features Extracted to file
# features_file = 'data/P&P/'+str(path)+"/"+str(gyro)+'_features.csv'

# features_file = 'data1/1/'+str(classes)+'_features_randomized.csv'
features_file = 'data1/1/'+str(classes)+'_features.csv'
#Train Test spilt
train_file = 'data1/1/train_'+str(trial_group)+'.csv'
test_file = 'data1/1/test_'+str(trial_group)+'.csv'

#number of train data
validation_split_i = 400

# the bigger, the more features selected
# default is 0.05
fdr_level = 0.05

# path = 1
# trial 1 = 0.005
# trial 2 = 0.05
# trial 3 = 0.1
# trial 4 = 0.5
# trial 5 = 1