
# data0 Balluff
# data1 ADXL
dataset = '0'
# dataset = '1'


# folder path
path = '1'

# trial run
trial_group = '1'


#Raw data
input_file = 'data'+str(dataset)+'/1.csv'


#Features Extracted to file
features_file = 'data'+str(dataset)+'/'+str(path)+'/'+str(trial_group)+'_features.csv'


#Train Test spilt
train_file = 'data'+str(dataset)+'/'+str(path)+'/train_'+str(trial_group)+'.csv'
test_file = 'data'+str(dataset)+'/'+str(path)+'/test_'+str(trial_group)+'.csv'

#number of train data
validation_split_i = 400

# the bigger, the more features selected
# default is 0.05
fdr_level = 0.05

