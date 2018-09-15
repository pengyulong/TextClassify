from sklearn.externals import joblib
import os
import numpy as np
import pandas as pd
from Parameter import ProjectPath
from utils import logging,train_classify,write_data,save_prob_file

paths = ProjectPath()
def concat_feature(modes_list,cols_list,dims_list):
    train_feature,test_feature = [],[]
    for mode in modes_list:
        for i,col in enumerate(cols_list):
            train_feature_file = os.path.join(paths.data_dir,'{}_{}_{}d.dat'.format(mode,col,dims_list[i]))
            if os.path.exists(train_feature_file):
                trainX,testX = joblib.load(train_feature_file)
                train_feature.append(trainX)
                test_feature.append(testX)
            else:
                raise FileNotFoundError("feature data not found,please run text_ml_classify.py")
                return False
    trainX,testX = np.hstack(train_feature),np.hstack(test_feature)
    concat_data_file = os.path.join(paths.data_dir,"concat_feature.dat")
    joblib.dump((trainX,testX),concat_data_file)
    logging.info("trainX's shape:{}".format(trainX.shape))
    logging.info("testX's shape:{}".format(testX.shape))
    return (trainX,testX)

if __name__ == "__main__":
    modes_list = ['lsi','lda']
    cols_list = ['word_seg','column']
    dims_list = [200,100]
    classify_mode = 'lightgbm'
    trainY = (pd.read_csv(os.path.join(paths.data_dir,"train_set.csv"))['class']-1).tolist()
    try:
        (trainX,testX)=concat_feature(modes_list,cols_list,dims_list)
    except Exception as err:
        print(err)
        os._exit(-1)
    ndims = trainX.shape[1]
    model,f1 = train_classify(trainX,trainY,classify_mode)
    y_pred = model.predict (testX)
    y_prob = model.predict_proba (testX)
    result_string = ['id,class\n']
    for i,pred in enumerate(y_pred):
        result_string.append('{},{}\n'.format(i,pred))
    result_file = os.path.join(paths.data_dir,"concat_feature_{}_{}d.csv".format(classify_mode,ndims))
    result_prob_file = os.path.join(paths.data_dir,"concat_feature_{}_{}d_prob.csv".format(classify_mode,ndims))
    write_data(result_string,result_string)
    save_prob_file(y_prob,result_prob_file)