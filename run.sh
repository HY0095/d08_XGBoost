#!/bin/bash

#***********************************************
# 1. Offline Model
#***********************************************

# ---- File Path ----
pydir='/home/zuning.duan/model/d03_xgboost/src'
# data='/home/zuning.duan/model/d03_xgboost/data'
data='/home/zuning.duan/model/d03_xgboost/data'
sparksubmit='/usr/install/spark2-yarn/bin/spark-submit --conf spark.yarn.executor.memoryOverhead=12g --conf spark.executor.memory=20g --conf spark.default.parallelism=3000 --conf spark.executor.cores=3 --driver-memory 6g --conf spark.executor.exetraJavaOptions="-XX:MaxPerSize=1024m" --total-executor-cores 60'


# ---- Grid Search ----
#python ${pydir}/main.py -task grid -pred_model aaa -dpath ${data}/cnn_data_12x24.csv

# ---- Train Model ----
#python ${pydir}/main.py -task train - estimator gridsearch.pkl -dpath ${data}/cnn_data_12x24.csv

# ---- Predict Model ----
python ${pydir}/main.py -task pred -pred_model best_model.pkl -dpath ${data}/cnn_data.csv


#***********************************************
# Online Model
#***********************************************

# Final Table : createmodel.xgbt_pred_score
#${sparksubmit} ${pydir}/online_predict.py -- -table eid_cnn_feature -pred_model best_model.pkl
