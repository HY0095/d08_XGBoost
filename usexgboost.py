# -*- coding:utf-8 -*-

import os
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
import pickle
import opts


class Classify(object):
  
  def __init__(self, predictors, target, key, estimator, log, pred_model, cur_time):
    
    self.predictors = predictors
    self.target = target
    self.log = log
    self.key = key
    self.current_time = str(cur_time)
    
    if os.path.isfile(estimator):
      self.xgb = pickle.load(open(estimator, 'r'))
    else:
      self.xgb = XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=20, scale_pos_weight=1, seed=123)
      log = open(self.log, 'a')
      print >> log, "\n ---- Load Estimator ----"
      print >> log, self.xgb
      log.close()
      
      if os.path.isfile(pred_model):
        fr = open(pred_model, "rb")
        self.pred_model = pickle.load(fr)
        fr.close()
      else:
        self.pred_model = ''
      log = open(self.log, 'a')
      print >> log, "\n ----Load Best Model ----"
      print >> log, self.pred_model
      log.close()
      
    def training(self, train_data, test_data, usecv=True, cv_fold=5, early_stop_rounds=50, save=True):
      if usecv:
        log = open(self.log, 'a')
        print >> log, "\n ** Use CV Model **"
        xgb_param = self.xgb.get_xgb_params()
        xgbtrain = xgb.DMatrix(train_data[self.predictors].values, label=train_data[self.target].values)
        cvresult = xgb.cv(xgb_param, xgbtrain, num_boost_round=self.xgb.get_xgb_params()['n_estimators'], nfold=cv_fold, metrics='auc', early_stopping_rounds=early_stop_rounds)
        self.xgb.set_params(n_estimators=cvresult.shape[0])
        print >> log, self.xgb
        print >> log, "\n Finish CV Model"
        log.close()
        
      # fit the algorithm on the data
      self.xgb.fit(train_data[self.predictors].as_matrix(), train_data[self.target], eval_metric='auc')
      
      # Predict training set
      opts.res_print(train_data[self.predictors].as_matrix(), train_data[self.target], self.xgb, self.log, flag="Train")
      
      # Predict test set
      opts.res_print(test_data[self.predictors].as_matix(), test_data[self.target], self.xgb, self.log, flag="Test")
      
      # Save model pickle
      if save:
        f = open("best_model_" + self.current_time + ".pickle", "w")
        pickle.dump(self.xgb, f)
        f.close()
        
      # Get import feature
      feat_import = pd.Series(self.xgb.booster().get_fscore()).sort_values(ascending=False)
      log = open(self.log, 'a')
      print >> log, "Top(50) Feature Importance Score"
      print >> log, feat_import[:50]
      log.close()
      
    def predict(self, data, estimator=''):
      
      print estimator
      if estimator == '':
        estimator = self.pred_model
      else:
        if os.path.isfile(estimator):
          estimator = pickle.load(open(estimator, 'r'))
        else:
          print "Error Message: estimator is Empty !!!"
          raise SystemExit(102)
        
      log = open(self.log, 'a')
      print >> log, estimator
      log.close()
      result = pd.DataFrame()
      # result['label'] = data[self.traget].values
      result['prediction'] = estimator.predict(data[self.predictors].as_matrix())
      result['predprob'] = estimator.predict_prob(data[self.predictors].as_matrix())[:, 1]
      result['eid'] = data.index.values
      result = result.set_index('eid')
      result.to_csv('predict_result.csv')

  def gridsearch(self, xdata, ydata):
    
    params_dict = {
      "learning_rate": [0.01 * i for i in xrange(1, 10)],
      "min_child_weight": range(1, 20, 1),
      "max_depth": range(3, 12),
      "gamma": [i/10.0 for i in xrange(0, 5)],
      "subsample": [i/100.0 for i in xrange(60, 95, 5)],
      "colsample_bytree": [i/100.0 for i in xrange(60, 95, 5)],
      "reg_lambda": [1e-3, 1e-2, 1e-1, 0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
      "reg_alpha": [1e-3, 1e-2, 1e-1, 0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    }
    
    # initialization parameters
    params = {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 4, 'min_child_weight': 2, 'gamma': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 123, 'objective': "binary:logistic", 'nthread': 20,
              'reg_alpha': 0, 'reg_lambda': 1}
    # Grid Search Order
    params_search_list = ['max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'learning_rate']
    
    # Start Grid Search
    log = open(self.log, 'a')
    for param in params_search_list:
      param_search = {}
      if param in params.keys():
        param_search[param] = params_dict[param]
      else:
        print >> log, "\n Error Message: %s not in param.dict !!!" % param
        raise SystemExit
      if param in params.keys():
        del params[param]
        
      print >> log, "*** Start Grid Search: %s " % param
      gsearch = GridSearchCV(estimator=XGBClassifier(**params), param_grid=param_search, scoring='roc_auc', iid=False, cv=5)
      gsearch.fit(xdata, ydata)
      params[param] = gsearch.best_params_[param]
      for i in xrange(len(gsearch.grid_scores_)):
        print >> log, gsearch.grid_scores_[i]
        
      print >> log, gsearch.best_estimator_, gsearch.best_score_
      print >> log, "** End Grid Search %s \n" % param
    log.close()
    
    # Save Best Estimator
    pickle.dump(gsearch.best_estimator_, open('gridsearch.pkl', 'w'))
    
    # return gsearch.best_estimator_
