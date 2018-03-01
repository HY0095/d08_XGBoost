# -*- coding:utf-8 -*-

import numpy as np
from sklearn import metrics
import pandas as pd


def res_print(xdata, ydata, estimator, log, flag):
  log = open(log, 'a')
  ypred = estimator.predict(xdata)
  yscore = estimator.predict_proba(xdata)[:, 1]
  print >> log, "Model Report"
  print >> log, "Accuracy (%s): %.4g" % (flag, metrics.accuracy_score(ydata, ypred))
  print >> log, "AUC Score(%s): %.4g" % (flag, metrics.roc_auc_score(ydata, yscore))
  print >> log, "K-S Score (%s): %.4g" % (flag, ks(ydata, yscore))
  log.close()
  
  
def ks(y_true, y_pred):
  data = pd.DataFrame()
  data['ytrue'] = y_true
  data['ypred'] = y_pred
  ks_data = data.sort_values(by='ypred', ascending=False)
  good = ks_data['ytrue'] == 1
  bad = ks_data['ytrue'] == 0
  good_cf = np.cumsum(good / good.sum())
  bad_cf = np.cumsum(bad / bad.sum())
  good_bad_cum = list(np.abs(good_cf - bad_cf))
  return max(good_bad_cum)

