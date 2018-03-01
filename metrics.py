# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

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

