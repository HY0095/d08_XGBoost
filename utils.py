# -*- coding:utf-8 -*-

import pandas as pd
import os


def load_data(dpath, col_names):
  if os.path.isfile(dpath):
    data = pd.read_csv(dpath, index_col=0, names=col_names)
    data.fillna(0, inplace=True)
  else:
    print "ErrorMessage: %s is not exists !!"
    raise SystemExit(103)
  return data

