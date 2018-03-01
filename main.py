# -*- coding:utf-8 -*-

import argparse
import time
import utils
import usexgboost as uxgbs
reload(uxgbs)


if __name__ == "__main__":
  
  current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
  parser = argparse.ArgumentParser()
  parser.add_argument('-task', default="train", help="train || pred || grid")
  parser.add_argument('-dpath', default="/home/zuning.duan/model/d03_xgboost/data/cnn_data.csv")
  parser.add_argument('-target', default="label")
  parser.add_argument('-ratio', default=0.6)
  parser.add_argument('-key', default='eid')
  parser.add_argument('-estimator', default='gridsearch.pkl')
  parser.add_argument('-pred_model', default='best_estimator.pkl')
  parser.add_argument('-ndim', default=288)
  parser.add_argument('-log', default=current_time+'.log')
  args = parser.parse_args()
  
  log = open(args.log, 'w')
  print >> log, '---- Start Params Initialization ----'
  print >> log, '\n -- task = %s -- ' % args.task
  print >> log, '\n model = %s' % args.pred_model
  
  predictors = ['v_'+str(i) for i in xrange(args.ndim)]
  
  cls = uxgbs.Classify(predictors=predictors, target=args.target, key=args.key, estimator=args.estimator, log=args.log, cur_time=current_time, pred_model='')
  
  print >> log, '\n ---- End Params Initialization ----'
  
  if args.task == 'train':
    print args.dpath
    data = utils.load_data(args.dpath, [args.key, args.target] + predictors)
    log = open(args.log, 'a')
    print >> log, "\n -- Training Model -- \n"
    print >> log, "\n Data Shape: "
    print >> log, data.shape
    log.close()
    
    train_data = data.sample(frac=args.ratio)
    test_index = [x for x in data.index if x not in train_data.index]
    test_data = data.loc[test_index]
    
    cls.training(train_data=train_data, test_data=test_data)
    log = open(args.log, 'a')
    print >> log, "\n -- End Training Model -- "
  elif args.task == 'pred':
    log = open(args.log, 'a')
    print >> log, "\n -- Predict Model --"
    # data = utils.load_data(args.dpath, [args.key, args.target] + predictors)
    data = utils.load_data(args.dpath, [args.key] + predictors)
    # print data.head()
    print >> log, "\n -- Data Shape: --"
    print >> log, data.shape
    log.close()
    cls.predict(data=data, estimator=args.pred_model)
    log = open(args.log, 'a')
    print >> log, "\n -- End Predict Model -- "
    log.close()
  elif args.task == 'grid':
    log = open(args.log, 'a')
    print >> log, '\n -- GridSearch Model --'
    log.close()
    data = utils.load_data(args.dpath, [args.key, args.target] + predictors)
    xdata = data[predictors]
    ydata = data[args.target]
    cls.gridsearch(xdata=xdata, ydata=ydata)
    log = open(args.log, 'a')
    print >> log, "\n -- End GridSearch Model -- "
    log.close()
  else:
    print "Error Message: Task: %s is wrong !!!"
    raise SystemExit(101)
    
  log = open(args.log, 'a')
  print >> log, '\n ======== Over ========'
  log.close()
