# -*- coding:utf-8 -*-

import pickle
import argparse
from pyspark import SparkContext
from pyspark import HiveContext
import pandas as pd
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

sc = SparkContext.getOrCreate()
hc = HiveContext(sc)


# def xgbt_pred(estimator, List, ndim):
#     df = pd.DataFrame(List)
#     df = df.transpose()
#     # columns = ['f'+str(i) for i in xrange(ndim)]
#     # df.columns = columns
#     x = df.as_matrix()
#     # return x
#     result = estimator.predict(x)
#     return result
#


def xgbt_pred(estimator, List, ndim):

    if len(List) == 0:
        List = list(np.zeros(ndim))
    df = pd.DataFrame(List)
    df = df.transpose()
    columns = ['f'+str(i) for i in xrange(ndim)]
    df.columns = columns
    predprob = estimator.predict_proba(df)[:, 1]
    return str(predprob[0])


def save2hive_table():

    drop_table = '''drop table if exists creditmodel.xgbt_pred_score'''

    create_table = '''
    create table creditmodel.xgbt_pred_score (
        eid string
       ,xgbt_score string
    )
    '''
    insert_table = '''
    insert overwrite table creditmodel.xgbt_pred_score
        select
            eid
           ,xgbt_score
        from xgbt_pred_score_tmp
    '''
    return drop_table, create_table, insert_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default='pred', help='')
    parser.add_argument('-table', default='eid_cnn_feature')
    parser.add_argument('-pred_model', default='best_model_20180124_155303.pkl')
    args = parser.parse_args()

    estimator = pickle.load(open(args.pred_model, 'r'))

    sql = '''
    select 
        eid
      ,eid_value
    from creditmodel.%s
    limit 1000
    ''' % args.table

    col_name = ['eid', 'score']

    predRdd = hc.sql(sql).rdd.filter(lambda x: len(x[1]) > 0).cache()
    predRdd2 = hc.sql(sql).rdd.cache()
    predRdd2.map(lambda x: [x[0], xgbt_pred(estimator, x[1], 288)]).toDF(col_name).registerTempTable('xgbt_pred_score_tmp')

    drop_table, create_table, insert_table = save2hive_table()

    hc.sql(drop_table)
    hc.sql(create_table)
    hc.sql(insert_table)

    hc.sql("select * from creditmodel.xgbt_pred_score limit 10").show()
