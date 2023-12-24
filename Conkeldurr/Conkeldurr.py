#coding=utf-8

from pickle import FALSE
import tushare as ts
import pandas as pd
import sys
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt

from datetime import datetime,timedelta

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import median_absolute_error,mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

def downloadall(pro):

    #df_calender = pro.trade_cal(exchange='SHFE', start_date='20230801', end_date='20231031')

    #print(df_calender)
    df=pd.read_pickle('Daily_opt_SZSE.pkl')
    
    print(df)

    df.to_csv('Daily_opt_SZSE.csv',encoding='utf-8-sig')


    sdafkasfd=1

def remove_first_two_chars(x):
    return str(x)[2:]  # 删除前两个字符

def get_df_opt_basics():
    
    # df = pro.opt_basic(exchange='DCE', fields='ts_code,name,exercise_type,list_date,delist_date')
    # df.to_csv("opt_basic1.csv",encoding='utf-8-sig')

    #这个接口似乎还能用，但是理论上来说是要5k积分的，所以先能不用就不用
    #但是这个必须要用因为后面的数据的id是根据这个列表来的，实在不行这里还是要付费
    #然后似乎还能通过调试之间获取，所以先不付费

    # 上证：SSE
    # 50ETF：华夏上证50ETF期权 510050.SH
    # 300ETF：华泰柏瑞沪深300ETF 510300.SH
    # 500ETF：南方中证500ETF 510500.SH
    # 科创版：华夏上证科创板50ETF 588000.SH

    # 深圳：SZSE
    # 创业板：易方达创业板ETF 159915.SZ

    # 1.用中文搜索找到相应的一个合约/所有合约？

    #df_opt_basic_sse = pro.opt_daily(trade_date='20231026',exchange='SSE')
    df_opt_basic_sse=pd.read_csv('seee0.csv',index_col=0,header=0,encoding='utf-8-sig')

    sse50etf = df_opt_basic_sse[df_opt_basic_sse['name'].str.contains("华夏上证50ETF期权")]
    sse300etf = df_opt_basic_sse[df_opt_basic_sse['name'].str.contains("华泰柏瑞沪深300ETF")]
    sse500etf = df_opt_basic_sse[df_opt_basic_sse['name'].str.contains("南方中证500ETF")]
    sseKCetf = df_opt_basic_sse[df_opt_basic_sse['name'].str.contains("华夏上证科创板50ETF")]

    df_opt_basic_szse=pd.read_csv('seee1.csv',index_col=0,header=0,encoding='utf-8-sig')

    szseCYBetf = df_opt_basic_szse[df_opt_basic_szse['name'].str.contains("易方达创业板ETF")]

    print(sse50etf)
    print(sse300etf)
    print(sse500etf)
    print(sseKCetf)
    print(szseCYBetf)

    # 2.找到相应合约的日线

    ##次月
    current_date=datetime.now()
    current_date=current_date+timedelta(days=30)

    year= str(current_date.year)[2:]
    month = str(current_date.month).zfill(2)
    result=year + month

    print(result)

    sse50etf_nextmonth = sse50etf[sse50etf['name'].str.contains(result)]

    ## 次月对应的所有合约
    print(sse50etf_nextmonth)
    #找到当天日期格式

    today_form=datetime.today().strftime("%Y%m%d")
    print(today_form)
    option_today_date=int(today_form)

    option_end_date=sse50etf_nextmonth['delist_date'].values[0]

    ## 找到其中某个合约的日线

    ## 判断合约剩余的交易日

    #df_calender = pro.trade_cal(exchange='SHFE', start_date='20230101', end_date='20231231')
    #df_calender.to_csv("df_calender.csv")
    df_calender=pd.read_csv('df_calender.csv',index_col=0,header=0,encoding='utf-8-sig')

    df_calender=df_calender[df_calender['cal_date']>=option_today_date]
    df_calender=df_calender[df_calender['cal_date']<=option_end_date]

    last_dates=df_calender['is_open'].sum()
    last_dates_all=df_calender['is_open'].count()

    print(last_dates)
    print(last_dates_all)


def get_ETF_info(pro):
    
    # 3.找到相应的ETF价格（日线）
    
    #df = pro.fund_basic(market='E')
    #df.to_csv("seee3.csv",encoding='utf-8-sig')

    df50etf = pro.fund_daily(ts_code='510050.SH', start_date='20200101', end_date='20231111')
    df300etf = pro.fund_daily(ts_code='510300.SH', start_date='20200101', end_date='20231111')
    df500etf = pro.fund_daily(ts_code='510500.SH', start_date='20200101', end_date='20231111')
    dfKCetf = pro.fund_daily(ts_code='588000.SH', start_date='20200101', end_date='20231111')
    dfCYBetf = pro.fund_daily(ts_code='159915.SZ', start_date='20200101', end_date='20231111')

    print(df50etf)
    print(df300etf)
    print(df500etf)
    print(dfKCetf)
    print(dfCYBetf)
    
    df50etf.to_csv('df50etf.csv',encoding='utf-8-sig')
    df300etf.to_csv('df300etf.csv',encoding='utf-8-sig')
    df500etf.to_csv('df500etf.csv',encoding='utf-8-sig')
    dfKCetf.to_csv('dfKCetf.csv',encoding='utf-8-sig')
    dfCYBetf.to_csv('dfCYBetf.csv',encoding='utf-8-sig')


def akshare_test():
    
    #这里可以直接用akshare接口获取行权价
    option_finance_board_df = ak.option_finance_board(symbol="华夏科创50ETF期权", end_month="2212")
    print(option_finance_board_df)

    option_cffex_zz1000_spot_sina_df = ak.option_cffex_zz1000_spot_sina(symbol="mo2208")
    print(option_cffex_zz1000_spot_sina_df)
    
    option_sse_spot_price_sina_df = ak.option_sse_spot_price_sina(symbol="10005533")
    print(option_sse_spot_price_sina_df)

def regression_metrics(true,pred):
    print('回归模型评估指标结果:')
    print('均方误差【MSE】:', mean_squared_error(true, pred))
    print('均方根误差【RMSE】:',np.sqrt(mean_squared_error(true,pred)))
    print('平均绝对误差【MAE】:',mean_absolute_error(true,pred))
    print('绝对误差中位数【MedianAE】:',median_absolute_error(true,pred))
    print('平均绝对百分比误差【MAPE】:',mean_absolute_percentage_error(true,pred))
    #print('绝对百分比误差中位数【MedianAPE】:',median_absolute_percentage_error(true,pred))

def from_chatgpt():

    # 读入数据
    df = pd.read_csv('opt_see1129.csv', encoding='gbk')
    #X_var = joblib.load('../data/X_var.pkl')  # 特征列表

    df.dropna(inplace=True)

    # 划分验证集
    valid = df.sample(frac=0.2, random_state=42)
    df.drop(index=valid.index, axis=1, inplace=True)

    print(df)

    mapping_call_put = {'C': 1, 'P': 2}
    
    mapping_opt_code = {'510050.SH': 1, '510300.SH': 2, '510500.SH': 3, '588000.SH': 4, '588080.SH': 5}

    df['call_put'] = df['call_put'].map(mapping_call_put)
    df['opt_code'] = df['opt_code'].map(mapping_opt_code)

    print(df)

    # 划分训练集和测试集
    
    X = df
    y = df['opt_fake_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.drop(['ts_code','opt_fake_value','close'],axis=1,inplace=False)
    X_test_ori = X_test
    X_test = X_test.drop(['ts_code','opt_fake_value','close'],axis=1,inplace=False)

    # 建立基础模型
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'max_depth': 7,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 10
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    m1 = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_train, lgb_eval])

    # 预测数据集
    y_pred = m1.predict(X_test)

    print(y_pred)
    print(X_test_ori)

    outputy=pd.DataFrame(y_pred,columns=['y_pred'])
    
    X_test_ori.reset_index(inplace=True)

    X_test_ori['y_pred']=outputy    

    print(X_test_ori)

    X_test_ori.to_csv('seey2.csv')

    # 评估模型
    print('回归模型评估指标结果:')
    regression_metrics(y_test, y_pred)
    xxxx=1

def train_option_exercise_probability_model():

    df=pd.read_csv('try1224.csv',index_col=0,header=0,encoding='utf-8-sig')

    df['ETF_close'].fillna(0, inplace=True)
    df['close'].fillna(0, inplace=True)
    df['pre_close'].fillna(0, inplace=True)
    df['pre_close'].fillna(0, inplace=True)

    # # 划分验证集
    # valid = df.sample(frac=0.2, random_state=42)
    # df.drop(index=valid.index, axis=1, inplace=True)

    print(df)

    mapping_call_put = {'C': 1, 'P': 2}
    
    mapping_opt_code = {'510050.SH': 1, '510300.SH': 2, '510500.SH': 3, '588000.SH': 4, '588080.SH': 5}

    df['call_put'] = df['call_put'].map(mapping_call_put)
    df['opt_code'] = df['opt_code'].map(mapping_opt_code)

    print(df)
    dfindex=df
    df = df.drop(['ts_code'],axis=1,inplace=False)

    #df=df[['pre_close','opt_code','call_put','days_remain','Exercise_Status']]


    # 划分训练集和测试集
    
    X = df[['pre_close','close','ETF_close','real_value','opt_code','call_put','days_remain']]
    y = df['Exercise_Status']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_ids = X.index.tolist()
    splitno=int(len(train_ids)*0.30)
    
    X_train = X.iloc[:splitno, :]
    X_test=X.iloc[(splitno+100):, :]
    
    y_train=y[:splitno]
    y_test=y[(splitno+100):]

    dfindex_train=dfindex.iloc[:splitno, :]
    dfindex_test=dfindex.iloc[(splitno+100):, :]
    
    print(X_train)
    print(dfindex_test)

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 设置参数
    params = {
        'objective': 'binary',  # 二分类任务
        'metric': 'binary_logloss',  # 使用对数损失作为评估指标
        'boosting_type': 'gbdt',  # 使用梯度提升树
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习率
        'feature_fraction': 0.9,  # 特征采样比例
        'bagging_fraction': 0.8,  # 数据采样比例
        'bagging_freq': 5,  # 数据采样频率
        'verbose': 0  # 控制输出信息
    }

    # 训练模型
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=100)

    # 预测
    predictions = model.predict(X_test)

    outputy=pd.DataFrame(predictions,columns=['y_pred'])

    dfindex_test.reset_index(inplace=True,drop=True)
    
    print(dfindex_test)

    dfindex_test['Exercise_pred']=outputy

    print(dfindex_test)
    
    dfindex_test.to_csv('see1224.csv')

    zzzz=1

def opt_backTesting():

    daily_all_df=pd.read_csv('see1224.csv',header=0,index_col=0,encoding='utf-8-sig')

    print(daily_all_df)

    datelist=daily_all_df['trade_date'].unique()

    show3=[]
    mean=0
    std_dev = 1
    
    baseline50=[]
    baseline=1
    baselinerd=1
    
    hold_list=[]
    hold_list=pd.DataFrame(columns=('ts_code','call_put','buy_amount','lastprice'))
    
    account=10000000
    buy_pct=0.5
    accountbase=account
    lastallvalue=account
    #当日结束总资产
    allvalue=account

    buyorsell=0

    seed_value = 66
    np.random.seed(seed_value)

    # skip1=0
    for cur_date in datelist:
        # skip1+=1;
        # if(skip1<2):
        #     continue;

        #获取当日的数据
        cur_df_all=daily_all_df[daily_all_df['trade_date'].isin([cur_date])]
        cur_df_all=cur_df_all.sort_values(by='exercise_price', ascending=True)

        #这里改成代号
        # mapping_opt_code = {'510050.SH': 1, '510300.SH': 2, '510500.SH': 3, '588000.SH': 4, '588080.SH': 5}
        filtered_df = cur_df_all.loc[cur_df_all['opt_code'] == 2]
        if(len(filtered_df)==0):
            curpctchg=0
        else:
            curpctchg=filtered_df['ETF_pct_chg'].values[0]
        
        baseline=baseline*(1+curpctchg/100)
        baseline50.append(baseline)


        # 计算新的价值，卖出hold的期权，计算资产
        if hold_list.shape[0]>0 :
            hold_list_buffer=pd.merge(hold_list,filtered_df, how='left', on=['ts_code'])
            hold_list_buffer.reset_index(inplace=True,drop=True)
            #print(hold_list_buffer)

            hold_list.loc[:,'lastprice']=hold_list_buffer['close']
            
            #print(hold_list)
            dsfsdf=1

        #卖出逻辑
        if hold_list.shape[0]>0 :
            #简单的直接清空
            hold_list_value=hold_list['buy_amount']*hold_list['lastprice']*10000
            #前面买入这里加
            if buyorsell == 0 :          
                account=account+hold_list_value.sum()
            else:
                account=account-hold_list_value.sum()
            hold_list=pd.DataFrame(columns=('ts_code','call_put','buy_amount','lastprice'))
            allvalue=account
            dsfsdf=1
            

        #策略一 简单的买入所有认购

        buy_all_value=allvalue*buy_pct

        if(cur_date>20231100):
            print(filtered_df)
            
        buylist=filtered_df    
        buylist=buylist.sort_values(by=['close'])
        #这里改成代号
        # mapping_call_put = {'C': 1, 'P': 2}
        buylist=buylist[buylist['call_put']==1]
        buylist=buylist[buylist['close']>0.05]
        buylist=buylist[buylist['days_remain']<50]
        buylist=buylist[buylist['days_remain']>5]

        buylist=buylist.head(1)
        
        if(cur_date>20231100):
            print(buylist)
            
        #根据buylist做后期计算
        buylist.loc[:,'buyuse']=buy_all_value/(buylist['ETF_close']*10000*buylist['Exercise_pred'])

        #buylist['buyuse']=code_amount_buy/buylist['close']
        buylist.loc[:,'buyuse']=buylist['buyuse'].round(0)
        buylist.loc[:,'buyuse']=buylist['buyuse'].astype(int)
        buylist['value']=buylist['close']*buylist['buyuse']*10000
        #print(buylist)

        savebuylist=buylist[['ts_code','call_put','buyuse','close']]
        savebuylist.columns = ['ts_code','call_put','buy_amount','lastprice']
        savebuylist['last_action_flag']=0
        
        if buyorsell == 0 :          
            account=account-buylist['value'].sum()-buylist['buyuse'].sum()*2
        else:
            account=account+buylist['value'].sum()-buylist['buyuse'].sum()*2
        hold_list=hold_list.append(savebuylist)
        hold_list.reset_index(inplace=True,drop=True)

        #计算当前总资产
        hold_list_value=hold_list['buy_amount']*hold_list['lastprice']*10000
        #这里当卖出        
        if buyorsell == 0 :          
            allvalue=hold_list_value.sum()+account
        else:
            allvalue=-hold_list_value.sum()+account

        single_random_number = np.random.normal(mean, std_dev, size=1)
        
        #baselinerd=baselinerd*(1+single_random_number/100)

        baselinerd=allvalue/accountbase
        print(allvalue)
        print(cur_date)        

        show3.append(baselinerd)


    days=np.arange(1,datelist.shape[0]+1)

    #每隔5日显示一个数据
    eee=np.where(days%5==0)
    daysshow=days[eee]
    datashow=datelist[eee]
    
    plt.plot(days,show3,c='green',label="TOPK _open_head30")
    plt.plot(days,baseline50,c='red',label="300ETF_baseline")
    plt.xticks(daysshow, datashow,color='blue',rotation=60)
    #plt.yscale("log")
    plt.legend()
    plt.show()

    #daily_all_df.to_csv('try1216.csv',encoding='utf-8-sig')

    #循环每日数据


    #根据数据生成图表


    zzzz=1

def opt_datemerge():

    pd.set_option('display.width', 5000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    
    #读取所有历史的日线数据
    daily_df=pd.read_csv('Daily_opt.csv',index_col=0,header=0,encoding='utf-8-sig')
    opt_basic_df=pd.read_csv('tushare_opt_basic_SSE.csv',header=0,encoding='utf-8-sig')

    df50etf=pd.read_csv('df50etf.csv',index_col=0,header=0,encoding='utf-8-sig')
    df300etf=pd.read_csv('df300etf.csv',index_col=0,header=0,encoding='utf-8-sig')
    df500etf=pd.read_csv('df500etf.csv',index_col=0,header=0,encoding='utf-8-sig')
    dfKCetf=pd.read_csv('dfKCetf.csv',index_col=0,header=0,encoding='utf-8-sig')
    dfCYBetf=pd.read_csv('dfCYBetf.csv',index_col=0,header=0,encoding='utf-8-sig')

    print(daily_df)
    print(opt_basic_df)
    
    opt_use_basic_df=opt_basic_df[['ts_code','opt_code','call_put','exercise_price','delist_date']]
    daily_use_df=daily_df[['ts_code','trade_date','pre_close','open','high','low','close','amount']]

    daily_all_df=pd.merge(daily_use_df,opt_use_basic_df,on='ts_code',how='left')

    df_etf_all = pd.concat([df50etf, df300etf,df500etf,dfKCetf,dfCYBetf], ignore_index=True)

    df_etf_use_all=df_etf_all[['ts_code','trade_date','open','high','low','close','pct_chg','vol','amount']]

    df_etf_use_all.rename(columns={'ts_code': 'opt_code','open': 'ETF_open', 'high': 'ETF_high', 'low': 'ETF_low', 'close': 'ETF_close'
                                   , 'pct_chg': 'ETF_pct_chg', 'vol': 'ETF_vol', 'amount': 'ETF_amount'}, inplace=True)


    daily_all_df['opt_code'] = daily_all_df['opt_code'].apply(remove_first_two_chars)

    daily_all_df=pd.merge(daily_all_df,df_etf_use_all,on=['opt_code','trade_date'],how='left')

    daily_all_df.sort_values(by='trade_date', ascending=True, inplace=True)    

    date1 = pd.to_datetime(daily_all_df['delist_date'], format='%Y%m%d')
    date2 = pd.to_datetime(daily_all_df['trade_date'], format='%Y%m%d')
    timedelta = date1 - date2
    daily_all_df['days_remain'] = timedelta.dt.days

    print(daily_all_df)

    daily_all_df=daily_all_df.sort_values(by='trade_date')
    last_values = daily_all_df.groupby('ts_code')['close'].last().reset_index()

    last_values.rename(columns={'close': 'last_close'}, inplace=True) 

    daily_all_df = pd.merge(daily_all_df,last_values,on='ts_code',how='left')
    
    daily_all_df['Exercise_Status']=0
    daily_all_df.loc[daily_all_df['last_close'] > 0.005, 'Exercise_Status'] = 1

    daily_all_df['real_value']=daily_all_df['ETF_close']-daily_all_df['exercise_price']


    daily_all_df.to_csv('try1224.csv',encoding='utf-8-sig')

    ddd=1

if __name__ == '__main__':

    print("hw")

    f = open('token.txt')
    token = f.read()     #将txt文件的所有内容读入到字符串str中
    f.close()

    #修改显示行列数
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)

    pro = ts.pro_api(token)
    
    #get_ETF_info(pro)

    #downloadall(pro)
    #opt_datemerge()

    #train_option_exercise_probability_model()
    
    opt_backTesting()

    #from_chatgpt()

    df_opt_basic_sse=pd.read_csv('tushare_opt_basic_1126.csv',index_col=0,header=0,encoding='utf-8-sig')

    print(df_opt_basic_sse)

    df_optdaily=pd.read_pickle('Daily_opt.pkl')

    print(df_optdaily)
    
    result=pd.merge(df_optdaily,df_opt_basic_sse,on='ts_code',how='left')

    print(result)


    action=input()



    