import pandas as pd  
from datetime import datetime
import backtrader as bt
import matplotlib.pyplot as plt

#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']

#使用tushare旧版接口获取数据
# import tushare as ts 
# def get_data(code,start='2015-01-01',end='2020-03-31'):
#     df=ts.get_k_data(code,autype='qfq',start=start,end=end)
#     df.index=pd.to_datetime(df.date)
#     df['openinterest']=0
#     df=df[['open','high','low','close','volume','openinterest']]
#     return df

class my_strategy1(bt.Strategy):
    #全局设定交易策略的参数
    params=(
        ('maperiod',20),
           )

    def __init__(self):
        #指定价格序列
        self.dataclose=self.datas[0].close
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buyprice = None
        self.buycomm = None

        #添加移动均线指标，内置了talib模块
        self.sma = bt.indicators.SimpleMovingAverage(
                      self.datas[0], period=self.params.maperiod)
        self.hilo_diff = self.datas[0].high - self.datas[0].low
        self.close_sma_diff = self.dataclose - self.sma
    def next(self):
        if self.order: # 检查是否有指令等待执行, 
            return
        # 检查是否持仓   
        if not self.position: # 没有持仓
            #执行买入条件判断：收盘价格上涨突破20日均线
            if self.dataclose[0] > self.sma[0]:
                #执行买入
                self.order = self.buy(size=500)         
        else:
            #执行卖出条件判断：收盘价格跌破20日均线
            if self.dataclose[0] < self.sma[0]:
                #执行卖出
                self.order = self.sell(size=500)

# 加载数据
# dataframe=get_data('600000')
# dataframe.to_csv("data.csv")
dataframe = pd.read_csv("data.csv", index_col=0, parse_dates=True)
#回测期间
start=datetime(2015, 3, 31)
end=datetime(2020, 3, 31)
data = bt.feeds.PandasData(dataname=dataframe,fromdate=start,todate=end)


# 初始化cerebro回测系统设置                           
cerebro = bt.Cerebro()  
#将数据传入回测系统
cerebro.adddata(data) 
# 将交易策略加载到回测系统中
cerebro.addstrategy(my_strategy1) 
# 设置初始资本为10,000
startcash = 10000
cerebro.broker.setcash(startcash) 
# 设置交易手续费为 0.2%
cerebro.broker.setcommission(commission=0.002) 

d1=start.strftime('%Y%m%d')
d2=end.strftime('%Y%m%d')
print(f'初始资金: {startcash}\n回测期间：{d1}:{d2}')
#运行回测系统
cerebro.run()
#获取回测结束后的总资金
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash
#打印结果
print(f'总资金: {round(portvalue,2)}')