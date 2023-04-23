import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.reload_library()
# plt.style.use(['science','no-latex'])
# myparams = {
#    'axes.labelsize': 6,
#    'xtick.labelsize': 6,
#     'lines.markersize' : 4.5,
#    'ytick.labelsize': 6,
#    'lines.linewidth': 1,
#    'legend.fontsize': 6,
#     "axes.titlesize": 14,
#     "figure.titlesize": 14,
#    'font.family': 'Times New Roman',
#       'figure.figsize': '7,7',  #图片尺寸
#     'mathtext.fontset':'stix',
#     'savefig.dpi':100,
#     'figure.dpi':100
# }
# plt.rcParams.update(myparams)  #更新自己的设置

def plotly_line(df, mode_lst=["lines", "markers"], show=True, jupyter_show=False, file_name=None):
    '''
    x:df.index 
    y:iter df.columns
    file_name: save if not None
    '''
    mode = "+".join(mode_lst)
    lst = []
    for col in df.columns:
        trace = go.Scatter(x=df.index, y=df[col], mode = mode, name=col)
        lst.append(trace)
    layout = go.Layout(xaxis=dict(title=""), yaxis=dict(title=""))
    fig = go.Figure(data=lst, layout=layout)

    if show:
        if jupyter_show:
            # 使用py.offline.init_notebook_mode()进行初始化，利用plotly.offline.iplot函数可在Jypyter notebook直接绘图。
            # 如果仅展示，可以直接fig.show() 或者 
            plotly.offline.init_notebook_mode()
            plotly.offline.iplot(fig)
        else:
            # 等价于fig.show()
            plotly.offline.iplot(fig)

    if file_name:
        # 在本地新建一个HTML文件，并可以选择是否在浏览器中打开这个文件。
        plotly.offline.plot(fig, filename=file_name)
        
    return fig


import statsmodels.tsa.api as smt
def plot_acf(x, acf_lag=10, pacf_lag=10):
    fig = plt.figure(figsize=(10,8))
    layout = (3,2)
    ax1 = plt.subplot2grid(layout, (0,0))
    ax2 = plt.subplot2grid(layout, (0,1))

    smt.graphics.plot_acf(x, ax=ax1, lags=acf_lag, alpha=0.5)
    smt.graphics.plot_pacf(x, ax=ax2, lags=pacf_lag, alpha=0.5)

    ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)
    ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)
    ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)
    ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)
    #ax1.set_ylim([0,0.2])
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    return fig
   
def plot_scatter(__y_train,__y_pred):
    # 散点图
    x_ = np.linspace(2, 10)
    y = x_
    fig = plt.gcf()
    plt.scatter(__y_train, __y_pred, edgecolors=(0, 0, 0), label = 'data')
    plt.plot(y, x_ ,'orange', label = 'Correct line')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.legend()
    return fig
def plot_line(__y_train,__y_pred):
    # 折线图
    fig = plt.gcf()
    plt.plot(__y_train, label="True")
    plt.plot(__y_pred, label="Pred")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    return fig
def plot_metrics(metrics):
    fig = plt.gcf()
    plt.plot(metrics[:, 0], 'o-', color="r",
             label="Cross validation R2")
    plt.plot(metrics[:, 1], 'o-', color="g",
             label="Cross validation MSE")
    plt.xlabel("Kth Fold")
    plt.title('Evaluation Metrics')
    plt.legend(loc="best")
    return fig
    
import seaborn as sns
# palette = sns.color_palette('Set1', 3)
# sns.lineplot(x='month', y='pm2.5', data=data, ci=50, hue='year', ax=ax,
#             palette=palette)
# sns.lineplot(x='timepoint',y='no2', data=data, color=palette[2],ax=ax, ci=c)
# sns.countplot(x='label', data=train_df, ax=ax1)
# sns.heatmap(train_tone_df[corr_columns].astype(float).corr(), cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1)

# # method1
# fig = plt.figure()
# # submethod1
# plt.subplot(221)
# plt.plot()
# # submethod2
# ax = plt.subplot(221)
# ax.plot()
# # submethod3
# ax = fig.add_subplot(2,2,1)
# # method2
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# fig.subplots_adjust(wspace=0.6,hspace=0.4)
# #plt.subplots_adjust(wspace=0.6,hspace=0.4)
# fig.suptitle(t='Hour',x=0.5,y=0.52,va='bottom',ha='center',fontsize=20)
# ax.plot(#x, y, label="", color="", marker="", linestyle="", linewidth=2, markersize=12)
# ax.set_xlim([min, max])
# ax.set_ylim([min, max])
# ax.set_xlabel('Hour')
# ax.set_ylabel('PM2.5 concentrations '+r'$\mathrm{(\mu g/m^3)}$')
# ax.set_title("(a)Beijing")
# ax.grid(alpha=0.3,axis='y')
# ax.set_xticks("")
# ax.set_xticklabels("") # 等价于 plt.xticks(xtick, xtick_label)
# ax.text(x=m, y=n, s=n, verticalalignment='top', horizontalalignment='center', fontsize=fzt)
# ax.legend()
# ax.legend((line1, line2), ('Cold', 'Warm'),loc='upper right',ncol=3,framealpha=0.5,fancybox=False,edgecolor='black', bbox_to_anchor=(1.8,0.65))

# width = 0.05
# x = np.arange(len(model_name_lst)).tolist()
# for i in range(len(x)):  
#     x[i] = x[i] - width/2

# a = plt.bar(x, r2_whole_data, width = width, label = 'R2', fc = 'C0', ec=(0,0,0))

# x = np.arange(len(model_name_lst)).tolist()
# for i in range(len(x)):  
#     x[i] = x[i] + width/2 

# b = plt.bar(x, mse_whole_data, width = width, label = 'MSE', fc = 'C1', ec=(0,0,0))
# plt.xticks(np.arange(len(model_name_lst)).tolist(), model_name_lst)
#     for m,n in zip(x,df.iloc[i,:].values):
#         _=axes[st].text(x=m,y=n,s=n,verticalalignment='bottom',horizontalalignment='center',
#                         fontsize=fzt)
#     for j in range(3):
#         x[j]+=wid
# _=axes[st].set_ylabel(col)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(dict(x=np.random.randn(100)), index=pd.date_range("2020-01-01",periods=100))
    a = plotly_line(df, False, False)
    print("end")