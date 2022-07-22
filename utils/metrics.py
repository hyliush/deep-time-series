import numpy as np
import scipy.stats as st

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt((((true-true.mean(0))**2).sum(0)*((pred-pred.mean(0))**2).sum(0)))
    return (u/d)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    assert len(pred) == len(true)
    denominator = (np.abs(true) + np.abs(pred))
    diff = np.abs(true - pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def ND(pred, true):
    assert len(pred) == len(true)
    demoninator = np.sum(np.abs(true))
    diff = np.sum(np.abs(true - pred))
    return 1.0*diff/demoninator

def RMSLE(pred, true) :
    assert len(pred) == len(true)
    assert len(true) == len(pred)
    return np.sqrt(np.mean((np.log(1+pred) - np.log(1+true))**2))

def NRMSE(pred, true):
    assert len(pred) == len(true)
    denominator = np.mean(true)
    diff = np.sqrt(np.mean(((pred-true)**2)))
    return diff/denominator

def rhoRisk2(pred,true,rho):
    assert len(pred) == len(true)
    diff1 = (true-pred)*rho*(true>=pred)
    diff2 = (pred-true)*(1-rho)*(true<pred)
    denominator = np.sum(true)
    return 2*(np.sum(diff1)+np.sum(diff2))/denominator

def rhoRisk(pred,true,rho):
    assert len(pred) == len(true)
    diff = -np.sum((pred-true)*(rho*(pred<=true)-(1-rho)*(pred>true)))
    denominator = np.sum(true)
    return diff/abs(denominator)

def distribution_metric(pred, true):
    '''高斯分布，最后一维为均值方差
    分位数分布，最后一维为0.1， 0.5， 0.9分位数'''
    if pred.shape[-1] == 2:
        mu, sigma = pred[..., 0], pred[..., 1]
        PredQ50 = mu.copy()
        PredQ90 = st.norm.ppf(0.9, mu, sigma)
    else:
        PredQ50 = pred[..., -2]
        PredQ90 = pred[..., -1]

    # The evaluation metrics
    rho50 = rhoRisk(PredQ50.reshape(-1,), true.reshape(-1,), 0.5)
    rho90 = rhoRisk(PredQ90.reshape(-1,), true.reshape(-1,), 0.9)
    return dict(zip(("rho50","rho90"), (rho50, rho90)))

def point_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return dict(zip(("mae","mse","rmse","mape","mspe"), (mae,mse,rmse,mape,mspe)))