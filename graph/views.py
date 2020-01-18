from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, mpld3
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns
import quandl
import scipy.optimize as sco
from scipy.optimize import minimize
from pandas.plotting import register_matplotlib_converters
from hedge.models import Post, Basic
from django.http import HttpResponse

## time consuming data collection and calculation (run once when the server start)
register_matplotlib_converters()
matplotlib.use('agg')

plt.style.use('fivethirtyeight')
np.random.seed(777)

num_stocks = Post.objects.count()
info = Basic.objects.first()
investment_duration = info.duration
return_target = info.target_return

stocks = []
expected_annual_returns = []
expected_daily_returns = []

for i in Post.objects.order_by('ticker'):
    stocks.append(i.ticker)
    expected_annual_returns.append(i.expected_return/info.duration)

for i in expected_annual_returns:
    expected_daily_returns.append(i/252.0)

# stocks = ['AAPL', 'AMZN', 'CSCO', 'CVS', 'DIS', 'GOOGL', 'JPM', 'MSFT', 'V', 'WFC']
# investment_duration = 1
# return_target = 0.22
# expected_daily_returns = [0.0007936507936507937, 0.001507936507936508, 0.0005555555555555556, 0.0011904761904761904, 0.0003492063492063492, 0.0004365079365079365, 0.0010317460317460319, 0.0004761904761904762, 0.00029365079365079364, 0.0011111111111111111]
# expected_annual_returns =  [0.2, 0.38, 0.14, 0.3, 0.088, 0.11, 0.26, 0.12, 0.074, 0.28]

quandl.ApiConfig.api_key = 'PrW6L55BexiSBEqWU25u'
data = quandl.get_table('WIKI/PRICES', 
                            ticker = stocks,
                            qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                            date = { 'gte': '2016-1-1', 'lte': '2017-12-31'}, paginate=True)
df = data.set_index('date')
table = df.pivot(columns='ticker')
# By specifying col[1] in below list comprehension
# You can select the stock names under multi-level column
table.columns = [col[1] for col in table.columns]

# -------------------------------------------------------------------------------

returns = table.pct_change()
mean_returns = returns.mean()
expected_returns = pd.Series(expected_daily_returns, index = stocks)
cov_matrix = returns.cov()
num_portfolios = 50000
risk_free_rate = 0.0178


# 2.1
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, return_target):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    if return_target!=0:  # 3.1
        target_min_vol = efficient_return(expected_returns, cov_matrix, return_target)
        target_sdp_min, target_rp_min = portfolio_annualised_performance(target_min_vol['x'], mean_returns, cov_matrix)
        target_min_vol_allocation = pd.DataFrame(target_min_vol.x,index=table.columns,columns=['allocation'])
        target_min_vol_allocation.allocation = [round(i*100,2)for i in target_min_vol_allocation.allocation]
        target_min_vol_allocation = target_min_vol_allocation.T
        return results, weights, sdp, rp, max_sharpe_allocation, min_vol_allocation, sdp_min, rp_min, target_min_vol, target_rp_min, target_sdp_min, target_min_vol_allocation
    else:   #2.1
        return results, weights, sdp, rp, max_sharpe_allocation, min_vol_allocation, sdp_min, rp_min
# for 2.1 
results21, weights21, sdp21, rp21, max_sharpe_allocation21, min_vol_allocation21, sdp_min21, rp_min21, = display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, return_target=0)


# 2.2
def con1(x):
    return p.sum(x) - 1

constraints = {'type': 'eq', 'fun': con1}

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
    method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
    method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result
def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, return_target=0):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    if return_target!=0:
        target_min_vol = efficient_return(expected_returns, cov_matrix, return_target)
        target_sdp_min, target_rp_min = portfolio_annualised_performance(target_min_vol['x'], mean_returns, cov_matrix)
        target_min_vol_allocation = pd.DataFrame(target_min_vol.x,index=table.columns,columns=['allocation'])
        target_min_vol_allocation.allocation = [round(i*100,2)for i in target_min_vol_allocation.allocation]
        target_min_vol_allocation = target_min_vol_allocation.T
        return max_sharpe, sdp, rp, max_sharpe_allocation, min_vol_allocation ,min_vol, sdp_min, rp_min, target_min_vol, target_sdp_min, target_rp_min, target_min_vol_allocation
    else:
        return max_sharpe, sdp, rp, max_sharpe_allocation, min_vol_allocation ,min_vol, sdp_min, rp_min

max_sharpe22, sdp22, rp22, max_sharpe_allocation22, min_vol_allocation22, min_vol22, sdp_min22, rp_min22 = display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, return_target=0)
results22 = results21

def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, return_target):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    if return_target!=0:
        target_min_vol = efficient_return(expected_returns, cov_matrix, return_target)
        target_sdp_min, target_rp_min = portfolio_annualised_performance(target_min_vol['x'], mean_returns, cov_matrix)
        target_min_vol_allocation = pd.DataFrame(target_min_vol.x,index=table.columns,columns=['allocation'])
        target_min_vol_allocation.allocation = [round(i*100,2)for i in target_min_vol_allocation.allocation]
        target_min_vol_allocation = target_min_vol_allocation.T
        return an_vol, an_rt, target_min_vol, target_rp_min, target_sdp_min, target_min_vol_allocation
    else:
        return an_vol, an_rt
# 2.3
an_vol, an_rt = display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, return_target=0)

# 3.1
results31, weights31, sdp31, rp31, max_sharpe_allocation31, min_vol_allocation31, sdp_min31, rp_min31, target_min_vol, target_rp_min, target_sdp_min, target_min_vol_allocation = display_simulated_ef_with_random(expected_returns, cov_matrix, num_portfolios, risk_free_rate, return_target)

# 3.2
weights32, sdp32, rp32, max_sharpe_allocation32, min_vol_allocation32, min_vol32, sdp_min32, rp_min32, target_min_vol2, target_sdp_min2, target_rp_min2, target_min_vol_allocation2 = display_calculated_ef_with_random(expected_returns, cov_matrix, num_portfolios, risk_free_rate, return_target)
results32 = results31
# 3.3
an_vol3, an_rt3, target_min_vol3, target_rp_min3, target_sdp_min3, target_min_vol_allocation3 = display_ef_with_selected(expected_returns, cov_matrix, risk_free_rate, return_target)
print('---------------------run calculation---------------------')
## end
# -------------------------------------------------------------------------------

from django.urls import reverse

def refresh(request):
    # 1.0
    global num_stocks, info, investment_duration, return_target, stocks, expected_annual_returns, expected_daily_returns, data, df, table
    num_stocks = Post.objects.count()
    info = Basic.objects.first()
    investment_duration = info.duration
    return_target = info.target_return

    stocks = []
    expected_annual_returns = []
    expected_daily_returns = []

    for i in Post.objects.order_by('ticker'):
        stocks.append(i.ticker)
        expected_annual_returns.append(i.expected_return/info.duration)

    for i in expected_annual_returns:
        expected_daily_returns.append(i/252.0)

    quandl.ApiConfig.api_key = 'PrW6L55BexiSBEqWU25u'
    data = quandl.get_table('WIKI/PRICES', 
                                ticker = stocks,
                                qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                                date = { 'gte': '2016-1-1', 'lte': '2017-12-31'}, paginate=True)
    df = data.set_index('date')
    table = df.pivot(columns='ticker')
    # By specifying col[1] in below list comprehension
    # You can select the stock names under multi-level column
    table.columns = [col[1] for col in table.columns]

    # 2.0
    global returns, mean_returns, expected_returns, cov_matrix, num_portfolios, risk_free_rate
    returns = table.pct_change()
    mean_returns = returns.mean()
    expected_returns = pd.Series(expected_daily_returns, index = stocks)
    cov_matrix = returns.cov()
    num_portfolios = 30
    risk_free_rate = 0.0178

    # 2.1
    global results21, weights21, sdp21, rp21, max_sharpe_allocation21, min_vol_allocation21, sdp_min21, rp_min21
    results21, weights21, sdp21, rp21, max_sharpe_allocation21, min_vol_allocation21, sdp_min21, rp_min21, = display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, return_target=0)

    # 2.2
    global max_sharpe22, sdp22, rp22, max_sharpe_allocation22, min_vol_allocation22, min_vol22, sdp_min22, rp_min22, results22
    max_sharpe22, sdp22, rp22, max_sharpe_allocation22, min_vol_allocation22, min_vol22, sdp_min22, rp_min22 = display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, return_target=0)
    results22 = results21

    # 2.3
    global an_vol, an_rt
    an_vol, an_rt = display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, return_target=0)

    # 3.1
    global results31, weights31, sdp31, rp31, max_sharpe_allocation31, min_vol_allocation31, sdp_min31, rp_min31, target_min_vol, target_rp_min, target_sdp_min, target_min_vol_allocation
    results31, weights31, sdp31, rp31, max_sharpe_allocation31, min_vol_allocation31, sdp_min31, rp_min31, target_min_vol, target_rp_min, target_sdp_min, target_min_vol_allocation = display_simulated_ef_with_random(expected_returns, cov_matrix, num_portfolios, risk_free_rate, return_target)

    # 3.2
    global weights32, sdp32, rp32, max_sharpe_allocation32, min_vol_allocation32, min_vol32, sdp_min32, rp_min32, target_min_vol2, target_sdp_min2, target_rp_min2, target_min_vol_allocation2
    weights32, sdp32, rp32, max_sharpe_allocation32, min_vol_allocation32, min_vol32, sdp_min32, rp_min32, target_min_vol2, target_sdp_min2, target_rp_min2, target_min_vol_allocation2 = display_calculated_ef_with_random(expected_returns, cov_matrix, num_portfolios, risk_free_rate, return_target)
    results32 = results31
    # 3.3
    global an_vol3, an_rt3, target_min_vol3, target_rp_min3, target_sdp_min3, target_min_vol_allocation3
    an_vol3, an_rt3, target_min_vol3, target_rp_min3, target_sdp_min3, target_min_vol_allocation3 = display_ef_with_selected(expected_returns, cov_matrix, risk_free_rate, return_target)
    print(expected_daily_returns)    
    return price(request)

# view -------------------------------------------------------------------------------

def price(request):
    fig = plt.figure(figsize=(11, 6))
    for c in table.columns.values:
        plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('price in $')
    plt.title('Past performance of price in $')
    context = {
        'chart': mpld3.fig_to_html(fig)
    }
    return render(request, 'graph/price.html', context)

def daily_return(request):
    fig = plt.figure(figsize=(11, 6))
    returns = table.pct_change()
    for c in returns.columns.values:
        plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel('daily returns')
    plt.title('Past performance on daily returns')

    context = {
        'chart': mpld3.fig_to_html(fig)
    }
    return render(request, 'graph/daily_return.html', context)

# -------------------------------------------------------------------------------

def avg_return(request):
    rtn = []
    for i in range(num_stocks):
        rtn.append({'name': stocks[i],
                        'mean': '{:.6f}'.format(mean_returns[i]),
                        'exp': '{:.6f}'.format(expected_returns[i]) })
    context = {
        'return': rtn
    }
    return render(request, 'graph/past/avg_return.html', context)
#  2.1
def pastEstP(request):
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.scatter(results21[0,:],results21[1,:],c=results21[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    legend_elements = [Line2D([0], [0], marker='*', color='w', label='Maximum Sharperatio', markerfacecolor='r', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility', markerfacecolor='g', markersize=25) ]
    ax.legend(handles=legend_elements)
    ax.scatter(sdp21,rp21,marker='*',color='r',s=500)
    ax.scatter(sdp_min21,rp_min21,marker='*',color='g',s=500)
    # ax.plot([], [], "*", color="r", markersize=30, label='Maximum Sharperatio')
    # ax.plot([], [], "*", color="g", markersize=30, label='Minimum Volatility')
    # if return_target!=0:
    #     plt.scatter(efficient_return(expected_returns, cov_matrix, return_target).fun ,return_target,marker='*',color='b',s=500, label='Minimum volatility of the target return')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')

    columns = list(max_sharpe_allocation21)
    tb = []
    for i in columns: 
        tb.append({ 'name': i,
                    'max': '{:.2f}'.format(max_sharpe_allocation21[i][0]),
                    'min': '{:.2f}'.format(min_vol_allocation21[i][0]),
        })
    context = {
        'chart': mpld3.fig_to_html(fig),
        'table': tb,
        'maxRtn': round(rp21,2),
        'maxVol': round(sdp21,2),
        'minRtn': round(rp_min21,2),
        'minVol': round(sdp_min21,2) 
    }
    return render(request, 'graph/past/estP.html', context)
# 2.2
def pastCalP(request):
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.scatter(results22[0,:],results22[1,:],c=results22[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp22,rp22,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min22,rp_min22,marker='*',color='g',s=500, label='Minimum volatility')
    legend_elements = [ Line2D([0], [0], ls='-.', color='black', label='Efficient Frontier'),
                        Line2D([0], [0], marker='*', color='w', label='Maximum Sharperatio', markerfacecolor='r', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility', markerfacecolor='g', markersize=25) ]
    ax.legend(handles=legend_elements)
    # if return_target!=0:
    #     plt.scatter(efficient_return(expected_returns, cov_matrix, return_target).fun, return_target, marker='*',color='b',s=500, label='Minimum volatility of the target return')
    #     target = np.linspace(rp_min, max(expected_daily_returns)*252, 50)
    # else:
    target = np.linspace(rp_min22, max(mean_returns)*252, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')

    columns = list(max_sharpe_allocation21)
    tb = []
    for i in columns: 
        tb.append({ 'name': i,
                    'max': '{:.2f}'.format(max_sharpe_allocation22[i][0]),
                    'min': '{:.2f}'.format(min_vol_allocation22[i][0]) 
        })
    context = {
        'chart': mpld3.fig_to_html(fig),
        'table': tb,
        'maxRtn': round(rp22,2),
        'maxVol': round(sdp22,2),
        'minRtn': round(rp_min22,2),
        'minVol': round(sdp_min22,2) 
    }
    return render(request, 'graph/past/CalP.html', context)
# 2.3 (2.2)
def pastPvsI(request):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(an_vol,an_rt,marker='o',s=200)
    for i, txt in enumerate(table.columns):
        plt.text(an_vol[i]+.005, an_rt[i]-.005, txt)
        # ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp22,rp22,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min22,rp_min22,marker='*',color='g',s=500, label='Minimum volatility')
    legend_elements = [ Line2D([0], [0], ls='-.', color='black', label='Efficient Frontier'),
                        Line2D([0], [0], marker='*', color='w', label='Maximum Sharperatio', markerfacecolor='r', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility', markerfacecolor='g', markersize=25) ]
    ax.legend(handles=legend_elements)
    # if return_target!=0:
    #     plt.scatter(efficient_return(expected_returns, cov_matrix, return_target).fun, return_target, marker='*',color='b',s=500, label='Minimum volatility of the target return')
    #     target = np.linspace(rp_min, max(expected_daily_returns)*252, 50)
    # else:
    target = np.linspace(rp_min22, max(mean_returns)*252, 50)

    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')

    columns = list(max_sharpe_allocation22)
    tb = []
    for i in columns: 
        tb.append({ 'name': i,
                    'max': '{:.2f}'.format(max_sharpe_allocation22[i][0]),
                    'min': '{:.2f}'.format(min_vol_allocation22[i][0]) ,
                    'anRtn': round(an_rt[i],2),
                    'anVol': round(an_vol[i],2)
        })
    context = {
        'chart': mpld3.fig_to_html(fig),
        'table': tb,
        'maxRtn': round(rp22,2),
        'maxVol': round(sdp22,2),
        'minRtn': round(rp_min22,2),
        'minVol': round(sdp_min22,2) 
    }

    return render(request, 'graph/past/PvsI.html', context)

# -------------------------------------------------------------------------------
# 3.1
def expEstP(request):
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.scatter(results31[0,:],results31[1,:],c=results31[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    legend_elements = [Line2D([0], [0], marker='*', color='w', label='Maximum Sharperatio', markerfacecolor='r', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility', markerfacecolor='g', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility of the target return', markerfacecolor='b', markersize=25)]
    ax.legend(handles=legend_elements)
    plt.scatter(sdp31,rp31,marker='*',color='r',s=500)
    plt.scatter(sdp_min31,rp_min31,marker='*',color='g',s=500)
    # ax.plot([], [], "*", color="r", markersize=30, label='Maximum Sharperatio')
    # ax.plot([], [], "*", color="g", markersize=30, label='Minimum Volatility')
    # if return_target!=0:
    plt.scatter(efficient_return(expected_returns, cov_matrix, return_target).fun ,return_target,marker='*',color='b',s=500)
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')

    columns = list(max_sharpe_allocation21)
    tb = []
    for i in columns: 
        tb.append({ 'name': i,
                    'max': '{:.2f}'.format(max_sharpe_allocation31[i][0]),
                    'min': '{:.2f}'.format(min_vol_allocation31[i][0]),
                    'tar': '{:.2f}'.format(target_min_vol_allocation[i][0]),
        })
    context = {
        'chart': mpld3.fig_to_html(fig),
        'table': tb,
        'maxRtn': round(rp31,2),
        'maxVol': round(sdp31,2),
        'minRtn': round(rp_min31,2),
        'minVol': round(sdp_min31,2),
        'tarRtn': round(target_rp_min,2),
        'tarVol': round(target_sdp_min,2)
    }
    return render(request, 'graph/expected/estP.html', context)
# 3.2
def expCalP(request):
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.scatter(results32[0,:],results32[1,:],c=results32[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp32,rp32,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min32,rp_min32,marker='*',color='g',s=500, label='Minimum volatility')
    legend_elements = [ Line2D([0], [0], ls='-.', color='black', label='Efficient Frontier'),
                        Line2D([0], [0], marker='*', color='w', label='Maximum Sharperatio', markerfacecolor='r', markersize=10),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility', markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility of the target return', markerfacecolor='b', markersize=10)]
    ax.legend(handles=legend_elements)
    # if return_target!=0:
    plt.scatter(efficient_return(expected_returns, cov_matrix, return_target).fun, return_target, marker='*',color='b',s=500, label='Minimum volatility of the target return')
    target = np.linspace(rp_min32, max(expected_daily_returns)*252, 50)
    # else:
    # target = np.linspace(rp_min, max(mean_returns)*252, 50)
    efficient_portfolios = efficient_frontier(expected_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')

    columns = list(max_sharpe_allocation21)
    tb = []
    for i in columns: 
        tb.append({ 'name': i,
                    'max': '{:.2f}'.format(max_sharpe_allocation32[i][0]),
                    'min': '{:.2f}'.format(min_vol_allocation32[i][0]),
                    'tar': '{:.2f}'.format(target_min_vol_allocation2[i][0]),
        })
    context = {
        'chart': mpld3.fig_to_html(fig),
        'table': tb,
        'maxRtn': round(rp32,2),
        'maxVol': round(sdp32,2),
        'minRtn': round(rp_min32,2),
        'minVol': round(sdp_min32,2),
        'tarRtn': round(target_rp_min2,2),
        'tarVol': round(target_sdp_min2,2)
    }
    return render(request, 'graph/expected/CalP.html', context)
# 3.3
def expPvsI(request):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(an_vol3,an_rt3,marker='o',s=200)
    for i, txt in enumerate(table.columns):
        plt.text(an_vol3[i]+.005, an_rt3[i]-.005, txt)
        # ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp32,rp32,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min32,rp_min32,marker='*',color='g',s=500, label='Minimum volatility')
    legend_elements = [ Line2D([0], [0], ls='-.', color='black', label='Efficient Frontier'),
                        Line2D([0], [0], marker='*', color='w', label='Maximum Sharperatio', markerfacecolor='r', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility', markerfacecolor='g', markersize=25),
                        Line2D([0], [0], marker='*', color='w', label='Minimum Volatility of the target return', markerfacecolor='b', markersize=25)]
    ax.legend(handles=legend_elements)
    # if return_target!=0:
    plt.scatter(efficient_return(expected_returns, cov_matrix, return_target).fun, return_target, marker='*',color='b',s=500, label='Minimum volatility of the target return')
    target = np.linspace(rp_min32, max(expected_daily_returns)*252, 50)
    # else:
    # target = np.linspace(rp_min, max(mean_returns)*252, 50)

    efficient_portfolios = efficient_frontier(expected_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')

    columns = list(max_sharpe_allocation21)
    tb = []
    for i in columns: 
        tb.append({ 'name': i,
                    'max': '{:.2f}'.format(max_sharpe_allocation32[i][0]),
                    'min': '{:.2f}'.format(min_vol_allocation32[i][0]),
                    'tar': '{:.2f}'.format(target_min_vol_allocation3[i][0]),
                    'anRtn': round(an_rt3[i],2),
                    'anVol': round(an_vol3[i],2)
        })
    context = {
        'chart': mpld3.fig_to_html(fig),
        'table': tb,
        'maxRtn': round(rp32,2),
        'maxVol': round(sdp32,2),
        'minRtn': round(rp_min32,2),
        'minVol': round(sdp_min32,2),
        'tarRtn': round(target_rp_min3,2),
        'tarVol': round(target_sdp_min3,2)
    }

    return render(request, 'graph/expected/PvsI.html', context)

# -------------------------------------------------------------------------------










