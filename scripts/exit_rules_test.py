import numpy as np, pandas as pd, sys
sys.path.insert(0,'/mnt/project-files/stock-backtest')
from twscreener.market import RESULTS, load_panel
from twscreener.backtest import COST
from twscreener.review import market_context
from twscreener.simulate import prices
raw,adj,opn,vol=prices(load_panel())
s=pd.read_csv(RESULTS/'market_signals.csv',dtype={'stock_id':str}); s['date']=pd.to_datetime(s.signal_date)
ctx=market_context()[['date','大盤狀態','大盤離高點']]
s=s.merge(ctx,on='date',how='left')
A=adj.to_numpy();O=opn.to_numpy();di={x:i for i,x in enumerate(adj.index)};ci={x:j for j,x in enumerate(adj.columns)}
P=[];keep=[]
for k,(t,sid) in enumerate(zip(s.date,s.stock_id)):
    i,j=di.get(t),ci.get(sid)
    if i is None or j is None or i+61>=len(A): continue
    e=O[i+1,j]
    if not np.isfinite(e): continue
    p=A[i+1:i+61,j]/e-1
    if np.isnan(p).any(): continue
    P.append(p);keep.append(k)
P=np.array(P);s=s.iloc[keep].reset_index(drop=True)
atr=adj.pct_change(fill_method=None).abs().rolling(20).mean()
s['atr']=[atr.iat[di[t],ci[x]] for t,x in zip(s.date,s.stock_id)]
cut=s.date.quantile(.5);s['h']=np.where(s.date<=cut,'前','後')
n=len(P);ar=np.arange(n)
def run(stop=None,tp=None,half_tp=None,trail=None):
    r=P[:,-1].copy();done=np.zeros(n,bool)
    if stop is not None:
        st=-stop*s.atr.to_numpy()[:,None]; hit=P<=st
        f=np.where(hit.any(1),hit.argmax(1),99)
    else: f=np.full(n,99)
    if tp is not None:
        h=P>=tp; g=np.where(h.any(1),h.argmax(1),99)
        r=np.where((g<f)&(g<99),P[ar,np.minimum(g,59)],np.where(f<99,P[ar,np.minimum(f,59)],r))
    elif half_tp is not None:
        h=P>=half_tp; g=np.where(h.any(1),h.argmax(1),99)
        # half sold at g, other half trailing stop trail from running max after g
        out=[]
        for k in range(n):
            if f[k]<=g[k] and f[k]<99: out.append(P[k,f[k]]);continue
            if g[k]==99: out.append(P[k,-1]);continue
            rest=P[k,g[k]:];peak=np.maximum.accumulate(rest);x=np.where(rest<=(1+peak)*(1-trail)-1)[0]
            out.append(0.5*P[k,g[k]]+0.5*(rest[x[0]] if len(x) else rest[-1]))
        r=np.array(out)
    else:
        r=np.where(f<99,P[ar,np.minimum(f,59)],r)
    return r-COST
res={}
for name,kw in {'抱60天':{},'停損3x波動':dict(stop=3),'停損+停利20%':dict(stop=3,tp=.2),'停損+停利30%':dict(stop=3,tp=.3),
                '停損+15%賣一半+移動停利10%':dict(stop=3,half_tp=.15,trail=.10)}.items():
    s[name]=run(**kw)
cols=['抱60天','停損3x波動','停損+停利20%','停損+停利30%','停損+15%賣一半+移動停利10%']
pd.set_option('display.width',250)
print('筆數',len(s),s.date.min().date(),s.date.max().date(),'cut',cut.date())
print('== 出場規則（全部策略，扣成本平均）'); print(s.groupby('h')[cols].mean().round(4).T.to_string())
print(s.groupby('strategy')[cols].mean().round(4).to_string())
print('== 大盤濾網：各狀態平均（停損3x波動）');print(s.groupby(['大盤狀態','h'])['停損3x波動'].agg(['mean','size']).round(4).unstack().to_string())
s['dd']=pd.cut(s['大盤離高點'],[-1,-.15,-.08,-.03,1])
print(s.groupby(['dd','h'])['停損3x波動'].agg(['mean','size']).round(4).unstack().to_string())
print(s.groupby(['strategy','大盤狀態'])['停損3x波動'].agg(['mean','size']).round(3).unstack().to_string())
print('== 停損寬度')
for k in (3,5,8):
    r=run(stop=k); print(f'{k}x波動', pd.Series(r).groupby(s.h).mean().round(4).to_dict(), '最差5%',np.quantile(r,.05).round(3), '勝率',(r>0).mean().round(3))
r=P[:,-1]-COST; print('不停損','最差5%',np.quantile(r,.05).round(3),'勝率',(r>0).mean().round(3))
for fx in (.15,.2,.25):
    hit=P<=-fx; f=np.where(hit.any(1),hit.argmax(1),99); r=np.where(f<99,P[ar,np.minimum(f,59)],P[:,-1])-COST
    print(f'固定-{fx:.0%}',pd.Series(r).groupby(s.h).mean().round(4).to_dict(),'最差5%',np.quantile(r,.05).round(3))
