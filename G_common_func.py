import pandas as pd
import numpy as np
import math
import os
import time
from pandas.tseries.offsets import *


def makefolder(savename):
    # Savename is filename and directory.  Function will check to see of file exists.  If it does not exist the it will make file
    if not os.path.isdir(savename):
        os.mkdir(savename)

def calc_yoy(df):
    # Will calculate the YOY change of monthly data
    l_df = df / df.shift(12) - 1 
    return l_df

def combday(df1, df2):
    #Combines to dataframes and fills empty values.  Useful when s
    
    dft = pd.concat([df1, df2], axis =1)
    dftt = dft.ffill()
    #df_BD = dftt.asfreq(BDay())
    return dftt

def calculate_returnDEF(l_df, ndays, future):
    global calculate_return
    # caclulates percent change in dataframe.  Also can be caculated by l_df_pc = l_df.pct_change(1) for one day
    # this function caculates the change from before
    # l_df_pcN = l_df/l_df.shift(ndays)-1
    # caculates the change for later
    if future == 1:
        l_df_DEF = l_df - l_df.shift(-ndays) # caculates the increase in the future
    elif future == 0:
        l_df_DEF = l_df - l_df.shift(ndays) # calculates the increase since past
        
    return l_df_DEF

def binarize(val):
    if val<0:
        oval = 1
    elif val>=0:
        oval = 0
    return oval


def import_main_file(INpath, sheet):
    df = pd.read_excel(INpath, sheet)  # ORGINAL DATA
    lastday = df.index.map(lambda t: t.strftime('%Y-%m-%d'))[-1] 
    pathp = INpath.split('/')
    Name = pathp[-1]
    PATH= INpath[0:-(len(Name))]
    return df, lastday, Name, PATH
    

def import_main_file1(INpath, sheet):
    df = pd.read_excel(INpath, sheet)  # ORGINAL DATA
    lastday = df.index.map(lambda t: t.strftime('%Y-%m-%d'))[-1] 
    pathp = INpath.split('/')
    Name = pathp[-1]
    PATH= INpath[0:-(len(Name))]
    df.index.name = 'Date'
    return df, lastday, Name, PATH


def extract_sigfrom_mat(df, target, indicator):
    #extracts a target signal from matrix
    val = df[target][indicator]
    val_row = df[target].values
    
    return val, val_row

def calculate_return(l_df, ndays, future):
    global calculate_return
    # caclulates percent change in dataframe.  Also can be caculated by l_df_pc = l_df.pct_change(1) for one day
    # this function caculates the change from before
    # l_df_pcN = l_df/l_df.shift(ndays)-1
    # caculates the change for later
    if future == 1:
        l_df_pcN = l_df.shift(-ndays) / l_df - 1 # caculates the increase in the future
    elif future == 0:
        l_df_pcN = l_df / l_df.shift(ndays) - 1 # calculates the increase since past
        
    return l_df_pcN

def calcRET(df, n, path, Name, future):
    
    writer = pd.ExcelWriter(path+'/RET_'+Name)
    ldf_ret = calculate_return(df, n, future)
    ldf_ret.to_excel(writer, 'RET'+ str(n))
    writer.save()
    
    return ldf_ret

def getdates(df, cutday):
    
    df_m = df.resample('BM').last()
    df_Fm1 = df_m[:-1]
    cutidx = df_Fm1.index.get_loc(cutday)
    df_Fm = df_Fm1[cutidx:]
    dates_idx = df_Fm.index
    dates = pd.Series(dates_idx.format())
    
    return dates

def calc_ema(dat, j, pre): # j is the number of days back
    # data = dat.values
    # fpdata = data[::-1]
    p = dat[-1]
    # p = fpdata[:j+2] # 2 is not necessary but just in case
    ema = np.zeros([1, j]) # array for EMA values
    
    for n in range(1, j+1):
        prea = pre[n-1]
        ema[0][n-1] = p*2/(1+n) + prea - prea*2/(1+n)
    # old implementation (is incorrect)
    # for n in range (0, j):
    #    if n ==0:
    #        a[n] = p[0]
    #    else:
    #        a[n] = p[n-1]*2/(1+n)+a[n-1]*(1-2/(1+n))   
    # New implementation
    return ema    
            
    
def pat_cor(PTNs, trgt):
    
    ptncor = np.corrcoef(PTNs)    
    data = ptncor[:,trgt-1]
    s_idx = np.argsort(data)[::-1]
    SI = s_idx+1
    
    return SI
    
    
def month_cut(df, ntime):
    # Takes a dataframe and creates a new dataframe from the current date to ntime back
    # where ntime is the number of months back 
    
    df_m = df.resample('BM').last()
    cut_month = df_m.index.map(lambda t: t.strftime('%Y-%m-%d'))[-(ntime)] #Gets month to cut data for analysis
    
    dfT = df[cut_month:] #Cuts data
    
    return dfT

def perf_m(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # Calculates the True/False negative and positive
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        elif y_actual[i]==y_hat[i]==0:
           TN += 1
        else:
           FN += 1

    return(TP, FP, TN, FN)
    
def copy_Mo2day(df, dfm):
    #Copies the values derived from BM Dataframe to the daily dataframe
    dates_idx = dfm.index
    dates = pd.Series(dates_idx.format())
    dfo = df.copy()
    
    for n in dates:
        dfo[n:n] = dfm[n:n]
    
    return dfo

def get_folder_names(D_path):
    
    dirs = [d for d in os.listdir(D_path) if os.path.isdir(os.path.join(D_path, d))]    
    return dirs

def normalize_sig(dfn):

    df_norm = 2*(dfn - dfn.min()) / (dfn.max() - dfn.min()) -1    
    return df_norm


def perf_normalize_sig(dfn):
    df_norm = (dfn - dfn[0]) / dfn[0]    
    return df_norm

def get_path(foldername):
    p1 = foldername.strip()
    p2 = p1.split('/')    
    filename = p2[-1]
    path = p1[0:-len(filename)]    
    return filename, path
    
def savename(idx):
    
    sname = idx.replace(' ', '_')    
    return sname 

def divide_dfby(dfw):
    #divides dataframe by date
    DFlist = [group[1] for group in dfw.groupby([dfw.index.year, dfw.index.month])]
    
    return DFlist

def selectnweight(DFlist):
    a = len(DFlist)
    mlist = []
    w = []
    
    for n in range(0, a):
        w.append(DFlist[n].shape[0])
        midx = DFlist[n]['COR'].idxmax()
        mlist.append(midx.strftime('%Y-%m-%d'))
    return w, mlist
    
#def give_weight_df(df, COR, array, w_array):
#        
#    
    
