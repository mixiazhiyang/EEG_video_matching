import numpy as np


def calc_L(X, k, m):
    """
    Return Lm(k) as the length of the curve.

    """
    N = X.size

    n = np.floor((N-m)/k).astype(np.int64)
    norm = (N-1) / (n*k)

    sum = np.sum(np.abs(np.diff(X[m::k], n=1)))

    Lm = (sum*norm) / k

    return Lm


def calc_L_average(X, k):
    """
    Return <L(k)> as the average value over k sets of Lm(k).

    """
    calc_L_series = np.frompyfunc(lambda m: calc_L(X, k, m), 1, 1)

    L_average = np.average(calc_L_series(np.arange(1, k+1)))

    return L_average


def measure(X, k_max):
    """
    Measure the fractal dimension of the set of points (t, f(t)) forming
    the graph of a function f defined on the unit interval.

    Parameters
    ----------
    X : ndarray
        time series.

    k_max : int
        Maximum interval time that a new series.

    Returns
    -------
    D : float
        Fractal dimension.

    Examples
    --------
    >>> N = np.power(2, 15)
    >>> X = np.sin(np.linspace(0, 1000, N))
    >>> j = 11
    >>> k_max = np.floor(np.power(2, (j-1)/4)).astype(np.int64)
    >>> D = hfda.measure(X, k_max)
    >>> D
    1.0005565919808783

    """
    calc_L_average_series = np.frompyfunc(lambda k: calc_L_average(X, k), 1, 1)

    k = np.arange(1, k_max+1)
    L = calc_L_average_series(k).astype(np.float64)

    D, _ = - np.polyfit(np.log2(k), np.log2(L), 1)

    return D

def measure_signal(sig,k=5):
    # sig:[channel,length]
    D=np.stack([measure(sig[i],k) for i in range(sig.shape[0])],axis=0)
    return D # [channel,]


# hjorth param
def activity(x):
    # x:[channel,length]
    return np.var(x,axis=-1)


def first_order_deriative(x,):
    # x:[channel,length]
    fo=x[:,2:]-x[:,:-2]
    return fo

def mobility(x):
    # x:[channel,length]
    return np.sqrt(np.var(first_order_deriative(x),axis=-1)/np.var(x,axis=-1))
    
def complexity(x):
    return mobility(first_order_deriative(x))/mobility(x)

from scipy import signal

def bandpass_filter(signal_in, lowcut, highcut, fs, order=5):
    # 计算归一化截止频率
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    # 使用firwin函数设计FIR滤波器
    taps = signal.firwin(order, [low, high], pass_zero=False)    
    # 使用lfilter函数对信号进行滤波
    signal_out = signal.lfilter(taps, 1.0, signal_in)    
    return signal_out

def filt_band_pass(x,freq_low,freq_high,sr,filter_type='fir', order=5):
    if filter_type=='fir':
        # 计算归一化截止频率
        nyquist_freq = 0.5 * sr
        low = freq_low / nyquist_freq
        high = freq_high / nyquist_freq
        # 使用firwin函数设计FIR滤波器
        taps = signal.firwin(order, [low, high], pass_zero=False)    
        # 使用lfilter函数对信号进行滤波
        signal_out = signal.lfilter(taps, 1.0, x)    
        return signal_out
    else:
        return signal.filtfilt(*signal.butter(4,[freq_low*2/sr,freq_high*2/sr],'bandpass'),x)

def filt_low_pass(x,freq_low,sr):
    return signal.filtfilt(*signal.butter(4,freq_low*2/sr,'lowpass'),x)

def filt_high_pass(x,freq_high,sr):
    return signal.filtfilt(*signal.butter(4,freq_high*2/sr,'highpass'),x)

def get_five_freq_bands(idx):
    fl=[[1,3],[4,7],[8,13],[14,30],[31,50]]
    return fl[idx]

def get_differential_entropy(x):
    # x:[channel,length]
    var=np.var(x,axis=-1)
    return 0.5*np.log(2*np.pi*np.e*var)


def ASM(feature,ch_names):
    # feature:[64,]
    assert feature.shape[0] == len(ch_names)
    selected,L,R=get_LR_ch(ch_names,return_index=True)
    fl=feature[L]
    fr=feature[R]
    return (fl-fr)/(fl+fr)
    
def delete_ch(feature,ch_names,delete_ch_names):
    return np.stack([feature[i] for i,name in enumerate(ch_names) if name not in delete_ch_names],axis=0)
    

import re
def get_LR_ch(ch_names,return_index=True):
    selected=[]
    L=[]
    R=[]
    for i,name in enumerate(ch_names):
        digits = "".join(re.findall(r'\d', name))
        letters  = "".join(re.findall(r'[A-Z]', name))
        # print(letters)
        if digits != '':
            digits=int(digits)
            if digits %2==1:
                pair_digits=digits+1
                l=f'{letters}{digits}'
                r=f'{letters}{pair_digits}'
                if l not in selected:
                    L.append(l)
                    R.append(r)
                    selected.append(l)
                    selected.append(r)
    if return_index:
        return ch_name_to_index(selected,ch_names),ch_name_to_index(L,ch_names),ch_name_to_index(R,ch_names)
    return selected,L,R

def ch_name_to_index(chs,ch_names):
    return [ch_names.index(ch) for ch in chs]


class FeatureExtractor():
    def __init__(self,ch_names=None,delete_ch_names=None,sr=1000,slice_sec=5,feature_types=['activity','mobility','complexity','de','asm','fd'],band=None):
        if ch_names is None:
            ch_names=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
            
        if delete_ch_names is None:
            delete_ch_names=['M1','M2']
        self.ch_names=ch_names
        self.delete_ch_names=delete_ch_names
        if band is None:
            band=[[1,3],[4,7],[8,13],[14,30],[31,50],[51,100]]
        self.bands=band
        self.sr=sr
        self.slice_sec=slice_sec
        self.slice_length=slice_sec*sr
        self.feature_types=feature_types
        
    def extract_one_band(self,feature):
        # feature [channel,length]
        feat_all=[]
        if 'activity' in self.feature_types:
            feat_activity=activity(feature)
            feat_all.append(feat_activity)
        if 'mobility' in self.feature_types:
            feat_mobility=mobility(feature)
            feat_all.append(feat_mobility)
        if 'complexity' in self.feature_types:
            feat_complexity=complexity(feature)
            feat_all.append(feat_complexity)
        if 'de' in self.feature_types:
            feat_DE=get_differential_entropy(feature)
            feat_all.append(feat_DE)
        if 'asm' in self.feature_types:
            if 'feat_DE' not in locals():
                feat_DE=get_differential_entropy(feature)
            feat_ASM=ASM(feat_DE,self.ch_names)
            feat_all.append(feat_ASM)
        if len(feat_all) > 0:
            feat_cat = np.concatenate(feat_all, axis=0)
        else:
            feat_cat = np.array([])
        return feat_cat
    
    def normalize_signal(self,x):
        x=(x-x.mean(-1,keepdims=True))/x.var(-1,keepdims=True)
        return x
    
    def extract(self,feature):
        feat_all=[]
        for i,band in enumerate(self.bands):
            filted_feature=filt_band_pass(feature,*band,self.sr)
            feat_band=self.extract_one_band(filted_feature)
            feat_all.append(feat_band)
        if 'fd' in self.feature_types:
            feat_FD=measure_signal(feature)
            feat_all.append(feat_FD)
        feat_all=np.concatenate(feat_all,axis=0)
        return feat_all
    
    def slice_and_extract(self,long_signal):
        # delete channels
        long_signal=delete_ch(long_signal,self.ch_names,self.delete_ch_names)
        self.ch_names=[name for i,name in enumerate(ch_names) if name not in self.delete_ch_names]
        # long_signal:[channel,length]
        # return feats:[seg_num,feat_dim], signal_length:int
        # long_signal=self.normalize_signal(long_signal)
        num=int(long_signal.shape[1]/self.slice_length)
        feats=[]
        for i in range(num):
            slice_signal=long_signal[:,i*self.slice_length:i*self.slice_length+self.slice_length]
            feat=self.extract(slice_signal)
            feats.append(feat)
        feats=np.stack(feats,axis=0)
        return feats,i*self.slice_length+self.slice_length
