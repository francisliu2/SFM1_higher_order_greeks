import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from _plotly_future_ import v4_subplots
from plotly.subplots import make_subplots
import plotly.plotly as py
import pickle
from random import gauss
import seaborn as sns


# Calculate Variables of a Call Option
def calculate_delta(S, K, r, d, Tau, sig):
    Z = (np.log(S / K) + (r - d + sig ** 2 / 2) * Tau) / (sig * np.sqrt(Tau))
    return np.exp(-d * Tau) * norm.cdf(Z)


def calculate_gamma(S, K, r, d, Tau, sig):
    Z = (np.log(S / K) + (r - d + sig ** 2 / 2) * Tau) / (sig * np.sqrt(Tau))
    return np.exp(-d * Tau) * norm.pdf(Z) / (S * sig * np.sqrt(Tau))


def calculate_charm(S, K, r, d, Tau, sig):
    Z = (np.log(S / K) + (r - d + sig ** 2 / 2) * Tau) / (sig * np.sqrt(Tau))
    d2 = Z - sig * np.sqrt(Tau)
    A = d * np.exp(-d * Tau) * norm.cdf(Z)
    B = np.exp(-d * Tau) * norm.pdf(Z) * ((2 * (r - d) * Tau) - d2 * sig * np.sqrt(Tau)) / (
                S * Tau * sig * np.sqrt(Tau))
    return A - B

def calculate_call_price(S, K, r, d, Tau, sig):
    d1 = (np.log(S / K) + (r - d + (sig ** 2) / 2) * Tau) / (sig * np.sqrt(Tau))
    d2 = d1 - sig * np.sqrt(Tau)
    price = S * np.exp(-(r - d) * Tau) * norm.cdf(d1) - K * np.exp(-r * Tau) * norm.cdf(d2)
    price[price<0] = 0
    return price


# Generate Stock Price
def generate_asset_price(S0, sig, r, dt, gauss = gauss(0, 1.0)):
    return S0 * np.exp((r - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * gauss)

class Stock():
    def __init__(self, n_sample=1000, n_step=100, S0=36, sig=0.02, r=0.06, d=0, dt=0.01):
        self.n_sample = n_sample
        self.n_step = n_step
        self.stock_price = np.zeros(shape=(n_sample, n_step))
        self.sig = sig
        self.r = r
        self.d = d
        self.dt = dt
        self.T = dt * n_step
        random = np.random.normal(0,1,size=(n_sample, n_step))

        # Start with S0
        s=S0
        self.stock_price[:,0] = s
        for i in range(1, n_step):
            s = self.stock_price[:,i-1]
            rand = random[:,i-1]
            self.stock_price[:,i] = generate_asset_price(S0=s, sig=sig, r=r, dt=dt, gauss = rand)

    def plot(self, verbose=0):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 3))
        for row in range(self.stock_price.shape[0]):
            axs.plot(self.stock_price[row])
        axs.set_title("Stock Price")
        plt.tight_layout()
        if verbose == 1:
            print('sigma:', self.sig, 'r:', self.r, 'dt', self.dt,
                  'total steps:', self.T)