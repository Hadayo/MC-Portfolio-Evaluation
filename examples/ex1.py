import numpy as np

from context import portopt as pt

# A utility function
def identity(x):
    return x

# Parameters
save = True
if save:
    output_dir = "../report/figures/"
    log_dir = "../report/logs/"
else:
    output_dir = None
    log_dir = None

np.random.seed(0xF00D)

dt = 1 / 365
T = 1
N = 100000

# Define the market
market_model = pt.ConstantMarketModel(dt, 0.1, 0.2, 0.01)
market = pt.MarketSimulator(market_model, dt)

# Define the traders
traders = []
traders.append(pt.Trader(pt.ConstantStrategy(0.5), dt))
traders.append(pt.Trader(pt.ConstantStrategy(1), dt))
traders.append(pt.Trader(pt.ConstantStrategy(4), dt))
traders.append(pt.Trader(pt.BestSoFarStrategy(), dt))
traders.append(pt.Trader(pt.ConstantCRRAOracleStrategy(R=1), dt))

# define simulater
simulator = pt.Simulator(traders, market, np.log, log_dir=log_dir)

# simulate
simulator.simulate(T, N)
simulator.display_histograms(save, output_dir)
