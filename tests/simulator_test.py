import numpy as np

from context import portopt as pt

dt = 1/365
T = 10
market_model = pt.ConstantMarketModel(dt, 0.1, 0.2, 0.01)
market = pt.MarketSimulator(market_model, dt)

traders = []
traders.append(pt.Trader(pt.ConstantStrategy(), dt))
traders.append(pt.Trader(pt.MaxWealthStrategy(), dt))
traders.append(pt.Trader(pt.ConstantCRRAOracleStrategy(R=1), dt))


simulator = pt.Simulator(traders, market, np.log, log_dir=None)

simulator.simulate(T, 100000)
simulator.display_histograms()
