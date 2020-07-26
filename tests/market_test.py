from time import time

import numpy as np
import matplotlib.pyplot as plt

from context import portopt

np.random.seed(10)
dt = 1/365
T_horizon = 100
num_points = int(T_horizon//dt)
M = 1000
# M = 1
plot = True
start = time()
for i in range(M):

    model = portopt.ConstantMarketModel(dt, 0.1, 0.2, 0.02)

    market_simulator = portopt.MarketSimulator(model, dt)

    prices, _ = market_simulator.simulate(T_horizon)
delta = time() - start
print(f"simulate took {delta/M} sec on average")

# start = time()
# for i in range(M):
#     model = portopt.ConstantMarketModel(dt, 0.1, 0.2, 0.02)
#
#     market_simulator = portopt.MarketSimulator(model, dt)
#     prices = portopt.array(2)
#     prices.append_col([1, 1])
#     for j in range(num_points):
#         information, _ = market_simulator.step()
#         prices.append_col(information[:2])
#
#     prices = prices.as_array()
#
# delta = time() - start
# print(f"step took {delta/M} sec on average")
if plot:
    fig, ax = plt.subplots()
    x_axis = np.arange(num_points+1) + 1
    ax.plot(x_axis, prices[0], label="money market")
    ax.plot(x_axis, prices[1], label="stock")
    ax.legend()
    ax.grid()
    plt.show()
