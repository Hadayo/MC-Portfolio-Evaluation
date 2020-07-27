import matplotlib.pyplot as plt
import numpy as np

from context import portopt as pt

dt = 1/365
strategist = pt.MaxWealthStrategy(dt)
trader = pt.Trader(strategist, dt)

market_model = pt.ConstantMarketModel(dt, 0.1, 0.2, 0.01)
market_simulator = pt.MarketSimulator(market_model, dt)

T = 1
plot = True

# online mode
# num_points = int(T/dt)-1
#
# for t in range(num_points):
#     information, _ = market_simulator.step()
#     positions, port_val = trader.step(information)
#
#     # debug
#     # money_market_price, stock_price, _ = information
#     # print("money_pos * money_price + stock_pos * stock_price = port val")
#     # print(f"{positions[0]:.3f}*{money_market_price:.3f} + {positions[1]:.3f}*{stock_price:.3f} = {port_val:.3f}")
#     # print(money_market_price*positions[0]+stock_price*positions[1] == port_val)
#     # print()
#
# information = market_simulator.get_prices()
# port_vals = trader.get_portfolio_values()
# positions = trader.get_positions()

# batch mode
# np.random.seed(10)
market_simulator.reset()
trader.reset()
information, _ = market_simulator.simulate(T)
positions, port_vals = trader.simulate(information)    

# if np.allclose(port_vals, port_vals2):
#     print("portfolio values are equal in both methods")
#
# if np.allclose(positions, positions2):
#     print("Positions are equal in both methods")

if plot:
    time_axis = np.arange(len(information[0])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].plot(time_axis, information[0], label="money market")
    axes[0].plot(time_axis, information[1], label="stock")
    axes[0].plot(time_axis, port_vals, label="portfolio")
    axes[0].grid()
    axes[0].set_title("Values")
    axes[0].legend()

    axes[1].plot(time_axis, positions[0], label="money market")
    axes[1].plot(time_axis, positions[1], label="stock")
    axes[1].set_title("positions")
    axes[1].grid()
    axes[1].legend()
    plt.show()
