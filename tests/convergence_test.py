"""
This script is used to check the convergence rate of the geometric Brownian
motion estimation.

The results: there is a huge error from time to time in the estimation of the
the mean rate of return. Thus in this project we decided not to tackle the
problem of estimating the brownian motion parameters."""

import numpy as np

from context import portopt as pt

dt = 1 / (365)
mu = 0.1
sigma = 0.2
market_model = pt.ConstantMarketModel(dt, mu, sigma, 0.01)
market_simulator = pt.MarketSimulator(market_model, dt)

alpha = 0.5

RE_mu = np.inf
RE_sigma = np.inf
i_mu = 0
i_sigma = 0
i = 0

mu_hat = 0
sigma_hat = 0

stock_price = [1.0]

while RE_mu > 0.05 or RE_sigma > 0.05:
    information, _ = market_simulator.step()
    stock_price.append(information[1])

    relative_gains = np.diff(stock_price) / np.array(stock_price)[:-1]

    mu_hat = alpha*mu_hat + (1-alpha)*np.mean(relative_gains) / dt
    sigma_hat = alpha*sigma_hat + (1-alpha)*np.std(relative_gains) / np.sqrt(dt)

    RE_mu = np.abs(mu_hat - mu) / mu
    RE_sigma = np.abs(sigma_hat - sigma) / sigma

    if RE_mu > 0.01:
        i_mu += 1
    if RE_sigma > 0.01:
        i_sigma += 1

    print(RE_mu)
    print(RE_sigma)
    print("\n")

print(f"i_mu = {i_mu}")
print(f"i_sigma = {i_sigma}")
