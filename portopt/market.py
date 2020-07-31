"""
Defines the MarketSimulator class and some MaketModels like the constant
market model.
"""
import numpy as np

from .expand_array import ExpandArray


class ConstantMarketModel(object):
    """Defines the underlying variables of a constant market.

    The class has two modes of operation: online or batch.
    1. online mode:
        Works through the `step` function. Each call to the function updates
        the tracked parameters and returns them to the caller.
    2. batch mode:
        Works with the function `sample_path` which receives the time horizon in
        advance. One call to the function returns all the tracked parameters
        and their paths over time.
    Working in batch mode increases performance by 2 orders of magnitude.

    Parameters
    ----------
    dt : float
        The time difference between each two points (in years).
    stock_mean : float
        The stock mean rate of return (in 1/years).
    stock_volatility : float
        The stock volatility (in 1/sqrt(years)).
    ir : float
        The money market account interest rate.

    Attributes
    ----------
    num_params : int
        The number of parameters the model tracks.
    dt
    stock_mean
    stock_volatility
    ir

    """
    def __init__(self, dt, stock_mean, stock_volatility, ir):
        self.dt = dt
        self.stock_mean = stock_mean
        self.stock_volatility = stock_volatility
        self.ir = ir
        self.num_params = 3

    def reset(self):
        pass  # here for consistency reasons

    def step(self):
        return self.stock_mean, self.stock_volatility, self.ir

    def sample_path(self, T_horizon):
        """Simulate the parameters over a path with length T_horizon.
        Since the market model describes a constant market, this equals
        duplicating the inital values through time.

        Parameters
        ----------
        T_horizon : float
            The time horizon (in years).

        Returns
        -------
        ndarray
            A [num_params x (T/dt - 1)] matrix where each row represents a path for
            one of the tracked parameters.

        """
        num_points = int(T_horizon / self.dt) - 1
        params = np.array([self.stock_mean, self.stock_volatility, self.ir]).reshape(-1, 1)
        return np.ones((self.num_params, num_points)) * params


class MarketSimulator(object):
    """The MarketSimulator can simulates a market with a stock and a money
    market account under a given market model.

    The stock prices is progressed according to the Generalized Geometric
    Brownian Motion SDE:
        dS_t = S_t * (mu_t*dt + sigma_t*dW_t)
    where:
        S - the price of the stock
        mu - the instantaneous mean rate of return
        sigma - instantaneous volatility
        W - a Brownian motion

    The money market price is progressed according to the continous compounding
    formula:
        dM_t = M_t*r_t*dt
    where:
        M - the money market price
        r - the instantaneous interest rate

    The class has two modes of operation: online or batch.
    1. online mode:
        Works with the `step` function. Each call to the function updates
        the stock prices and returns them and the interest rate to the caller.
    2. batch mode:
        Works with the function `sample_path` which receives the time horizon in
        advance. One call to the function returns all the stock prices and
        their paths over time.
    Working in batch mode increases performance by 2 orders of magnitude.

    Parameters
    ----------
    market_model : MarketModel
        The model that describes the dynamics of the mean rate of return
        for the stock, the volatility of the stock and the interest rate.
    dt : float
        The time difference between two points (in years).

    Attributes
    ----------
    num_assets : int
        The number of assets simulated.
    prices : ExpandArray
        An expandable array for tracking the prices over time.
    dt
    market_model

    """
    def __init__(self, market_model, dt):
        self.dt = dt
        self.num_assets = 2  # including the money market account
        self.market_model = market_model

        self.reset()

    def reset(self):
        self.market_model.reset()
        self.prices = ExpandArray(self.num_assets)
        self.prices.append_col([1, 1])

    def step(self):
        """Performs one step of updates to the stock price and money market
        accound price.

        The money market accound follows a simple compound
        interest dynamics. The stock follows a generalized geometric Brownian
        motion with the parameters supplied by the market model.

        Returns
        -------
        information : list
            The current stock price, money market account price, and interest
            rate. This information is assumed to be available to the trader.
        secret_information : list
            The current stock volatility and stock mean rate of return. This
            information is used for debugging and is not assumed to be
            available to the trader.

        """
        stock_mean, stock_volatility, ir = self.market_model.step()
        money_market_price, stock_price = self.prices.last_col()

        delta_stock = stock_price * (stock_mean * self.dt
                                     + stock_volatility * np.sqrt(self.dt) * np.random.randn())
        new_stock_price = stock_price + delta_stock
        new_money_market_price = money_market_price * (1 + ir * self.dt)

        information = [new_money_market_price, new_stock_price, ir]
        secret_information = [stock_mean, stock_volatility]
        self.prices.append_col([new_money_market_price, new_stock_price])

        return information, secret_information

    def sample_path(self, T_horizon):
        """Computs a path for the prices of the stock and the money market
        account. This is the batch version of the `step` function that runs
        faster.

        Parameters
        ----------
        T_horizon : float
            The time horizon to be simulated (in years).

        Returns
        -------
        information : ndarray
            A 3 by T_horizon/dt array containing the prices of the stock,
            the money market account and the interest rates for the given
            horizon.
        secret_information : ndarray
            A 2 by T_horizon/dt array containing the market parameters path
            over the given horizon (stock mean and volatility).

        """
        num_points = int(T_horizon / self.dt) - 1
        market_params = self.market_model.sample_path(T_horizon)
        stock_means = market_params[0]
        stock_volatilities = market_params[1]
        irs = market_params[2]

        brownian_motion = np.random.randn(num_points)
        stock_multipliers = (1 + stock_means * self.dt
                             + stock_volatilities * np.sqrt(self.dt) * brownian_motion)
        stock_prices = np.hstack(([1], np.cumprod(stock_multipliers)))

        money_market_prices = np.hstack(([1], np.exp(self.dt * np.cumsum(irs))))

        irs_long = np.insert(irs, 0, irs[0])
        information = np.vstack([money_market_prices, stock_prices, irs_long])
        secret_information = market_params[:2]
        return information, secret_information

    def get_prices(self):
        return self.prices.as_array()
