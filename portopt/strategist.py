"""
This script defines the strategist classes, which are the brains of the trader,
deciden what portion of the portfolio should be dedicated to the stock at each
time step.

As all classes, they have two modes of operation: 1) online with `step` and
2) batch with `simulate`.
"""
import numpy as np


class ConstantStrategy(object):
    """A strategy that hold a fixed fraction of stock in the portfolio all the
    time.

    Parameters
    ----------
    dt : float
        The time difference between each two points (in years).
    stock_frac : float
        The fraction of the portfolio invested in the stock.

    Attributes
    ----------
    dt
    stock_frac

    """
    def __init__(self, dt, stock_frac=0.5):
        self.dt = dt
        self.stock_frac = stock_frac

    def reset(self):
        return self.stock_frac

    def step(self, information):
        return self.stock_frac

    def simulate(self, information):
        num_points = information.shape[1]
        return np.ones(num_points)*self.stock_frac


class MaxWealthStrategy(object):
    """A strategy that picks at every moment the best performing asset so far
    in terms of relative gain.

    Since by the model definition, both assets (stock and money market accout)
    start at 1$ per share, the best performing asset is determined by the more
    valued asset.

    The strategy starts with 100% of the portfolio invested in the stock.

    Parameters
    ----------
    dt : float
        The time difference between each two points (in years).

    Attributes
    ----------
    stock_frac : float
        The fraction of the portfolio invested in the stock.
    dt

    """
    def __init__(self, dt):
        self.dt = dt
        self.stock_frac = 1.0

    def reset(self):
        self.stock_frac = 1.0
        return self.stock_frac

    def step(self, information):
        """Update the stock fraction based on the last prices information.

        Parameters
        ----------
        information : list
            A list of: 1) money market price, stock price, interest rate.

        Returns
        -------
        float
            The fraction invested in the stock.

        """
        self.stock_frac = 1.0 if information[1] > information[0] else 0.0
        return self.stock_frac

    def simulate(self, information):
        """Compute the whole path for the stock fraction over the prices'
        paths.

        Parameters
        ----------
        information : ndarray
            A 3 by T/dt array where T is the horizon. First row corresponds to
            the money market price, second row to the stock price, and third
            row to the interest rate.

        Returns
        -------
        ndarray
            A 1D array of length T/dt specifying the fraction of the portfolio
            invested in the stock for each timestep.

        """
        stock_fracs = np.hstack([[1.0], np.greater_equal(information[1][:-1], information[0][:-1], dtype=np.float)])
        return stock_fracs
