"""
This script defines the strategist classes, which are the brains of the trader,
deciden what portion of the portfolio should be dedicated to the stock at each
time step.

As all classes, they have two modes of operation: 1) online with `step` and
2) batch with `compute_path`.
"""
import numpy as np

from .expand_array import ExpandArray


class ConstantStrategy(object):
    """A strategy that hold a fixed fraction of stock in the portfolio all the
    time.

    Parameters
    ----------
    stock_frac : float (Optional)
        The fraction of the portfolio invested in the stock.

    Attributes
    ----------
    stock_frac

    """
    def __init__(self, stock_frac=0.5):
        self.stock_frac = stock_frac

    def reset(self):
        return self.stock_frac

    def step(self, information, *args):
        return self.stock_frac

    def compute_path(self, information, *args):
        num_points = information.shape[1]
        return np.ones(num_points) * self.stock_frac

    def identify(self):
        return f"Constant strategy {self.stock_frac*100:.1f}% stock"


class MaxWealthStrategy(object):
    """A strategy that picks at every moment the best performing asset so far
    in terms of relative gain.

    Since by the model definition, both assets (stock and money market accout)
    start at 1$ per share, the best performing asset is determined by the more
    valued asset.

    The strategy starts with 100% of the portfolio invested in the stock.

    Parameters
    ----------

    Attributes
    ----------
    stock_frac : float
        The fraction of the portfolio invested in the stock.

    """
    def __init__(self):
        self.stock_frac = 1.0

    def reset(self):
        self.stock_frac = 1.0
        return self.stock_frac

    def step(self, information, *args):
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

    def compute_path(self, information, *args):
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

    def identify(self):
        return f"MaxWealth strategy"


class ConstantCRRAOracleStrategy(object):
    """The optimal strategy for constant markect and constant relative risk
    aversions.

    This is an 'Oracle' strategy thus it uses information that is not shown
    to the trader in real life as the stock volatility and mean rate of
    return. It is used mainly for debugging and comparisons.

    The strategy starts with 100% of the portfolio invested in the stock.

    Parameters
    ----------
    R : float
        The relativ risk aversion for the given utility. If u denotes the
        utility, then R = - (u''*x)/u'

    Attributes
    ----------
    stock_frac : float
        The fraction of the portfolio invested in the stock.
    R

    """
    def __init__(self, R):
        self.stock_frac = 1.0
        self.R = R

    def reset(self):
        self.stock_frac = 1.0
        return self.stock_frac

    def step(self, information, secret_information):
        """Update the stock fraction based on the last prices information.

        Parameters
        ----------
        information : list
            A list of: money market price, stock price, interest rate.
        secret_information : list
            A list of: stock mean rate of return, stock volatility.

        Returns
        -------
        float
            The fraction invested in the stock.

        """
        interest_rate = information[2]
        mean_rate, volatility = secret_information
        mpr = (mean_rate - interest_rate) / volatility
        self.stock_frac = mpr / (self.R * volatility)
        return self.stock_frac

    def compute_path(self, information, secret_information):
        """Compute the whole path for the stock fraction over the prices'
        paths.

        Parameters
        ----------
        information : ndarray
            A 3 by T/dt array where T is the horizon. First row corresponds to
            the money market price, second row to the stock price, and third
            row to the interest rate.
        secret_information : ndarray
            A 2 by T/dt array where T is the horizon. First row corresponds to
            the stock mean rate of return, second row to the stock volatility.

        Returns
        -------
        ndarray
            A 1D array of length T/dt specifying the fraction of the portfolio
            invested in the stock for each timestep.

        """
        mprs = (secret_information[0] - information[2][1:]) / secret_information[1]
        stock_fracs = np.hstack([[1.0], mprs / (secret_information[1] * self.R)])
        return stock_fracs

    def identify(self):
        return f"Opt. Oracle const. market CRRA={self.R}"
