import numpy as np

from .expand_array import ExpandArray


class Trader(object):
    """The trader sees the market values and uses a strategy to determine its
    portfolio. Trader starts with one dollar and cand trade fractions of the
    stocks.

    The trader progresses the portfolio value (Xt) according to the SDE:
        dX_t = phi_t*dS_t + (X_t - phi_t*S_t)*r_t*dt
    where:
        phi - the number of shares of the stock.
        S - the price of the stock.
        r - the interest rate.

    Parameters
    ----------
    strategist : Strategy Class
        The trading strategy used by the trader.
    dt : float
        The time difference between two points (in years).

    Attributes
    ----------
    last_stock_price : float
        The last recorded price of the stock
    positions : ExpandArray
        A 2 rows array where each row represents the number of shares in the
        asset for every time instant. First row is for the money market account
        and the second row is for the stock.
    portfolio_val : ExpandArray
        A 1 row array containing the portfolio value for each time instant.
    strategist
    dt

    """
    def __init__(self, strategist, dt):
        self.strategist = strategist
        self.dt = dt
        self.reset()

    def reset(self):
        self.last_stock_price = 1
        self.positions = ExpandArray(2)  # one for mon. market, one stock
        self.portfolio_val = ExpandArray(1)

        self.portfolio_val.append_col([1])  # start with one dollar
        init_stock_fraction = self.strategist.reset()
        init_stock_position = self.frac2position(init_stock_fraction, 1)
        init_money_market_position = self.frac2position(1-init_stock_fraction, 1)
        init_positions = [init_money_market_position, init_stock_position]
        self.positions.append_col(init_positions)

    def step(self, information, *args):
        """Updates the portfolio value and positions according to the latest
        prices and strategy.

        Parameters
        ----------
        information : list
            The current money maker price, stock price and interest rate.

        Returns
        -------
        positions : list
            The amount of shares in the money market acount, and the amount of
            shares in the stock.

        """
        money_market_price, stock_price, ir = information
        stock_pos = self.positions.last_col()[1]
        port_val = self.portfolio_val.last_col()

        # update portfolio value
        delta_port_val = (stock_pos*(stock_price-self.last_stock_price)
                          + ir*(port_val - stock_pos*self.last_stock_price)*self.dt)

        new_port_val = port_val+delta_port_val
        self.portfolio_val.append_col([new_port_val])

        # update positions
        stock_fraction = self.strategist.step(information, *args)
        stock_position = self.frac2position(stock_fraction, stock_price)
        money_market_position = self.frac2position(1-stock_fraction,
                                                   money_market_price)
        positions = [money_market_position, stock_position]
        self.positions.append_col(positions)
        self.last_stock_price = stock_price
        return positions, new_port_val

    def compute_path(self, information, *args):
        """Computes the portfolio value and positions over the paths described
        in `information`.

        Parameters
        ----------
        information : ndarray
            A 3 by T/dt array describing the money market price, stock price
            and interest rate.

        Returns
        -------
        positions : ndarray
            A 2 by T/dt array describing the amount of shares in the money
            market account and the stock over time.
        portfolio_values : ndarray
        A 1D array describing the portfolio value over time.

        """
        past_stock_prices = information[1][:-1]
        current_stock_prices = information[1][1:]
        irs = information[2][1:]
        stock_fractions = self.strategist.compute_path(information, *args)

        stock_relative_gains = (current_stock_prices-past_stock_prices)/past_stock_prices
        aux_arr = (1 + stock_fractions[1:]*stock_relative_gains
                   + (1-stock_fractions[1:])*irs*self.dt)
        portfolio_values = np.hstack([[1], np.cumprod(aux_arr)])

        stock_poisitions = stock_fractions*portfolio_values/information[1]
        money_market_positions = (1-stock_fractions)*portfolio_values/information[0]
        positions = np.vstack([money_market_positions, stock_poisitions])
        return positions, portfolio_values

    def frac2position(self, fraction, stock_price):
        """Computes the amount of shares needed such that the fraction of the
        stock in the portfolio will equal `fraction`.

        Parameters
        ----------
        fraction : type
            Description of parameter `fraction`.
        stock_price : type
            Description of parameter `stock_price`.

        Returns
        -------
        type
            Description of returned object.

        """
        return fraction * self.portfolio_val.last_col() / stock_price

    def get_positions(self):
        return self.positions.as_array()

    def get_portfolio_values(self):
        return self.portfolio_val.as_array().reshape(-1)

    def identify(self):
        return self.strategist.identify()
