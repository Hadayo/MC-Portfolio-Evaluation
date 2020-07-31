from os.path import join
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .expand_array import ExpandArray

plt.rc('font', family='serif')


def listify(x):
    """If x is a list, nothing is done, else create a one element list out
    of it.

    Parameters
    ----------
    x : type
        can be anything.

    Returns
    -------
    list
        either x is a list or a list containing x.

    """
    return x if isinstance(x, list) else [x]


def pprint(x):
    """Some helper function for printing to the user.

    Parameters
    ----------
    x : str
        The string to be printed.

    Returns
    -------

    """
    num = 10
    print()
    print('#' * num + ' ' + x.upper() + ' ' + '#' * num)
    print()


def to_ordinal(n):
    """A helper function to convernt integers to ordinal numbers, e.g.,
    1 -> 1st, 22 -> 22nd, etc.

    Parameters
    ----------
    n : int
        The number to be converted.

    Returns
    -------
    str
        The ordinal representation.

    """
    # credit : https://codegolf.stackexchange.com/questions/4707/outputting-ordinal-numbers-1st-2nd-3rd#answer-4712
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


class Simulator(object):
    """The simulator makes Monte Carlo simulations in order to compare
    the performance of different strategies under the same market conditions.
    The performance measure is the expected utility of the final portfolio
    value. The expectation is approximated with an empirical average.

    Parameters
    ----------
    traders : list
        A list of Trader objects with their different strategies.
    market : MarketSimulator
        The market simulator used to sample prices.
    utility : function
        A numpy like function that can operate on ndarrays.
    log_dir : str
        The directory to save the results. If None then the results are not
        saved.

    Attributes
    ----------
    final_portfolio_values : ExpandArray
        An expandable array where each row represents a trader and each column
        represents a sample of the final portfolio values for the traders.
    traders
    market
    utility
    log_dir

    """
    def __init__(self, traders, market, utility, log_dir='logs'):
        self.traders = listify(traders)
        self.market = market
        self.utility = utility
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        self.reset()

    def reset(self):
        for trader in self.traders:
            trader.reset()
        self.market.reset()
        self.final_portfolio_values = ExpandArray(len(self.traders))

    def sample(self, T_horizon):
        """Sample final portfolio values for all traders.

        Parameters
        ----------
        T_horizon : float
            The horizon (in years).

        Returns
        -------
        list
            The final portfolio values for each trader in this sample.

        """
        information, secret_information = self.market.sample_path(T_horizon)
        final_portfolio_values = []
        for trader in self.traders:
            _, portfolio_values = trader.compute_path(information, secret_information)
            final_portfolio_values.append(portfolio_values[-1])
        return final_portfolio_values

    def simulate(self, T_horizon, carlo_iterations):
        """Sample many final portfolio value for all the traders over the given
        horizon.

        The function ends by calling the `print_results` function.

        Parameters
        ----------
        T_horizon : float
            The horizon (in years).
        carlo_iterations : int
            The number of iterations to be done.

        Returns
        -------

        """
        pprint(f"begining simulation with {carlo_iterations:,} iterations")
        for _ in tqdm(range(carlo_iterations)):
            self.final_portfolio_values.append_col(self.sample(T_horizon))
        print("\nFinished running simulation.")

        portfolio_utilities = self.utility(self.final_portfolio_values.as_array())
        self.print_results(portfolio_utilities)

    def print_results(self, portfolio_utilities):
        """Prints the results to the user and writes them to a log file, if
        specified.

        Parameters
        ----------
        portfolio_utilities : ndarray
            Each row represents a trader, each column represents the final
            portfolio values for this sample.


        """
        pprint("printing results")
        N = portfolio_utilities.shape[1]
        means = portfolio_utilities.mean(axis=1)  # mean per trader
        stds = portfolio_utilities.std(axis=1) / np.sqrt(N)  # std per trader
        relative_errors = stds / means  # relative error per trader

        log_parts = []
        text = f"Using {self.utility.__name__} utility\n"
        log_parts.append(text)
        print(text)
        for i, trader in enumerate(self.traders):
            text = f"""
            {to_ordinal(i+1)} {trader.identify()}
            mean : {means[i]:.4f}
            std : {stds[i]:.4f}
            relative_errors : {relative_errors[i]:.4f}
            """
            log_parts.append(text)
            print(text)

        if self.log_dir is not None:
            # write to log file
            date = datetime.now().strftime("%d%m%y_%H-%M")
            filename = join(self.log_dir, date + ".txt")
            with open(filename, "w") as f:
                f.write("".join(log_parts))

    def display_histograms(self, save=False, output_dir=None):
        """Displays a pair plot to the user

        """
        err_msg = "Simulation must be run before results can be displayed."
        assert not self.final_portfolio_values.is_empty, err_msg
        err_msg = "If you want to save, you have to specify output_dir"
        assert (save and output_dir is not None) or not save, err_msg

        portfolio_utilities = self.utility(self.final_portfolio_values.as_array())
        N = portfolio_utilities.shape[1]
        names = [trader.identify() for trader in self.traders]
        dataframe = pd.DataFrame(portfolio_utilities.T, columns=names)
        sns.pairplot(dataframe, corner=True, diag_kws={"bins": int(np.sqrt(N))})
        if save:
            name = f"pairplot_u={self.utility.__name__}.jpg"
            plt.savefig(join(output_dir, name), bbox_inches="tight", dpi=200)

        plt.show()
