import numpy as np
import matplotlib.pyplot as plt
import Payoff
import Greeks
from Underlying import Underlying
from NonLinearPDESolver import NonLinearPDESolver
from multiprocessing import Pool
from copy import deepcopy

def computeGreeksAtSpot(spot : float, payoff : Payoff, underlying : Underlying):
    udl = deepcopy(underlying)
    udl.setReferenceSpot(spot)
    udl.setSpot(spot)
    pde_solver = NonLinearPDESolver(deepcopy(payoff), udl,
                                    nb_t_steps = 253, nb_x_steps = 500)
    return Greeks.computeGreeks(pde_solver)

def displayGreeks(x_axis, y_axis, names):
    nb_graphs = len(names)
    figure, axis = plt.subplots(nb_graphs)
    figure.subplots_adjust(hspace=0.5)
    for i in range(nb_graphs):
        axis[i].plot(x_axis[i], y_axis[i])
        axis[i].set_title(names[i])
        axis[i].grid(True)
    plt.show()

def displayGreeksNonLinear(x_axis, y_axis, y_axis_lin, names):
    nb_graphs = len(names)
    figure, axis = plt.subplots(nb_graphs)
    figure.subplots_adjust(hspace=0.5)
    for i in range(nb_graphs):
        axis[i].plot(x_axis[i], y_axis[i], x_axis[i], y_axis_lin[i])
        axis[i].set_title(names[i])
        axis[i].grid(True)
    plt.show()

if __name__ == '__main__':
    # We are declaring here the market data.
    S0 = 4133.52
    vol = 0.1677
    vol_spread = 0.02
    vol_non_linear = (vol - vol_spread, vol + vol_spread)
    underlying = Underlying(spot = S0, vol = vol_non_linear)

    # Now we declare the payoff.
    payoff = Payoff.DownAndOutPut({
        "Notional": -1.0,
        "Expiry": 0.01,
        "Strike": 4200.0,
        "KOBarrier": 3900.0,
        "Rebate": 0.0
    })
    """payoff = Payoff.DoubleNoTouch({
        "Notional": -1.0,
        "Expiry": 1.0 / 12.0,
        "KOBarrier_Down": 0.80 * S0,
        "KOBarrier_Up": 1.20 * S0
    })"""

    # Let's now run the PDE engine by sliding the spots, 
    # and computing the risks (Premium, Delta, Gamma and Non-Linear Surprime) for every slide.
    nb_slides = 100
    min_spot, max_spot = 3800.0, 4500.0
    spots = np.linspace(min_spot, max_spot, nb_slides)
    greek_names = Greeks.getGreeksNames()
    nb_greeks = len(greek_names)
    greeks = np.zeros((nb_greeks, nb_slides))

    # Compute the Greeks with multi-threading.
    with Pool() as p:
        result = p.starmap(computeGreeksAtSpot, [(s, payoff, underlying) for s in spots])
        for i in range(nb_slides):
            greeks[:, i] = result[i]

    # Finally, we display the Greeks
    displayGreeks([spots] * nb_greeks, greeks[range(nb_greeks)], greek_names)

    # Compare Linear and Non-Linear Greeks now.
    if underlying.isNonLinear():
        greeks_linear = np.zeros((nb_greeks, nb_slides))
        vol = underlying.getVol()
        underlying.setVol(0.5 * (vol[0] + vol[1]))
        with Pool() as p:
            result = p.starmap(computeGreeksAtSpot, [(s, payoff, underlying) for s in spots])
            for i in range(nb_slides):
                greeks_linear[:, i] = result[i]
        displayGreeksNonLinear([spots] * nb_greeks, greeks[range(nb_greeks)], greeks_linear[range(nb_greeks)], greek_names)