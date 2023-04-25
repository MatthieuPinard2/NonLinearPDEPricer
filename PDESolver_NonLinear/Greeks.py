import NonLinearPDESolver

def computeGreeks(pde_solver     : NonLinearPDESolver,
                  bump_for_delta : float = 0.01):
    underlying = pde_solver.underlying
    # Preserve previous state.
    spot = underlying.getSpot()
    # Compute the states.
    spot_bump = bump_for_delta * spot
    premium = pde_solver.solvePDE()
    underlying.setSpot(spot - spot_bump)
    premium_minus = pde_solver.solvePDE()
    underlying.setSpot(spot + spot_bump)
    premium_plus = pde_solver.solvePDE()
    underlying.setSpot(spot)
    if underlying.isNonLinear():
        vol = underlying.getVol()
        underlying.setVol(0.5 * (vol[0] + vol[1]))
        premium_linear = pde_solver.solvePDE()
        underlying.setVol(vol)
    else:
        premium_linear = premium
    # Compute the Greeks by finite differences. (a.k.a bump-and-reprice)
    delta = 100.0 * (premium_plus - premium_minus) / (2.0 * spot_bump)
    gamma = 100.0 * (premium_plus - 2.0 * premium + premium_minus) / (spot_bump * spot_bump)
    surprime = premium - premium_linear
    # Return the Greeks (the order is important).
    return premium, delta, gamma, surprime

def getGreeksNames():
    return ["Premium", "Delta", "Gamma", "Surprime"]