from abc import ABC, abstractmethod
import numpy as np

# Helper functions for Continuous Barriers (inclusive)
def barrier(spot, barrier, is_down, is_out):
    perf = (spot - barrier) * (-1.0 if is_down else 1.0)
    ind = np.zeros(perf.shape)
    ind[perf >= 0.0] = 1.0
    return (1.0 - ind) if is_out else ind

# Base abstract class
class Payoff(ABC):
    def __init__(self, args):
        self.t_expiry = args["Expiry"]
        self.notional = args.get("Notional", None)
    # This function overrides the solution at every solving step:
    #   S(X, t) <- f(X, t, S(X, t))
    def getConstraint(self, S, t, solution):
        # By default, do nothing.
        return solution
    # This function sets the payoff at expiry.
    @abstractmethod
    def getPayoff(self, S):
        return NotImplemented
    # This function sets the boundaries at every solving step.
    @abstractmethod
    def getDirichletBoundaries(self, S, t):
        return NotImplemented
    # This function returns the Down/Up barrier as a pair. (+/- infty is no barrier)
    def getContinousBarriers(self):
        return (-np.inf, +np.inf)
    # Getters (not overriden)
    def getExpiry(self):
        return self.t_expiry
    def getNotional(self):
        return self.notional

# Implementation of a few payoffs.
class AmericanCall(Payoff):
    def __init__(self, args):
        self.strike = args["Strike"]
        super().__init__(args)
    def getConstraint(self, S, t, solution):
        exercise_value = np.maximum(S - self.strike, 0.0)
        return np.maximum(exercise_value, solution)
    def getPayoff(self, S):
        return np.maximum(S - self.strike, 0.0)
    def getDirichletBoundaries(self, S, t):
        return (0.0, np.maximum(S[-1] - self.strike, 0.0))

class UpAndOutCallSpread(Payoff):
    def __init__(self, args):
        self.strike = args["Strike"]
        self.upper_strike = args["UpperStrike"]
        self.ko_barrier = args["KOBarrier"]
        super().__init__(args)
    def getConstraint(self, S, t, solution):
        if S[-1] >= self.ko_barrier:
            solution[-1] = 0.0
        return solution
    def getPayoff(self, S):
        call = np.maximum(S - self.strike, 0.0)
        upper_call = np.maximum(S - self.upper_strike, 0.0)
        ko_indicator = barrier(S, self.ko_barrier, is_down=False, is_out=True)
        return (call - upper_call) * ko_indicator
    def getDirichletBoundaries(self, S, t):
        return (0.0, 0.0)
    def getContinousBarriers(self):
        return (-np.inf, self.ko_barrier)

class DownAndOutPut(Payoff):
    def __init__(self, args):
        self.strike = args["Strike"]
        self.ko_barrier = args["KOBarrier"]
        self.rebate = args["Rebate"]
        super().__init__(args)
    def getConstraint(self, S, t, solution):
        if S[0] <= self.ko_barrier:
            solution[0] = self.rebate
        return solution
    def getPayoff(self, S):
        put = np.maximum(self.strike - S, 0.0)
        ko_indicator = barrier(S, self.ko_barrier, is_down=True, is_out=True)
        return put * ko_indicator + self.rebate * (1.0 - ko_indicator)
    def getDirichletBoundaries(self, S, t):
        return (self.rebate, 0.0)
    def getContinousBarriers(self):
        return (self.ko_barrier, +np.inf)

class DoubleNoTouch(Payoff):
    def __init__(self, args):
        self.ko_barrier_down = args["KOBarrier_Down"]
        self.ko_barrier_up = args["KOBarrier_Up"]
        super().__init__(args)
    def getConstraint(self, S, t, solution):
        if S[0] <= self.ko_barrier_down:
            solution[0] = 0.0
        if S[-1] >= self.ko_barrier_up:
            solution[-1] = 0.0
        return solution
    def getPayoff(self, S):
        ko_indicator_down = barrier(S, self.ko_barrier_down, is_down=True, is_out=True)
        ko_indicator_up = barrier(S, self.ko_barrier_up, is_down=False, is_out=True)
        return ko_indicator_down * ko_indicator_up
    def getDirichletBoundaries(self, S, t):
        return (0.0, 0.0)
    def getContinousBarriers(self):
        return (self.ko_barrier_down, self.ko_barrier_up)