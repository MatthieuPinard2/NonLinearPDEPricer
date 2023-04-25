import numpy as np
import scipy.linalg as lin
import scipy.interpolate as interp
import Payoff
import Underlying

class NonLinearPDESolver:
    def __init__(self, payoff : Payoff, underlying : Underlying,
                 nb_t_steps, nb_x_steps,
                 nb_std_down = -6.0, nb_std_up = 6.0,
                 nb_non_linear_iter = 50, nb_non_linear_tol = 1.0e-8):
        self.payoff = payoff
        self.underlying = underlying
        # Uniform time meshing (assuming today is t = 0)
        self.t_mesh, self.dT = np.linspace(payoff.getExpiry(), 0.0, nb_t_steps, endpoint=True, retstep=True)
        self.nb_t_steps = nb_t_steps
        # Uniform log-spot meshing.
        down_barrier, up_barrier = self.payoff.getContinousBarriers()
        x_min = np.log(underlying.getReferenceSpot()) + nb_std_down * underlying.getReferenceVol() * np.sqrt(payoff.getExpiry())
        x_max = np.log(underlying.getReferenceSpot()) + nb_std_up * underlying.getReferenceVol() * np.sqrt(payoff.getExpiry())
        # Special treatment of continuous barrier options : we want the domain to be bounded by the barriers,
        # and the barriers to be on the grid.
        if np.isfinite(down_barrier) and np.log(down_barrier) >= x_min:
            x_min = np.log(down_barrier)
        if np.isfinite(up_barrier) and np.log(up_barrier) <= x_max:
            x_max = np.log(up_barrier)
        self.x_mesh, self.dX = np.linspace(x_min, x_max, nb_x_steps, endpoint=True, retstep=True)
        self.nb_x_steps = nb_x_steps
        # Compute the spot mesh as well (with the same treatment for continuous barriers)
        self.s_mesh = np.exp(self.x_mesh)
        if np.isfinite(down_barrier) and np.log(down_barrier) == x_min:
            self.s_mesh[0] = down_barrier
        if np.isfinite(up_barrier) and np.log(up_barrier) == x_max:
            self.s_mesh[-1] = up_barrier
        # For the non-linear solver:
        self.nb_non_linear_iter = nb_non_linear_iter
        self.nb_non_linear_tol = nb_non_linear_tol
        
    def solvePDE(self):
        spot = self.underlying.getSpot()
        notional = self.payoff.getNotional()
        # For expired payoffs or breached continuous barriers, we simply return the payoff.
        continuous_barriers = self.payoff.getContinousBarriers()
        if self.dT >= 0.0 or not (continuous_barriers[0] < spot < continuous_barriers[1]):
            premium = self.payoff.getPayoff(spot)
            return float(premium)
        # The non-linear method requires a notional. 
        if self.underlying.isNonLinear() and (notional is None or notional == 0.0):
            raise Exception('Non Linear PDE: Expecting a non-null notional')
        # This function below computes the solution at time t - dt given the solution at time t, with a fully implicit scheme.
        # Some constants first.
        firstOrder = -0.5 * self.dT / self.dX
        secondOrder = -self.dT / (self.dX * self.dX)
        bandedMatrix = np.zeros((3, self.nb_x_steps))
        bandRange = np.arange(0, self.nb_x_steps - 1)
        def solveOneStep(vol, solution_before):    
            # solution_before is the solution at time t_{i}.
            halfVar = 0.5 * vol * vol
            bandedMatrix[2, bandRange] = halfVar[bandRange + 1] * (-firstOrder - secondOrder)
            bandedMatrix[1, :] = (1.0 + 2.0 * halfVar * secondOrder)
            bandedMatrix[0, bandRange + 1] = halfVar[bandRange] * ( firstOrder - secondOrder)
            solution_after = solution_before.copy()
            boundaries = self.payoff.getDirichletBoundaries(self.s_mesh, self.t_mesh[i + 1], solution_after)
            solution_after[0]  -= bandedMatrix[2, 0] * boundaries[0]
            solution_after[-1] -= bandedMatrix[0, -1] * boundaries[-1]
            lin.solve_banded((1, 1), bandedMatrix, solution_after, overwrite_ab=True, overwrite_b=True, check_finite=False)
            # Apply the constraint at time t_{i + 1}
            solution_after = self.payoff.getConstraint(self.s_mesh, self.t_mesh[i + 1], solution_after)
            return solution_after
        # Initialize the solution at expiry.
        solution = self.payoff.getPayoff(self.s_mesh)
        solution = self.payoff.getConstraint(self.s_mesh, self.t_mesh[0], solution)
        # Solving : Loop on time steps.
        for i in range(self.nb_t_steps - 1):
            vol = self.underlying.getVol()
            if self.underlying.isNonLinear():
                nlIter = 0
                # We are computing one step of the linear PDE with vol = (vol_bid + vol_ask) / 2
                # in order to have an initial guess of the optimized vol.
                initVol = np.full((self.nb_x_steps), 0.5 * (vol[0] + vol[1]))
                solution_prev_iter = solveOneStep(initVol, solution)
                # Non linear loop.
                while nlIter < self.nb_non_linear_iter:
                    gamma = np.zeros((self.nb_x_steps))
                    arange = np.arange(1, self.nb_x_steps - 1)
                    gamma.flat[arange] = (1.0 + 0.5 * self.dX) * solution_prev_iter[arange - 1] - 2.0 * solution_prev_iter[arange] + (1.0 - 0.5 * self.dX) * solution_prev_iter[arange + 1]
                    gamma /= self.dX * self.dX
                    # We are choosing the volatility that minimizes the Hamiltonian : \forall S, du/dt(S) + inf_Vol(L(Vol)(u)(S))
                    optimizedVol = vol[0] + (vol[1] - vol[0]) * 0.5 * (1.0 + np.sign(-gamma * notional))        
                    optimizedVol[0] = optimizedVol[1]
                    optimizedVol[-1] = optimizedVol[-2]
                    # This is really a fixed point algorithm.
                    solution_curr_iter = solveOneStep(optimizedVol, solution)
                    nlIter += 1
                    if lin.norm(solution_curr_iter - solution_prev_iter) / self.nb_x_steps <= self.nb_non_linear_tol:
                        break
                    solution_prev_iter = solution_curr_iter
                solution = solution_curr_iter
            else:
                # Linear case.
                vol = np.full((self.nb_x_steps), vol)
                solution = solveOneStep(vol, solution)
        # Return the premium computed by the PDE solver.
        cubic_spline = interp.CubicSpline(self.x_mesh, solution)
        premium = cubic_spline(np.log(spot))
        return premium
