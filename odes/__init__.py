import numpy as np


def euler_scheme(f, x0, tn, n=1000, t0=0.0):
    """ Numerically solve the ODE x'(t) = f(t, x(t)) where x: [a,b] -> R^d, using the Euler
    scheme for integration.

    Parameters:
        f: the RHS of the ODE, must be a function of (t,x) and be R^d-valued
        x0: numpy array of shape d, the initial value x0 in R^d
        tn: the endpoint of the time interval [t0, tn]
        n: number of time sub-intervals
        t0: optional starting point, defaults to 0.0
    """
    h = (tn - t0) / n
    tt = np.linspace(t0, tn, n + 1)
    d = x0.shape[0]
    x = np.zeros((n + 1, d))
    x[0, :] = x0
    for i in range(n):
        x[i + 1, :] = x[i, :] + h * f(tt[i], x[i, :])
    return x


def rk4(f, x0, tn, n=1000, t0=0.0):
    """ Numerically solve the ODE x'(t) = f(t, x(t)) where x: [a,b] -> R^d, using the Runge-Kutta four
    scheme for integration. Note this can also be used to solve second order DEs as follows. The 2nd
    order ODE r''(t) + a r'(t)+b r(t) = h(t) can be re-written as the first order system in position-velocity
    space (r,v) where
    r'(t) = v
    v'(t) = h(t)- a v(t)-b r(t)
    with initial conditions x(0)=(r(0), v(0))^T

    Parameters:
        f: the RHS of the ODE, must be a function of (t,x) and be R^d-valued
        x0: numpy array of shape d, the initial value x0 in R^d
        tn: the endpoint of the time interval [t0, tn]
        n: number of time sub-intervals
        t0: optional starting point, defaults to 0.0
    """
    h = (tn - t0) / n
    tt = np.linspace(t0, tn, n + 1)
    d = x0.shape[0]
    x = np.zeros((n + 1, d))
    x[0, :] = x0
    for i in range(n):
        k1 = f(tt[i], x[i, :])
        k2 = f(tt[i] + 0.5 * h, x[i, :] + 0.5 * h * k1)
        k3 = f(tt[i] + 0.5 * h, x[i, :] + 0.5 * h * k2)
        k4 = f(tt[i] + h, x[i, :] + h * k3)
        x[i + 1, :] = x[i, :] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
