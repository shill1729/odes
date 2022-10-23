# Python implementation of Gradient descent, entirely relying on numpy alone
# This is probably not the most efficient implementation
# This algorithm follows the presentation given by Professor Yunlong Feng in
# AMAT 590 Non-linear optimization
import numpy as np


def finite_diff_grad(f, x, h=10 ** -5):
    """ Finite difference approximation to gradient of a function f: R^n to R.
    """
    n = x.shape[0]
    del_f = np.zeros(n)
    for i in range(n):
        eps = np.zeros(n)
        eps[i] = 1
        u = x + h * eps
        d = x - h * eps
        del_f[i] = (f(u) - f(d)) / (2 * h)
    return del_f


def inexact_line_search(f, x_i, gradf, alpha=0.5, beta=0.5):
    """ Inexact-line-search for computing efficient yet suboptimal stepsizes for
    gradient descent.

    Keyword arguments:
    f       -- real-valued function on R^n, convex and Lipschitz continuous
    x_i     -- np.array of dimension n, the current estimate
    gradf   -- the gradient of the function f
    alpha   -- float, the parameters for the inexact-line-search, in between 0  and 1
    beta    -- float, the parameters for the inexact-line-search, in between 0  and 1

    """
    eta = 1
    while f(x_i - eta * gradf(x_i)) > f(x_i) - alpha * eta * np.linalg.norm(gradf(x_i)) ** 2:
        eta = beta * eta
    return eta


# TODO: 1. consider returning a boolean result for convergence
#       2. add optional arguments to pass to f(x,...)
def gradient_descent(f, x0, gradf=None, N=100, epsilon=10 ** -9, alpha=0.5, beta=0.5):
    """ Solve an unconstrained minimization problem using the gradient descent method.

    Keyword arguments:
    -----------------
    f       -- real-valued function on R^n, convex and Lipschitz continuous
    x0      -- np.array of dimension n, the initial guess
    gradf   -- the gradient of the function f, if not passed numerical estimates are used
    N       -- int, the max number of iterations to compute of the descent
    epsilon -- error threshold for the minimum value of 'f'
    alpha   -- float, the parameters for the inexact-line-search, in between 0  and 1
    beta    -- float, the parameters for the inexact-line-search, in between 0  and 1

    Returns:
    The optimal input x, an array of n dimensions

    """

    x = [x0]
    i = 1

    if gradf is None:
        # Brute force idea:
        # Loop through the number of dimensions of x and compute
        # finite difference approximations to partial derivative
        gradf = lambda x: finite_diff_grad(f, x, h=epsilon)

    current_error = 1 + epsilon
    while i <= N and np.abs(current_error) > epsilon:
        eta_i = inexact_line_search(f, x[i - 1], gradf, alpha, beta)
        xi = x[i - 1] - eta_i * gradf(x[i - 1])
        x.append(xi)
        # If new point is not too small, use relative error
        if np.abs(f(x[i])) > epsilon / 2:
            current_error = np.abs(f(x[i]) - f(x[i - 1])) / np.abs(f(x[i]))
        else:  # otherwise use absolute error
            current_error = np.abs(f(x[i]) - f(x[i - 1]))
        i += 1
    if i > N:
        print("Max iterations hit")
    else:
        print("Within error tolerance")
    return x[i - 1]
