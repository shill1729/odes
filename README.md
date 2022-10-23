# odes
A python package for rk4 and euler solvers.

## Installation
```
python -m pip install git+https://github.com/shill1729/odes.git
```

## Euler scheme
The basic Euler scheme for the ODE
$$\frac{dx}{dt} = F(t,x)$$
with initial condition $x(0)=x_0$ is 
$$x_{t+h}=x_t+hF(t,x_t).$$

## rk4
The RK4 scheme is even better, but more complicated to write out.

## Gradient descent
Also implemented is gradient descent with inexact line search.
Here we maximize the function $f: \mathbb{R}^d \to \mathbb{R}$ by the
iterative scheme
$$x_i = x_{i-1} -\eta_i \nabla f(x_{i=1}),$$
where the step-size $\eta_i$ (called the learning-rate in machine learning circles)
is chosen via teh inexact line search criterion:
pick $\eta_i$ such that
$$f(x_i) \leq f(x_{i-1}) -\alpha \eta_i \|\nabla f(x_{i-1})\|^2,$$
where $\|\cdot \|$ is the Euclidean norm.


