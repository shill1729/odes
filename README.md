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


