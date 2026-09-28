import numpy as np

g = 9.81
v0 = 25.0
theta = np.radians(40.0)
T = 1.0

vx0 = v0*np.cos(theta)
vy0 = v0*np.sin(theta)

def F(u):
    x, y, vx, vy = u
    return np.array([vx, vy, 0.0, -g])

def euler(u0, h, T):
    n = int(T/h)
    u = u0.copy()
    for _ in range(n):
        u = u + h*F(u)
    return u

def rk2(u0, h, T):
    n = int(T/h)
    u = u0.copy()
    for _ in range(n):
        k1 = F(u)
        k2 = F(u + h*k1)
        u = u + (h/2.0)*(k1 + k2)
    return u

def rk4(u0, h, T):
    n = int(T/h)
    u = u0.copy()
    for _ in range(n):
        k1 = F(u)
        k2 = F(u + 0.5*h*k1)
        k3 = F(u + 0.5*h*k2)
        k4 = F(u + h*k3)
        u = u + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return u

u0 = np.array([0.0, 0.0, vx0, vy0])
r_exacta = np.array([vx0*T, vy0*T - 0.5*g*T**2])

for h in [0.10, 0.05]:
    u_e = euler(u0, h, T)
    u_r2 = rk2(u0, h, T)
    u_r4 = rk4(u0, h, T)
    E_e = np.linalg.norm(u_e[:2] - r_exacta)
    E_r2 = np.linalg.norm(u_r2[:2] - r_exacta)
    E_r4 = np.linalg.norm(u_r4[:2] - r_exacta)
    print(f"h={h:.2f}  Euler={E_e:.3e}  RK2={E_r2:.3e}  RK4={E_r4:.3e}")