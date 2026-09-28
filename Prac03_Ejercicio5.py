import numpy as np
import matplotlib.pyplot as plt

GM = 3.986e14
r0 = 7.0e6
v0 = np.sqrt(GM/r0)
T_orb = 2*np.pi*np.sqrt(r0**3/GM)

def aceleracion(x, y):
    r = np.sqrt(x*x + y*y)
    factor = -GM/(r**3)
    return factor*x, factor*y

def euler_cromer(h, T):
    n = int(T/h)
    x = np.zeros(n); y = np.zeros(n)
    vx = np.zeros(n); vy = np.zeros(n)
    x[0] = r0; y[0] = 0.0
    vx[0] = 0.0; vy[0] = v0
    for i in range(n-1):
        ax, ay = aceleracion(x[i], y[i])
        vx[i+1] = vx[i] + h*ax
        vy[i+1] = vy[i] + h*ay
        x[i+1] = x[i] + h*vx[i+1]
        y[i+1] = y[i] + h*vy[i+1]
    return x, y, vx, vy

def rk4(h, T):
    n = int(T/h)
    x = np.zeros(n); y = np.zeros(n)
    vx = np.zeros(n); vy = np.zeros(n)
    x[0] = r0; y[0] = 0.0
    vx[0] = 0.0; vy[0] = v0
    for i in range(n-1):
        def F(x, y, vx, vy):
            ax, ay = aceleracion(x, y)
            return np.array([vx, vy, ax, ay])
        u = np.array([x[i], y[i], vx[i], vy[i]])
        k1 = F(*u)
        k2 = F(*(u + 0.5*h*k1))
        k3 = F(*(u + 0.5*h*k2))
        k4 = F(*(u + h*k3))
        u_new = u + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x[i+1], y[i+1], vx[i+1], vy[i+1] = u_new
    return x, y, vx, vy

def energia(x, y, vx, vy):
    r = np.sqrt(x*x + y*y)
    return 0.5*(vx**2 + vy**2) - GM/r

h = 10.0 ## Aqui cambio el h (10 o 20) //////////////////////////
x_ec, y_ec, vx_ec, vy_ec = euler_cromer(h, T_orb)
x_rk, y_rk, vx_rk, vy_rk = rk4(h, T_orb)

eps_ec = energia(x_ec, y_ec, vx_ec, vy_ec)
eps_rk = energia(x_rk, y_rk, vx_rk, vy_rk)

delta_ec = np.max(np.abs((eps_ec - eps_ec[0])/eps_ec[0]))
delta_rk = np.max(np.abs((eps_rk - eps_rk[0])/eps_rk[0]))

print(f"Error máximo de energía Euler-Cromer: {delta_ec:.3e}")
print(f"Error máximo de energía RK4: {delta_rk:.3e}")

# Aqui Gráficas
plt.figure()
plt.plot(x_ec, y_ec, label='Euler-Cromer')
plt.plot(x_rk, y_rk, '--', label='RK4')
plt.axis('equal')
plt.xlabel('x [m]'); plt.ylabel('y [m]')
plt.legend(); plt.title('Trayectoria orbital')
plt.grid(True)

plt.figure()
t = np.linspace(0, T_orb, len(eps_ec))
plt.semilogy(t, np.abs((eps_ec - eps_ec[0])/eps_ec[0]), label='Euler-Cromer')
plt.semilogy(t, np.abs((eps_rk - eps_rk[0])/eps_rk[0]), label='RK4')
plt.xlabel('t [s]'); plt.ylabel('Error relativo de energía')
plt.legend(); plt.title('Error relativo de energía')
plt.grid(True)
plt.show()