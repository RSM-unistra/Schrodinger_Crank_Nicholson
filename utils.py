import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

def animRealImag(x, psi, V):
    Nt = psi.shape[1]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
    fig.suptitle(r"Time evolution of $Re(\Psi)$ and $Im(\Psi)$")

    for ax in axs[-1]:
        ax.set_xlabel(r'Position $x$ [a.u.]')
    for ax in axs[:, 0]:
        ax.set_ylabel('Amplitude')

    axs = axs.flatten()

    for ax in axs:
        ax.set_xlim((x[0], x[-1]))
        ax.set_ylim((-1.1 * np.max(abs(psi)), 1.1 * np.max(abs(psi))))
        ax.grid()
        ax.plot(x, V, c='gray', alpha=0.7)

    line_real = axs[1].plot([], [], c='#1f77b4', label=r'Re$(\Psi(x, t))$')[0]
    line_imag = axs[3].plot([], [], c='#ff7f0e', label=r'Im$(\Psi(x, t))$')[0]
    line_mod_1 = axs[1].plot([], [], 'g-', label=r'$|\Psi(x, t)|^2$')[0]
    line_mod_2 = axs[3].plot([], [], 'g-', label=r'$|\Psi(x, t)|^2$')[0]

    def update(i):
        line_real.set_data(x, np.real(psi[:, i]))
        line_imag.set_data(x, np.imag(psi[:, i]))
        line_mod_1.set_data(x, np.abs(psi[:, i]) ** 2)
        line_mod_2.set_data(x, np.abs(psi[:, i]) ** 2)
        return line_real, line_imag, line_mod_1, line_mod_2

    anim = FuncAnimation(fig, update, frames=range(Nt), blit=True, interval=5)

    # Plot initial wavefunction at t=0
    axs[0].plot(x, np.real(psi[:, 0]), c='#1f77b4', label=r'Re$(\Psi(x, t=0))$')
    axs[2].plot(x, np.imag(psi[:, 0]), c='#ff7f0e', label=r'Im$(\Psi(x, t=0))$')

    for ax in axs:
        ax.legend()

    plt.show()

def animDensity(x, psi, V):
    Nt = psi.shape[1]

    fig, ax = plt.subplots()

    ax.set_xlabel("Position")
    ax.set_ylabel(r"$|\Psi|^2$")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(np.min(np.abs(psi) ** 2) / 2, np.max(np.abs(psi) ** 2) / 2)
    
    ax.plot(x, V, "r", label=r"$V$")
    
    lines, = ax.plot([], [], lw=1, color="black", label=r"$|\Psi|^2$")
    
    def update(frame):
        lines.set_data(x, np.abs(psi[:, frame]) ** 2)
        return lines,

    anim = FuncAnimation(fig, update, frames=range(Nt), blit=True, interval=1)
    
    ax.legend()
    plt.grid()

    plt.show()

def plotTimeEvolution(x, t, psi):
    fig, ax = plt.subplots()
    elem = np.abs(psi) ** 2
    mesh = ax.pcolormesh(x, t, elem.T, cmap='gray')
    fig.colorbar(
        mesh,
        ax=ax,
        orientation='vertical',
        label=r'$|\psi(x)|^2$',
        fraction=0.06, pad=0.02
    )
    plt.ylabel("Time [a.u.]", fontsize=13)
    plt.xlabel("Position [a.u.]", fontsize=13)
    plt.title("Time Evolution of the packet", fontsize=16)
    plt.show()

def plotExpectedPosition(x, t, psi):
    plt.title("Expected Position [a.u.]", fontsize=16)
    plt.xlabel("Time [a.u.]", fontsize=12)
    plt.ylabel(r"$\langle x \rangle$", fontsize=12)
    
    expected = np.zeros(len(t))
    for i in range(len(t)):
        y = psi[:, i]
        expected[i] = np.trapz(x * np.abs(y) ** 2, x=x)

    plt.plot(t, expected, c="black", lw=1)
    plt.grid(True)
    plt.show()

def plotUncertainty(x, t, psi):
    mean_x = np.zeros(len(t))
    mean_x_2 = np.zeros(len(t))
    
    for i in range(len(t)):
        y = psi[:, i]
        mean_x[i] = simps(x * np.abs(y) ** 2, x=x)
        mean_x_2[i] = simps(x ** 2 * np.abs(y) ** 2, x=x)

    plt.plot(t, mean_x_2 - mean_x ** 2, lw=1, c='black')

    plt.xlabel(r"Time $t$ [a.u.]")
    plt.ylabel(r"$\Delta x$")
    plt.title("Uncertainty")
    plt.grid(True)
    plt.show()

def plotSome(x, t, psi):
    Nt = len(t)
    fig, axs = plt.subplots(ncols=1, nrows=5, sharex=True, sharey=True, figsize=(9, 7))
    t_ = 0
    dt_ = Nt // axs.size - 1
    for ax in axs.flatten():
        ax.set_title(f'Time: {t_ / Nt} [a.u.]')
        ax.plot(x, np.real(psi[:, t_]), label=r"Re($\Psi$)")
        ax.plot(x, np.imag(psi[:, t_]), label=r"Im($\Psi$)")
        ax.grid(True)
        t_ += dt_
        ax.legend()
    plt.tight_layout()
    plt.show()