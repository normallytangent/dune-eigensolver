#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams['figure.constrained_layout.use'] = True

# Export data
def plot_frame(ax, status, nev, submethod):
    match status:
        case 'tol':
            filename = ""
            for x in map(str, np.arange(1,6)):
                filename = 'load'+x
                filename = np.loadtxt('../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_2e-'+x+'_overlap_3_method_std_submethod_'+submethod, delimiter=' ',comments='#')
                ax.semilogy(np.arange(nev),filename[:,5],label='2e-'+x,marker='')

            # Create legend
            ax.set_title('method: '+submethod)
            ax.set_aspect('equal','box')
            labelLines(ax.get_lines(),align=False,zorder=6.5)

        case _:
            return "Try again!"

def beautify(fig, title, xlabel, ylabel, location):
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.savefig(location)

def main():
    nev = 32
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    plot_frame(ax1, 'tol', nev, 'ftw')
    plot_frame(ax2, 'tol', nev, 'stw')

    title ='Accuracy of computed eigenvalues given various error tolerances \n n:40000, m:32, shift: 1e-3, overlap: 3'
    xlabel='Eigenvalues ranked smallest to largest'
    ylabel='Error in eigenvalues'
    location='img/tol.pdf'
    beautify(fig,title,xlabel,ylabel,location)

if __name__ == "__main__":
        main()
