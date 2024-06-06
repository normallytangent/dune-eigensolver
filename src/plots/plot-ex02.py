#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines
plt.rcParams["figure.figsize"] = (9, 7)

# Export data
def plot_frame(ax, status, nev, xlabel, ylabel,submethod):
    match status:
        case 'tol':
            filename = ""
            for x in ['1','2','3','4','5']:
                filename = 'load'+x
                filename = np.loadtxt('../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_2e-'+x+'_overlap_3_method_std_submethod_'+submethod, delimiter=' ',comments='#')
                ax.semilogy(np.arange(nev),filename[:,5],label='2e-'+x,marker='')

            # Create legend
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(10e-18,10e-1)
            ax.set_title('n:40000, m:32, shift: 1e-3, overlap: 3, method: '+submethod)
            #ax.legend(loc='upper left')
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=6.5)

        case _:
            return "Try again!"

def main():
    nev = 32
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    plot_frame(ax1, 'tol', nev, 'Eigenvalues ranked smallest to largest', 'Error in eigenvalues','ftw')
    plot_frame(ax2, 'tol', nev, 'Eigenvalues ranked smallest to largest', 'Error in eigenvalues', 'stw')

    fig.suptitle('Accuracy of computed eigenvalues w.r.t. decreasing error tolerances')
    fig.tight_layout()
    fig.savefig('img/tol.pdf')
#   plt.clf()

if __name__ == "__main__":
        main()
