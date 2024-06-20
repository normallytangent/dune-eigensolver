#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams['figure.constrained_layout.use'] = True

# Export data
def plot_frame(ax, status, nev, submethod, matrix):
    match status:
        case 'tol-std':
            for y in map(str, np.arange(2,10,2)):
                filename = '../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_'+y+'e-2_overlap_3_method_std_submethod_'+submethod+'_'+matrix
                dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                itr = find_iterations(filename)
                ax.semilogy(np.arange(nev),dataset[:,-1],label=y+'e-2, itr: '+itr,marker='')
            for x in map(str, np.arange(1,5)):
                for y in map(str, np.arange(1,2)):
                    filename = '../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_std_submethod_'+submethod+'_'+matrix
                    dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                    itr = find_iterations(filename)
                    ax.semilogy(np.arange(nev),dataset[:,-1],label=y+'e-'+x+', itr: '+itr,marker='')

            # Create legend
            ax.set_title('problem: ' +matrix +', method: '+submethod)
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=6.5)

        case 'tol-gen':
            for y in map(str, np.arange(2,10,2)):
                filename = '../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_'+y+'e-2_overlap_3_method_gen_submethod_'+submethod+'_'+matrix
                dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                itr = find_iterations(filename)
                ax.semilogy(np.arange(nev),dataset[:,-2],label=y+'e-2, itr: '+itr,marker='')
            for x in map(str, np.arange(1,5)):
                for y in map(str, np.arange(1,2)):
                    filename = '../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_gen_submethod_'+submethod+'_'+matrix
                    dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                    itr = find_iterations(filename)
                    ax.semilogy(np.arange(nev),dataset[:,-2],label=y+'e-'+x+', itr: '+itr,marker='')

            # Create legend
            ax.set_title('problem: ' +matrix +', method: '+submethod)
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=6.5)

        case _:
            return "Try again!"

def find_iterations(filename):
    with open(filename) as f:
        datafile = f.readlines()
    itr = 0
    for line in datafile:
        if "iterations=" in line:
            itr = line.partition("iterations=")[2]
    return itr

def beautify(fig, title, xlabel, ylabel, location):
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.savefig(location)

def main():
    nev = 32
    matrix = "dir"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    plot_frame(ax1, 'tol-std', nev, 'ftw', matrix)
    plot_frame(ax2, 'tol-std', nev, 'stw', matrix)

    title ='Accuracy of computed eigenvalues given various error tolerances \n n:40000, m:32, shift: 1e-3, overlap: 3'
    xlabel='Eigenvalues ranked smallest to largest'
    ylabel='Error in eigenvalues'
    location='img/tol-std-'+matrix+'.pdf'
    beautify(fig,title,xlabel,ylabel,location)

    nev = 32
    matrix = "neu"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    plot_frame(ax1, 'tol-gen', nev, 'ftw', matrix)
    plot_frame(ax2, 'tol-gen', nev, 'stw', matrix)

    title ='Accuracy of computed eigenvalues given various error tolerances \n n:40000, m:32, shift: 1e-3, regularization: 0, overlap: 3'
    xlabel='Eigenvalues ranked smallest to largest'
    ylabel='Error in eigenvalues'
    location='img/tol-gen-'+matrix+'.pdf'
    beautify(fig,title,xlabel,ylabel,location)

if __name__ == "__main__":
        main()
