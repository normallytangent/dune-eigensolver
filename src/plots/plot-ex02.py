#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (11.69,8.27)
plt.rcParams['figure.constrained_layout.use'] = True

# Export data
def plot_frame(ax, status, n, nev, submethod, matrix):
    match status:
        case 'tol-std':
            tarry = []
            #for x in map(str, np.arange(2,3)):
            #    for y in map(str, np.arange(2,10,2)):
            #        filename = '../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_std_submethod_'+submethod+'_'+matrix
            #        dataset = np.loadtxt(filename, delimiter=' ',comments='#')
            #        itr = find_iterations(filename)
            #        time = '{0:.3g}'.format(float(find_time(filename)))
            #        tarry.append(time)
            #        custom_label=y+'e-'+x+', itr: '+itr+'t: '+time
            #        ax.semilogy(np.arange(nev),dataset[:,-1],label=custom_label, marker='')
            for x in map(str, np.arange(1,5)):
                for y in map(str, np.arange(1,3)):
                    filename = '../measurements/N_'+n+'_m_'+nev+'_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_std_submethod_'+submethod+'_'+matrix
                    dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                    itr = find_iterations_time(filename)
                    time = '{0:.3g}'.format(float(find_iterations_time(filename)))
                    tarry.append(time)
                    custom_label=y+'e-'+x+', itr: '+itr+'t: '+time
                    ax.semilogy(np.arange(nev),dataset[:,-1],label=custom_label, marker='')

            # Create legend
            ax.set_title('problem: ' +matrix +', method: '+submethod)
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=2)
            #ax.legend(labels=[custom_label.partition("t: ")[2]])
            #ax.legend(labels=tarry)
            #ax.legend()

        case 'tol-gen':
            tarry = []
            # for x in map(str, np.arange(1,3)):
            #     for y in map(str, np.arange(2,10,2)):
            #         filename = '../measurements/N_40k_m_32_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_gen_submethod_'+submethod+'_'+matrix
            #         dataset = np.loadtxt(filename, delimiter=' ',comments='#')
            #         itr, time = find_iterations_time(filename)
            #         time = '{0:.4g}'.format(float(time))
            #         tarry.append(time)
            #         custom_label = y+'e-'+x+', itr: '+itr+'t: '+time
            #         ax.semilogy(np.arange(nev),dataset[:,-2],label=custom_label, marker='')
            for x in map(str, np.arange(1,5)):
                for y in map(str, np.arange(1,2)):
                    filename = '../simu/N_'+str(n)+'_m_'+str(nev)+'_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_gen_submethod_'+submethod+'_'+matrix
                    dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                    itr, time = find_iterations_time(filename)
                    time = '{0:.4g}'.format(float(time))
                    tarry.append(time)
                    custom_label = y+'e-'+x+', itr: '+str(itr)+' t: '+time
                    ax.semilogy(np.arange(nev),dataset[:,-2],label=custom_label, marker='')

            # Create legend
            ax.set_title('problem: ' +matrix +', method: '+submethod)
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=2)
            #ax.set_ylim(1e-17, 1e+0)
            #ax.legend(labels=tarry)
            # legend = ax.legend()
            # legend._legend_box.width = 100  # <-- Setting new width``
            # legend.get_frame().set_linewidth(0.0)

        case _:
            return "Try again!"


def plot_frame_timeratio(ax, status, n, nev, submethod, matrix):
    match status:
        case 'tol-gen':
            xarray = []
            yarray = []
            for x in map(str, np.arange(1,5)):
                for y in map(str, np.arange(1,0,-1)):
                    filename = '../simu/N_'+str(n)+'_m_'+str(nev)+'_maxiter_4k_shift_1e-3_tol_'+y+'e-'+x+'_overlap_3_method_gen_submethod_'+submethod+'_'+matrix
                    itr, time = find_iterations_time(filename)
                    arp_itr, arp_time = find_arpack_iterations_time(filename)
                    xarray.append(float(y+'e-'+x))
                    # print(float(time))
                    # print(float(arp_time))
                    yarray.append(float(time)/float(arp_time))
                    strtime = '{0:.4g}'.format(float(time))
                    arp_strtime = '{0:.2g}'.format(float(arp_time))
                    ax.annotate(rf"$\frac{{{strtime}}}{{{arp_strtime}}}$", xy=(xarray[-1], yarray[-1]), xytext=(xarray[-1], yarray[-1]), textcoords='data', fontsize=10, ha=('right' if int(n)%80==0 else 'left'), \
                            va=('top' if int(n)%40==0 else 'bottom'), rotation=10, annotation_clip=True, bbox=dict(boxstyle='round,pad=0.2', fc='none', alpha=0.1))
                    
            ax.semilogx(xarray, yarray, marker='o', label= str(n))

            # Create legend
            ax.set_title('Problem: ' +matrix+ ', method: ' + ("Ftworth" if submethod=="ftw" else "Stewart"))
            ax.set_aspect('auto','box')
            legend = ax.legend()

        case _:
            return "Try again!"

def find_iterations_time(filename):
    with open(filename) as f:
        datafile = f.readlines()
    itr = 0
    time = 0
    for line in datafile:
        if "GeneralizedInverse:" in line:
            itr = line.partition("iterations=")[2]
            time = line.partition("time_total=")[2]
            time = time.partition(" ")[0]
        elif "GeneralizedSymmetricStewart:" in line:
            itr = line.partition("iterations=")[2]
            time = line.partition("time_total=")[2]
            time = time.partition(" ")[0]
    return itr, time

def find_arpack_iterations_time(filename):
    with open(filename) as f:
        datafile = f.readlines()
    itr = 0
    time = 0
    for line in datafile:
        if "Arpack:" in line:
            itr = line.partition("iterations=")[2]
            time = line.partition("time_total=")[2]
            time = time.partition(" ")[0]
    return itr, time

def beautify(fig, title, xlabel, ylabel, location):
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.savefig(location)

def main():
    n = 67601
    nev = 32
    matrix = "neu" # "dir"
    prob = "tol-gen" # "tol-std"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    plot_frame(ax1, prob, n, nev, 'ftw', matrix)
    plot_frame(ax2, prob, n, nev, 'stw', matrix)

    title ='Accuracy of computed eigenvalues given various error tolerances \n n:'+str(n)+', m:'+str(nev)+', shift: 1e-3, regularization: 0, overlap: 3'
    xlabel='Eigenvalues ranked smallest to largest'
    ylabel='Error in eigenvalues'
    location='img/'+prob+'-'+matrix+'-n-'+str(n)+'-m-'+str(nev)+'.pdf'
    beautify(fig,title,xlabel,ylabel,location)

    nev = 32
    matrix = "neu"
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False)
    n_list = [40, 80]
    for n in n_list:
        n*=n
        plot_frame_timeratio(ax1, 'tol-gen', n, nev, 'ftw', matrix)
        plot_frame_timeratio(ax2, 'tol-gen', n, nev, 'stw', matrix)
    
    n = 67600
    # # plot_frame_timeratio(ax1, 'tol-gen', n, nev, 'ftw', matrix)
    plot_frame_timeratio(ax2, 'tol-gen', n, nev, 'stw', matrix)

    n = 67601
    plot_frame_timeratio(ax1, 'tol-gen', n, nev, 'ftw', matrix)
    plot_frame_timeratio(ax2, 'tol-gen', n, nev, 'stw', matrix)

    title ='Runtime ratios of eigensolver w.r.t arpack given error tolerances \n m:'+str(nev)+', shift: 1e-3, regularization: 0, overlap: 3'
    xlabel='Tolerance'
    ylabel='Runtime of Eigensolver relative to Arpack'
    location='img/timeratio-gen-'+matrix+'-m-'+str(nev)+'.pdf'
    beautify(fig,title,xlabel,ylabel,location)
    # plt.show()

if __name__ == "__main__":
        main()
