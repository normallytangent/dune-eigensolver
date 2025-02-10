#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine, labelLines
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (11.69,8.27)
# plt.rcParams['figure.constrained_layout.use'] = True

# Export data
def plot_convergence_ratio(ax, status, n, nev, acc, submethod, matrix):
    match status:
        case 'eig-ratio':
            tarry = []
            for x in map(str, np.arange(1,5)):
                for y in map(str, range(1,0,-1)):
                    filename = '../adapt_0_symmetric_gen/N_'+str(n)+'_m_'+str(nev)+'_acc_'+str(acc)+'_tol_'+y+'e-'+x+'_gen_'+submethod+'_'+matrix
                    dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                    itr, time = find_iterations_time(filename)
                    time = '{0:.4g}'.format(float(time))
                    tarry.append(time)
                    ratios = []
                    ratios = [dataset[i,1] / dataset[i+1,1] for i in np.arange(0, nev-1)]  
                    custom_label = y+'e-'+x #+', itr: '+str(itr)+' t: '+time
                    ax.plot(np.arange(0, nev-1),ratios,label=custom_label, marker='')

            # Create legend
            ax.set_title('Method: ' + ("GeneralizedInverse" if submethod=="ftw" else "GeneralizedSymmetricStewart"))
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=2)
            ax.set_xlim(-1, nev+1)
            # ax.legend(labels=tarry)
            legend = ax.legend()
            # legend._legend_box.width = 100  # <-- Setting new width``
            # legend.get_frame().set_linewidth(0.0)

        case _:
            return "Try again!"

def plot_frame(ax, status, n, nev, acc, submethod, matrix):
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
            for x in map(str, np.arange(1,5)):
                for y in map(str, range(2,0,-1)):
                    filename = '../adapt_0_symmetric_gen/N_'+str(n)+'_m_'+str(nev)+'_acc_'+str(acc)+'_tol_'+y+'e-'+x+'_gen_'+submethod+'_'+matrix
                    dataset = np.loadtxt(filename, delimiter=' ',comments='#')
                    itr, time = find_iterations_time(filename)
                    time = '{0:.4g}'.format(float(time))
                    tarry.append(time)
                    custom_label = y+'e-'+x+', itr: '+str(itr)+' t: '+time
                    ax.semilogy(dataset[:,1],dataset[:,-2],label=custom_label, marker='')

            # Create legend
            # ax.set_title('Problem: ' +matrix +', method: '+submethod)
            ax.set_title('Method: ' + ("GeneralizedInverse" if submethod=="ftw" else "GeneralizedSymmetricStewart"))
            ax.set_aspect('auto','box')
            labelLines(ax.get_lines(),align=False,zorder=2)
            # ax.set_xlim(1e-17, 1e+0)
            #ax.legend(labels=tarry)
            # legend = ax.legend()
            # legend._legend_box.width = 100  # <-- Setting new width``
            # legend.get_frame().set_linewidth(0.0)

        case _:
            return "Try again!"

def plot_time(ax, status, n, nev, acc, submethod, matrix):
    match status:
        case 'time':
            xarray = []
            yarray = []
            arparray = []
            acc_array = []
            for x in map(str, np.arange(1,5)):
                for y in map(str, range(2,0,-1)):
                    filename = '../adapt_0_symmetric_gen/N_'+str(n)+'_m_'+str(nev)+'_acc_'+str(acc)+'_tol_'+y+'e-'+x+'_gen_'+submethod+'_'+matrix
                    itr, time = find_iterations_time(filename)
                    arp_itr, arp_time = find_arpack_iterations_time(filename)
                    xarray.append(float(y+'e-'+x))
                    yarray.append(float(time)) #### CHANGE4 THIS LINE TO A RATIO OF TIME/ITERATION
                    arparray.append(float(arp_time))
                    # strtime = '{0:.4g}'.format(float(time))
                    # arp_strtime = '{0:.2g}'.format(float(arp_time))
                    # ax.annotate(rf"$\frac{{{strtime}}}{{{arp_strtime}}}$", xy=(xarray[-1], yarray[-1]), xytext=(xarray[-1], yarray[-1]), textcoords='data', fontsize=10, ha=('right' if int(n)%80==0 else 'left'), \
                            # va=('top' if int(n)%40==0 else 'bottom'), rotation=10, annotation_clip=True, bbox=dict(boxstyle='round,pad=0.2', fc='none', alpha=0.1))
                    
                    # acc_array.append(str(n) + ' ' +str(acc))
            ax.semilogx(xarray, yarray, marker='.', label= str(n)+', '+str(int(nev*acc/4)) + ' eigvals' ) ## CHANGE5
            # ax.semilogx(xarray, arparray, marker='.',label= str(n)+',' + ' arpack' )

            # Create legend
            ax.set_title('Problem: '+ ("GeneralizedInverse" if submethod=="ftw" else "GeneralizedSymmetricStewart"))
            ax.set_aspect('auto','box')
            # ax.legend(labels=acc_array)
            legend = ax.legend()

        case _:
            return "Try again!"

def plot_frame_timeratio(ax, status, n, nev, acc, submethod, matrix):
    match status:
        case 'time-ratio':
            xarray = []
            yarray = []
            acc_array = []
            for x in map(str, np.arange(1,5)):
                for y in map(str, range(2,0,-1)):
                    filename = '../adapt_0_symmetric_gen/N_'+str(n)+'_m_'+str(nev)+'_acc_'+str(acc)+'_tol_'+y+'e-'+x+'_gen_'+submethod+'_'+matrix
                    itr, time = find_iterations_time(filename)
                    arp_itr, arp_time = find_arpack_iterations_time(filename)
                    xarray.append(float(y+'e-'+x))
                    # print(float(time))
                    # print(float(arp_time))
                    yarray.append(float(time)/float(arp_time))
                    # strtime = '{0:.4g}'.format(float(time))
                    # arp_strtime = '{0:.2g}'.format(float(arp_time))
                    # ax.annotate(rf"$\frac{{{strtime}}}{{{arp_strtime}}}$", xy=(xarray[-1], yarray[-1]), xytext=(xarray[-1], yarray[-1]), textcoords='data', fontsize=10, ha=('right' if int(n)%80==0 else 'left'), \
                            # va=('top' if int(n)%40==0 else 'bottom'), rotation=10, annotation_clip=True, bbox=dict(boxstyle='round,pad=0.2', fc='none', alpha=0.1))
                    
                    # acc_array.append(str(n) + ' ' +str(acc))
            ax.loglog(xarray, yarray, marker='.', label= str(n)+', '+str(int(nev*acc/4)) + ' eigvals' )

            # Create legend
            ax.set_title('Problem: '+ ("GeneralizedInverse" if submethod=="ftw" else "GeneralizedSymmetricStewart"))
            ax.set_aspect('auto','box')
            # ax.legend(labels=acc_array)
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
    n_list = [28900, 67600]
    # acc = 4
    acc_list = [1, 2, 3, 4]
    nev = 144
    matrix = "neu" # "dir"

    ###########################
    #
    #  Error in eigenvalues
    #
    ##########################
    # for n in n_list:
    #     for acc in acc_list:
    #         prob = "tol-gen" # "tol-std"
    #         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #         plot_frame(ax1, prob, n, nev, acc, 'ftw', matrix)
    #         plot_frame(ax2, prob, n, nev, acc, 'stw', matrix)

    #         title =''#'Accuracy of '+str(nev)+' eigenvalues computed with error tolerances ranging from 2e-1 to 1e-4.\n Convergence governed by the smallest '+str(int(nev * acc/4))+ ' eigenvalues. Problem size: ' +str(n)+', shift: 1e-3, regularization: 0, overlap: 3'
    #         xlabel='Eigenvalues ordered from smallest to largest'
    #         ylabel='Error in eigenvalues w.r.t. Arpack\'s solution with error tolerance of 1e-14'
    #         location='fig/adapt_0_symmetric_'+prob+'-'+matrix+'-n-'+str(n)+'-m-'+str(nev)+'-acc-'+str(acc)+'.pdf'
    #         beautify(fig,title,xlabel,ylabel,location)


    # ##########################
    
    #  Theoretical ratios of eigenvalues
    
    # #########################
    # prob = "eig-ratio" # "tol-std"
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # # for n in n_list:
    # n = 67600
    # plot_convergence_ratio(ax1, prob, n, nev, acc, 'ftw', matrix)
    # plot_convergence_ratio(ax2, prob, n, nev, acc, 'stw', matrix)

    # title =''#'Theoretical convergence of '+str(nev)+' eigenvalues computed with error tolerances ranging from 2e-1 to 1e-4.\n Convergence governed by the smallest '+str(int(nev * acc/4))+ ' eigenvalues. Problem size: ' +str(n)+', shift: 1e-3, regularization: 0, overlap: 3'
    # xlabel='Indices i corresponding to '+rf"${{\lambda_i}}$"+' from 1 to '+ str(nev)
    # ylabel=rf"$\frac{{\lambda_i}}{{\lambda_{{i+1}}}}$"
    # location='fig/adapt_0_symmetric_'+prob+'-'+matrix+'-n-'+str(n)+'-m-'+str(nev)+'-acc-'+str(acc)+'.pdf'
    # beautify(fig,title,xlabel,ylabel,location)

    ###########################
    #
    #  Time ratios of eigenvalues
    #
    ##########################
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # for n in n_list:
    #     for acc in acc_list:
    #         plot_frame_timeratio(ax1, 'time-ratio', n, nev, acc, 'ftw', matrix)
    #         plot_frame_timeratio(ax2, 'time-ratio', n, nev, acc, 'stw', matrix)

    # title =''#Runtime of Eigensolver normalised w.r.t Arpack given error tolerances \n m:'+str(nev)+', shift: 1e-3, regularization: 0, overlap: 3\n convergence controlled by some smallest eigenvalues'
    # xlabel='Tolerance'
    # ylabel='Runtime of Eigensolver relative to Arpack'
    # location='fig/adapt_0_symmetric_time-ratio-gen-'+matrix+'-m-'+str(nev)+'.pdf'
    # beautify(fig,title,xlabel,ylabel,location)

    ###########################
    #
    #  Timing 
    #
    ##########################
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    # for n in n_list:
    #     for acc in acc_list:
    #         plot_time(ax1, 'time', n, nev, acc, 'ftw', matrix)
    #         plot_time(ax2, 'time', n, nev, acc, 'stw', matrix)

    # title =''#'Runtime comparison Eigensolver given error tolerances \n m:'+str(nev)+', shift: 1e-3, regularization: 0, overlap: 3\n convergence controlled by some smallest eigenvalues'
    # xlabel='Tolerance'
    # ylabel='Runtime (in seconds)'
    # location='fig/adapt_0_time-gen-'+matrix+'-m-'+str(nev)+'.pdf'
    # beautify(fig,title,xlabel,ylabel,location)

    ###########################
    #
    #  Time per iteration, iteration per
    #
    ##########################
    ## CHANGE THE MARKED LINE IN THE FUNCTION BEFORE RUNNING IN (5) LOCATIONS
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True) ## CHANGE1
    for n in n_list:
        for acc in acc_list:
            plot_time(ax1, 'time', n, nev, acc, 'ftw', matrix)
            plot_time(ax2, 'time', n, nev, acc, 'stw', matrix)

    title =''#'Runtime comparison Eigensolver given error tolerances \n m:'+str(nev)+', shift: 1e-3, regularization: 0, overlap: 3\n convergence controlled by some smallest eigenvalues'
    xlabel='Tolerance'
    ylabel='Iterations per second' ## CHANGE2
    location='fig/adapt_0_iteration-per-time-gen-'+matrix+'-m-'+str(nev)+'.pdf' ## CHANGE3
    beautify(fig,title,xlabel,ylabel,location)

if __name__ == "__main__":
        main()
