#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Export data
def plot(status, tol, nev, xlabel, ylabel):
    match status:
        case 'tol':
            iterlow = np.loadtxt('../measurements/N_40k_m_32_'+tol+'_maxiter_100', delimiter=' ',comments='#')
            itermid = np.loadtxt('../measurements/N_40k_m_32_'+tol+'_maxiter_200', delimiter=' ',comments='#')
            iterhigh = np.loadtxt('../measurements/N_40k_m_32_'+tol+'_maxiter_300', delimiter=' ',comments='#')
            itergiant = np.loadtxt('../measurements/N_40k_m_32_'+tol+'_maxiter_10k', delimiter=' ',comments='#')

            plt.plot(np.arange(nev),iterlow[:,6],label='iter 100, eigen',marker='o')
            plt.plot(np.arange(nev),iterlow[:,8],label='iter 100, stewart',marker='o')
            plt.plot(np.arange(nev),iterlow[:,6],label='iter 200, eigen',marker='o')
            plt.plot(np.arange(nev),iterlow[:,8],label='iter 200, stewart',marker='o')
            plt.plot(np.arange(nev),iterhigh[:,6],label='iter 300, eigen',marker='o')
            plt.plot(np.arange(nev),iterhigh[:,8],label='iter 300, stewart',marker='o')
            plt.plot(np.arange(nev),itergiant[:,6],label='iter 10k, eigen',marker='o')
            plt.plot(np.arange(nev),itergiant[:,8],label='iter 10k, stewart',marker='o')

            #plt.xscale('log')
            plt.yscale('log')

            # Create legend
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title('Error in eigenvalues: N:40K, m:32, tol: 1e-3')
            plt.legend(loc='lower right')

            # Save plot
            plt.savefig('img/'+tol+'.pdf')

            #Show plot
            #plt.show()
            
        case _:
            return "Try again!"

def main():
    tol = ['tol_1e-1', 'tol_1e-2'] #, 'tol_1e-3']
    nev = 32
    for x in tol:
        plot('tol', x, nev, 'Eigenvalues', 'Error in eigenvalues')
        plt.clf()

if __name__ == "__main__":
        main()
