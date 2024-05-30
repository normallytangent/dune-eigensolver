#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Export data
def plot(status, xlabel, ylabel):
    match status:
        case 'tol':
            iterlow = np.loadtxt('./measurements/maxiter_3_tol_1e-1_N_144_m_32', delimiter=' ',comments='#')
            iterhigh = np.loadtxt('./measurements/maxiter_300_tol_1e-1_N_144_m_32', delimiter=' ',comments='#')

            plt.plot(iterlow[:,0],iterlow[:,5],label='iter 3, eigen',marker='o')
            plt.plot(iterlow[:,1],iterlow[:,7],label='iter 3, stewart',marker='o')
            plt.plot(iterhigh[:,0],iterhigh[:,5],label='iter 300, eigen',marker='o')
            plt.plot(iterhigh[:,1],iterhigh[:,7],label='iter 3, stewart',marker='o')

            #plt.xscale('log')
            #plt.yscale('log')

            # Create legend
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title('Error in eigenvalues')
            plt.legend(loc='lower right')

            # Save plot
            plt.savefig('img/2_2.pdf')

            #Show plot
            #plt.show()
            
        case 'bw':
            bw1nodeB = np.loadtxt('./measurements/b-flood-1-node-500-itr.txt', delimiter=' ',comments='#')
            bw1nodeNB = np.loadtxt('./measurements/nb-flood-1-node-500-itr.txt', delimiter=' ',comments='#')
          
            bw2nodeB = np.loadtxt('./measurements/b-flood-2-node-500-itr.txt', delimiter=' ',comments='#')
            bw2nodeNB = np.loadtxt('./measurements/nb-flood-2-node-500-itr.txt', delimiter=' ',comments='#')

            plt.plot(bw1nodeB[:,0],bw1nodeB[:,1],label='1 node, blocking',marker='o')
            plt.plot(bw1nodeNB[:,0],bw1nodeNB[:,1],label='1 node, non-blocking',marker='^')
            plt.plot(bw2nodeB[:,0],bw2nodeB[:,1],label='2 node, blocking',marker='o')
            plt.plot(bw2nodeNB[:,0],bw2nodeNB[:,1],label='2 nodes, nonblockin',marker='^')

            plt.xscale('log')
            plt.yscale('log')

            # Create legend
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title('Ex 2.3 Bandwidth Measurement using Ping Pong')
            plt.legend(loc='lower right')

            # Save plot
            plt.savefig('plots/img/2_3.pdf')

            #Show plot
            # plt.show()

        case _:
            return "Try again!"

def main():
        plot('tol','Eigenvalues', 'Error in eigenvalues')
        #plt.clf()
        #plot('bw','Message size [KB]', 'Bandwidth [GB/sec]')


if __name__ == "__main__":
        main()
