from __future__ import print_function
import sys
import os
import numpy as np


if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    for egene in range(len(content)):
        gene = str(content[egene])
        
        X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
        n, p = X.shape
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))
        
        beta = np.load(os.path.join(inpath + "b_" + gene) + ".npy")
        beta = beta.reshape((beta.shape[0],))
        beta = np.sqrt(n) * beta
        true_mean = X.dot(beta)
        
        y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
        y = y.reshape((y.shape[0],))
        
        E = np.load(os.path.join(inpath + "s_" + gene) + ".npy")
        E = E.reshape((E.shape[0],))
        
        active = np.zeros(p, dtype=bool)
        active[E] = 1
        nactive = active.sum()
        
        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        true_val = projection_active.T.dot(true_mean)
        
        point_est = projection_active.T.dot(y) 
        sd = np.sqrt(np.linalg.inv(X[:, active].T.dot(X[:, active])).diagonal())
        unad_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T
        unad_length = (unad_intervals[:,1]- unad_intervals[:,0]).sum() / nactive
        unad_risk = np.power(point_est- true_val, 2.).sum() / nactive
        coverage_unad = np.zeros(nactive)
        
        for k in range(true_val.shape[0]):
            if (unad_intervals[k, 0] <= true_val[k]) and (true_val[k] <= unad_intervals[k, 1]):
                coverage_unad[k] += 1
        coverage_unad = coverage_unad.sum() / nactive

        output = np.vstack((nactive, coverage_unad, unad_length, unad_risk))

        outfile = os.path.join(outdir + "fwd_bwd_output_" + gene + ".txt")
        np.savetxt(outfile, output)
        sys.stderr.write("Iteration completed" + str(egene) + "\n")