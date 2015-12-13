import numpy.random as rn
import cPickle as pickle
import numpy as np
import sktensor as skt
import sys, os
import scipy.sparse

infn = sys.argv[1]
inbase, _ = os.path.splitext(infn)
spikes = np.genfromtxt(infn, dtype=int, delimiter=' ') 

N = spikes[:,0].max()+1
T = spikes[:,1].max()
L = 100


spikemtx = scipy.sparse.coo_matrix((np.ones(spikes.shape[0]), (spikes[:,0], spikes[:,1]))).tocsc().todense()
spikemtx[4,] = rn.poisson(np.roll(spikemtx[1,:], 5)) #cheat!
spikemtx[8,] = rn.poisson(np.roll(spikemtx[15,:], 5)) #cheat!
spikemtx = spikemtx[np.fliplr(spikemtx.mean(axis=1).ravel().argsort())][0,:,:] #sort so densest in bottom right

mat = np.zeros((N, N, L))
for l in xrange(L):
    print "lag", l
    for j in xrange(N):
        mat[:,j,l] = np.sum(np.logical_and(spikemtx, np.roll(np.roll(spikemtx, shift=-l, axis=1), shift=j, axis=0)), axis=1)[1,:]

data = skt.dtensor(mat)

outfn = inbase + '.dtensor.dat'
with open(outfn, 'w+') as f:            # can be stored as a .dat using pickle
    pickle.dump(data, f)

with open(outfn, 'r') as f:             # can be loaded back in using pickle.load
    tmp = pickle.load(f)
    assert np.allclose(tmp, data)
