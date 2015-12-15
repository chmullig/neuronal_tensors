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
L = 20


spikemtx = scipy.sparse.coo_matrix((np.ones(spikes.shape[0]), (spikes[:,0], spikes[:,1]))).tocsc().todense()
print spikemtx.shape

#ok, now insert fake data...
been_from = set()
with open('log.txt', 'w') as f:
	permutations = rn.poisson(5)
	f.write("Permuting: " + str(permutations) + " times\n")
	for i in xrange(permutations):
		swaps = (0, 0)
		while swaps[0] == swaps[1] or swaps[1] in been_from:
			swaps = rn.random_integers(0, high=N-1, size=2)
		frm = swaps[0]
		to = swaps[1]
		been_from.add(frm)
		displacement = rn.poisson(5)
		f.write("%s: %s -> %s by %s\n" % (i, frm, to, displacement))
		spikemtx[to,:] = rn.poisson(np.roll(spikemtx[frm,:], displacement))
	swaps = (0, 0)
	while swaps[0] == swaps[1] or swaps[1] in been_from:
		swaps = rn.random_integers(0, high=N-1, size=2)
	frm = swaps[0]
	to = swaps[1]
	displacement = rn.poisson(5)
	overlaps = np.logical_and(spikemtx[to,:], np.roll(spikemtx[frm,:], displacement))
	spikemtx[to, np.where(overlaps)] = np.repeat(0, np.sum(overlaps))
	f.write("\nSUPPRESSING %s: %s -> %s by %s\n" % (i, frm, to, displacement))

mat = np.zeros((N, N, L))
for i in xrange(N):
    for j in xrange(N):
        for l in xrange(1, L):
        	mat[i,j,l] = np.sum(np.logical_and(spikemtx[i,:], np.roll(spikemtx[j,:], shift=-l))) #/ np.sum(spikemtx[i,:])
data = skt.dtensor(mat)

outfn = inbase + '.dtensor.dat'
with open(outfn, 'w+') as f:            # can be stored as a .dat using pickle
    pickle.dump(data, f)

with open(outfn, 'r') as f:             # can be loaded back in using pickle.load
    tmp = pickle.load(f)
    assert np.allclose(tmp, data)
