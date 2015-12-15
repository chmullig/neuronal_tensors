import numpy.random as rn
import cPickle as pickle
import numpy as np
import sktensor as skt
import sys, os
import scipy.sparse

infn = sys.argv[1]
inbase, inext = os.path.splitext(infn)

L = 250
if inext == 'txt':
	spikes = np.genfromtxt(infn, dtype=int, delimiter=' ') 

	N = spikes[:,0].max()+1
	T = spikes[:,1].max()
	spikemtx = scipy.sparse.coo_matrix((np.ones(spikes.shape[0]), (spikes[:,0], spikes[:,1]))).tocsc().todense()
elif inext == '.npz':
	THRESHOLD = .05
	dat = np.load(open(infn))
	spikemtx = (dat['n_best']>THRESHOLD).astype(int)
	N, T = spikemtx.shape


if False:
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
for l in xrange(0, L):
	rolled = np.roll(spikemtx, shift=-l)
	for i in xrange(N):
        mat[i,:,l] = np.sum(np.logical_and(spikemtx[i,:], rolled), axis=1)
np.fill_diagonal(mat[:,:,0], 0) #get rid of the neuron's spike counts
data = skt.dtensor(mat)

outfn = inbase + '.dtensor.dat'
with open(outfn, 'w+') as f:            # can be stored as a .dat using pickle
    pickle.dump(data, f)

with open(outfn, 'r') as f:             # can be loaded back in using pickle.load
    tmp = pickle.load(f)
    assert np.allclose(tmp, data)
