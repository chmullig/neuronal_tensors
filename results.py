import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


infn = sys.argv[1]
inbase, _ = os.path.splitext(infn)
results = np.load(infn)

# Variational parameters
gamma_DK_M = results['gamma_DK_M'] # shape variational parameters
delta_DK_M = results['delta_DK_M'] # rate variational parameters

E_DK_M = results['E_DK_M']  # arithmetic expectation

G_DK_M = results['G_DK_M']  # geometric expectation

beta_M = results['beta_M']
print beta_M

N = G_DK_M[0].shape[0]
K = G_DK_M[0].shape[1]
L = G_DK_M[2].shape[0]


with PdfPages(inbase + '.pdf') as pdf:
	for k in xrange(K):
	    for TYPE, DK_M in (('E', E_DK_M), ('G', G_DK_M)):
	        plt.figure(figsize=(10,10))
	        plt.subplot(2, 2, 1)
	        sns.barplot(x="neuron", y="value",
	                          data=pd.DataFrame({'neuron':xrange(N), 'value':DK_M[0][:,k]}))
	        plt.subplot(2, 2, 2)
	        sns.barplot(x="neuron", y="value",
	                          data=pd.DataFrame({'neuron':xrange(N), 'value':DK_M[1][:,k]}))
	        plt.subplot(2, 2, (3,4))
	        sns.barplot(x="time", y="value",
	                          data=pd.DataFrame({'time':xrange(L), 'value':DK_M[2][:,k]}))
	        plt.title(TYPE + ' ' + str(k))
	        pdf.savefig()
	        plt.close()

