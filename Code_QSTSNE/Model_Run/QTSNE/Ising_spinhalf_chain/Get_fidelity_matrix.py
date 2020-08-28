import numpy as np
import os.path as path
import library.MPSClass as MPSClass
import library.BasicFunctions as bf
from library.BasicFunctions import mkdir
import time
N = 80
chi = 30
Jxy = 0
Jz = 1
hx = 0
hz = 0
bounc = 'periodic'
chainname = 'chain'
#=====================================
p0 = [0,0.02, 0.04, 0.06, 0.08, 0.1]
p1 = [0.12, 0.14, 0.16, 0.18, 0.2]
p2 = [0.22, 0.24, 0.26, 0.28, 0.3]
p3 = [0.32, 0.34, 0.36, 0.38, 0.4]
p4 = [0.42, 0.44, 0.46, 0.48, 0.5]
p5 = [0.52, 0.54, 0.56, 0.58, 0.6]
p6 = [0.62, 0.64, 0.66, 0.68, 0.7]
p7 = [0.72, 0.74, 0.76, 0.78, 0.8]
p8 = [0.82, 0.84, 0.86, 0.88, 0.9]
p9 = [0.92, 0.94, 0.96, 0.98]
P = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

Np=np.array(P).shape[0]
P1=P
P2=P
Np1=np.size(np.array(P1))
Np2=np.size(np.array(P2))
lable=np.array(P)
print(lable.T)
fe=np.zeros((Np1,Np2))
fe_log=np.zeros((Np1,Np2))

t0 = time.time()
for it1 in range(Np1):
    print('label=', P1[it1])
    for it2 in range(Np2):
        hx1 = P1[it1]
        hx2 = P2[it2]
        #load_path = '..\\..\\..\\data_dmrg\\QTSEN\\' + modelpath + '\\data_pr\\L%g_chi%g\\' % (N, chi)
        load_exp1 = chainname + 'N%d_J(%g,%g)_h(%g,%g)_chi%d' % (N, Jxy, Jz, hx1, hz, chi) + bounc + '.pr'
        load_exp2 = chainname + 'N%d_J(%g,%g)_h(%g,%g)_chi%d' % (N, Jxy, Jz, hx2, hz, chi) + bounc + '.pr'
        datap1 = bf.load_pr(load_exp1)
        datap2 = bf.load_pr(load_exp2)
        Ap1 = datap1['A']
        Ap2 = datap2['A']
        f_log, f = MPSClass.ln_fidelity_per_site_yy(Ap1.mps, Ap2.mps)
        fe_log[it1, it2] = f_log
        fe[it1, it2] = f

print('the cost of time:', time.time()-t0)
#save_path='..\\..\\..\\data_dmrg\\QTSEN\\' +  modelpath + '\\data_fidelity_matrix\\'
save_exp='fidelitymatrix_N%d_chi%d_hx_(%g,%g,%g)'%(N,chi,P[0],P[-1],Np)+'.npz'
#mkdir(save_path)
np.savez(save_exp,fe=fe,fe_log=fe_log,lable=lable,P1=P1,P2=P2)