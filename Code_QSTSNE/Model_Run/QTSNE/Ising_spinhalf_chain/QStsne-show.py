import numpy as np
import library.Qtsne as Qtsne
import sklearn.manifold
import library.BasicFunctions as bf
#========================================================================================
# this code used to show the 4 subplots(fe,fe_log,fe_eu,fe_log_eu) with
# fixed perplexity and n_iter
#========================================================================================
N = 80
chi = 30
h1 = 0
h2 = 0.98
nh = 50
Distance_matrix = 'fidelity'
cpoint = [0.5,0.5]
perplexity = [24]
#perplexity = [8,10,12,14,16,18,20,22,24,26,28,30,32,34]
n_iter = 2000
tsne_case = '2d'  #
seeds = 231
#========================================================================================
if Distance_matrix is 'fidelity':
    #load_path = '..\\..\\..\\data_dmrg\\QTSEN\\' + modelpath + '\\data_fidelity_matrix\\'
    load_exp = 'fidelitymatrix_N%d_chi%d_hx_(%g,%g,%g)'%(N,chi,h1,h2,nh)+'.npz'
    r = np.load( load_exp)
    fe = r['fe']
    fe_log = r['fe_log']
    lable = r['lable']
    P1 = r['P1']
    print(lable)
    D_fe_log = Qtsne.distance_matrix(fe_log, 'log_mps_fidelity')
    #===================================================================================
    for it in range(0,len(perplexity)):
        if tsne_case is '2d':
            tsne_fun = sklearn.manifold.TSNE(n_components=2, metric="precomputed", random_state=seeds,
                                             perplexity=perplexity[it], n_iter=n_iter)
            x_2d_fe_log = tsne_fun.fit_transform(D_fe_log)
            x = [x_2d_fe_log, x_2d_fe_log, x_2d_fe_log, x_2d_fe_log]
            lengend = ['fe', 'fe_log', 'fe_eu', 'fe_log_eu']
            Qtsne.plot_4subfig(x, P1, lengend, perplexity[it], colorcut=cpoint)
            Qtsne.plot_4subfig_scatter(x, P1, lengend, perplexity[it],colorcut=cpoint)