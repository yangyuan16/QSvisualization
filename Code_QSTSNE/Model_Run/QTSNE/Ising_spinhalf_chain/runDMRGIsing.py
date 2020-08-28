#%%
import numpy as np
from library import Parameters as pm
from algorithms import DMRG_anyH as dmrg
from library.BasicFunctions import save_pr as save
from library.BasicFunctions import mkdir
from scipy.sparse.linalg import eigsh as eigs

L = 80
chi = 30
jxy = 0
jz = 1
hx = np.arange(0,1,0.02)
hz = 0
eigWay = 1

ini_way = 'artificial' #

if ini_way == 'read_mps':
    load_path = '..\\..\\..\\data_load\\QTSNE\\Ising_spinhalf\\'  # 用于加载数据的路径。
    load_exp = 'chainN12_j(0,1)_h(0.7,0)_chi12periodic'   # 选择从那个基态下开始演化。
    Is_continue = False
time_cut = 0.0001

lattice = 'chain'
para = pm.generate_parameters_dmrg(lattice)
para['spin'] = 'half'
para['bound_cond'] = 'periodic'
para['chi'] = chi
para['l'] = L
para['jxy'] = jxy
para['jz'] = jz
para['hx'] = hx
para['hz'] = hz
para['ini_way'] = ini_way
para['eigWay'] = eigWay
para['data_path'] = '..\\..\\..\\data_dmrg\\QIsing\\'
para['time_cut'] = time_cut
if ini_way == 'read_mps':
    para['load'] = {}
    para['load']['path'] = '..\\..\\..\\data_load\\QTSNE\\Ising_spinhalf\\'
    para['load']['Is_continue'] = Is_continue # 判断是基于未收敛mps继续计算 还是基于同一个基态进行计算。
    para['load']['exp'] = load_exp
    mkdir(para['load']['path'])
else:
    para['load'] = '.'
mkdir(para['data_path'])

para = pm.make_consistent_parameter_dmrg(para)

# Run DMRG
ob, A, info, para = dmrg.dmrg_finite_size(para)
save('.', para['data_exp'] + '.pr', (ob, A, info, para),('ob', 'A', 'info', 'para'))
#save(para['data_path'], para['data_exp'] + '.pr', (ob, A, info, para),('ob', 'A', 'info', 'para'))


