import numpy as np
import scipy.sparse.linalg as la
import time
from library.MPSClass import MpsFiniteTEBD
from library.Parameters import generate_parameters_finite_tebd_gs as generate_para
from library.HamiltonianModule import hamiltonian_heisenberg
from library.BasicFunctions import empty_list


def tebd(para=None):
    start = time.time()
    if para is None:
        para = generate_para()
    ob = dict()
    ob['lm'] = list()
    ob['ent'] = list()
    ob['mx'] = list()
    ob['mz'] = list()
    ob['eb'] = list()
    ob['e_site'] = list()
    mps = MpsFiniteTEBD(para['l'], para['d'], para['chi'], para['spin'], evolve_way='gates')
    mps.correct_orthogonal_center(0, True)
    hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                    para['hx']/2, para['hz']/2)
    n_now = 0
    time_now = 0
    e0 = 100  # for checking convergence
    for nt in range(0, para['taut']):
        tau = para['tau0'] * (para['dtau'] ** nt)
        print('Now tau = ' + str(tau))
        u2 = la.expm(-tau * hamilt)
        gates = [[], []]
        gates[0], lm0, gates[1] = np.linalg.svd(u2.reshape(
            para['d'], para['d'], para['d'], para['d']).transpose(0, 2, 1, 3).reshape(
            para['d'] * para['d'], para['d'] * para['d']))
        gates[0] = (gates[0].dot(np.diag(np.sqrt(lm0)))).reshape(
            para['d'], para['d'], para['d'] * para['d']).transpose(0, 2, 1)
        gates[1] = (np.diag(np.sqrt(lm0)).dot(gates[1])).reshape(
            para['d'] * para['d'], para['d'], para['d']).transpose(1, 0, 2)
        for t in range(para['iterate_time']):
            for nh in range(para['num_h2']):
                p1 = para['positions_h2'][nh, 0]
                p2 = para['positions_h2'][nh, 1]
                mps.evolve_gate_tebd(p1, p2, gates)
                mps.truncate_mps_tebd(p1, p2)
            mps.norm_mps(True)  # normalize MPS
            time_now += tau
            if (para['save_mode'] is 'all') and ((t % para['dt_ob']) == 0):
                print('time(beta) = ' + str(time_now))
                ob['lm'].append(mps.lm)
                ob['ent'].append(mps.ent)
                ob['mx'].append(mps.observe_magnetization(para['op'][1]))
                ob['mz'].append(mps.observe_magnetization(para['op'][3]))
                ob['e_site'].append(sum(ob['eb'][n_now]) / para['l'] + sum(
                    ob['mx'][n_now]) * para['hx'] / para['l'] + sum(
                    ob['mz'][n_now]) * para['hz'] / para['l'])
                print('E per site = ' + str(ob['e_site'][n_now]))
                n_now += 1
            elif para['if_break'] is True:
                e1 = sum(mps.observe_bond_energy_from_jxyz(
                    para['positions_h2'], para['jxy'], para['jxy'], para['jz'])) / para['l']
                if (abs(e0 - e1) < para['break_tol'] and ((t % para['dt_ob']) == 0)) \
                        or (t == para['iterate_time']):
                    ob['lm'].append(mps.lm)
                    ob['ent'].append(mps.ent)
                    ob['mx'].append(mps.observe_magnetization(1))
                    ob['mz'].append(mps.observe_magnetization(3))
                    ob['eb'].append(mps.observe_bond_energy_from_jxyz(
                        para['positions_h2'], para['jxy'], para['jxy'], para['jz']))
                    ob['e_site'].append(sum(ob['eb'][n_now]) / para['l'] + sum(ob['mx'][n_now]) *
                                        para['hx'] / para['l'] + sum(ob['mz'][n_now]) * para['hz'] /
                                        para['l'])
                    print('effective beta = ' + str(time_now))
                    print('Converged with E = ' + str(ob['e_site'][n_now]) + '; Convergence  = '
                          + str(abs(e0 - e1)))
                    n_now += 1
                    break
                elif (t % para['dt_ob']) == 0:
                    print('effective beta = ' + str(time_now))
                    print('<ss> = ' + str(e1) + '; Convergence  = ' + str(abs(e0 - e1)))
                    e0 = e1
        print('TEBD finished with ' + str(time.time() - start) + 's')
    return ob, para
