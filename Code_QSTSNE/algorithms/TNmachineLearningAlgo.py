from library import Parameters as pm, BasicFunctions as bf, TNmachineLearning
from library.MPSClass import ln_fidelity_per_site
import numpy as np
from library.BasicFunctions import save_pr, load_pr
import os


def gcmpm_one_class(para=None):
    if para is None:
        para = pm.parameters_gcmpm_one_class()
    para['save_exp'] = save_exp_gcmpm_one_class(para)
    if para['parallel'] is True:
        par_pool = para['n_nodes']
    else:
        par_pool = None
    if para['if_load'] and os.path.isfile(para['save_exp']):
        a = bf.load_pr(os.path.join(para['data_path'], para['save_exp']), 'a')
    else:
        a = TNmachineLearning.MachineLearningMPS(para['d'], para['chi'], para['dataset'],
                                                 par_pool=par_pool)
    a.images2vecs([para['class']], [100])
    a.initialize_virtual_vecs_train()
    a.update_virtual_vecs_train('all', 'all', 'both')
    a.mps.correct_orthogonal_center(0, normalize=True)
    a.mps.mps[0] /= np.linalg.norm(a.mps.mps[0].reshape(-1, ))
    mps0 = a.mps.mps.copy()
    for t in range(0, para['sweep_time']):
        # from left to right
        if para['if_print_detail']:
            print('At the ' + str(t) + '-th sweep, from left to right')
        for nt in range(0, a.length):
            a.update_tensor_gradient(nt, para['step'])
            if nt != a.length-1:
                a.update_virtual_vecs_train('all', nt, 'left')
        # from left to right
        print('At the ' + str(t) + '-th sweep, from right to left')
        for nt in range(a.length-1, -1, -1):
            a.update_tensor_gradient(nt, para['step'])
            if nt != 0:
                a.update_virtual_vecs_train('all', nt, 'right')
        if t > para['check_time0'] and ((t+1) % para['check_time'] == 0
                                        or t+1 == para['sweep_time']):
            fid = ln_fidelity_per_site(mps0, a.mps.mps)
            if fid < (para['step'] * para['ratio_step_tol']):
                print('After ' + str(t+1) + ' sweeps: fid = %g' % fid)
                para['step'] *= para['step_ratio']
            elif t+1 == para['sweep_time']:
                print('After all ' + str(t+1) + ' sweeps finished, fid = %g. '
                                                'Consider to increase the sweep times.' % fid)
            else:
                print('After ' + str(t+1) + ' sweeps, fid = %g.' % fid)
                mps0 = a.mps.mps.copy()
            if para['step'] < para['step_min']:
                print('Now step = ' + str(para['step']) + ' is sufficiently small. Break the loop')
                break
            else:
                print('Now step = ' + str(para['step']))
    if para['if_save']:
        save_pr(para['data_path'], para['save_exp'], [a, para], ['a', 'para'])
    return a, para


def gcmpm(para_tot=None):
    print('Preparing parameters')
    if para_tot is None:
        para_tot = pm.parameters_gcmpm()
    n_class = len(para_tot['classes'])
    paras = bf.empty_list(n_class)
    for n in range(0, n_class):
        paras[n] = para_tot.copy()
        paras[n]['class'] = para_tot['classes'][n]
        paras[n]['chi'] = para_tot['chi'][n]
        paras[n]['save_exp'] = save_exp_gcmpm_one_class(paras[n])
    classifiers = bf.empty_list(n_class)
    for n in range(0, n_class):
        data = '../data_tnml/gcmpm/' + paras[n]['save_exp']
        if para_tot['if_load'] and os.path.isfile(data):
            print('The classifier already exists. Load directly')
            classifiers[n] = load_pr(data, 'classifier')
        else:
            print('Training the MPS of ' + str(para_tot['classes'][n]))
            classifiers[n] = gcmpm_one_class(paras[n])[0]
            if para_tot['if_save']:
                save_pr('../data_tnml/gcmpm/', paras[n]['save_exp'],
                        [classifiers[n]], ['classifier'])
    # Testing accuracy
    print('Calculating the testing accuracy')
    labels = para_tot['classes']
    b = TNmachineLearning.MachineLearningFeatureMap('MNIST', para_tot['d'],
                                                    file_sample='t10k-images.idx3-ubyte',
                                                    file_label='t10k-labels.idx1-ubyte')
    b.images2vecs(para_tot['classes'], ['all', 'all'])
    fid = np.zeros((n_class, ))
    num_wrong = 0
    for ni in range(0, b.numVecSample):
        for n in range(0, n_class):
            fid[n] = b.fidelity_mps_image(classifiers[n].mps.mps, ni)
        n_max = int(np.argmax(fid))
        if labels[n_max] != b.LabelNow[ni]:
            num_wrong += 1
    accuracy = num_wrong/b.numVecSample
    print(accuracy)


def save_exp_gcmpm_one_class(para):
    exp = 'gcmpm' + str(para['class']) + '_dim(' + str(para['d']) + ',' + str(para['chi']) + ')' \
          + para['dataset']
    return exp


def decision_mps(para=None):
    if para is None:
        para = pm.parameters_decision_mps()
    a = TNmachineLearning.DecisionTensorNetwork(para['dataset'], 2, para['chi'], 'mps',
                                                para['classes'], para['numbers'],
                                                if_reducing_samples=para['if_reducing_samples'])
    a.images2vecs_test_samples(para['classes'])
    print(a.vLabel)
    for n in range(0, a.length):
        bf.print_sep()
        print('Calculating the %i-th tensor' % n)
        a.update_tensor_decision_mps_svd(n)
        # a.update_tensor_decision_mps_svd_threshold_algo(n)
        # a.update_tensor_decision_mps_gradient_algo(n)
        a.update_v_ctr_train(n)
        a.calculate_intermediate_accuracy_train(n)
        if a.remaining_samples_train.__len__() == 0:
            print('All samples are classified correctly. Training stopped.')
            break
        print('The current accuracy = %g' % a.intermediate_accuracy_train[n])
        print('Entanglement: ' + str(a.lm[n].reshape(1, -1)))
        if para['if_reducing_samples']:
            print('Number of remaining samples: ' + str(a.remaining_samples_train.__len__()))


