import numpy as np
from library import BasicFunctions as bf, TensorBasicModule as tm
from library.MPSClass import MpsOpenBoundaryClass as MPS
from multiprocessing.dummy import Pool as ThreadPool
import os.path as path

is_debug = False
is_GPU = False


class MachineLearningBasic:

    def __init__(self, dataset='mnist', data_path='..\\..\\..\\MNIST\\',
                 file_sample='train-images.idx3-ubyte', file_label='train-labels.idx1-ubyte',
                 is_normalize=True, par_pool=None):
        # MNIST files for training: 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte'
        # MNIST files for testing:  't10k-images.idx3-ubyte',  't10k-labels.idx1-ubyte'
        self.dataset = dataset
        self.dataInfo = dict()
        self.images = bf.decode_idx3_ubyte(path.join(data_path, file_sample))
        self.labels = bf.decode_idx1_ubyte(path.join(data_path, file_label))
        self.length = self.images.shape[0]
        if is_normalize:
            self.images /= 256
        self.analyse_dataset()

        self.tmp = None
        self.nPool = par_pool
        self.parPool = None
        if is_debug:
            self.check_consistency()

    def clear_tmp_data(self):
        self.tmp = None

    def open_par_pool(self):
        self.parPool = ThreadPool(self.nPool)

    def analyse_dataset(self):
        # Total number of samples
        self.dataInfo['NumTotalTrain'] = self.labels.__len__()
        # Order the samples and labels
        order = np.argsort(self.labels)
        self.images = bf.sort_vecs(self.images, order, which=1)
        self.labels = np.array(sorted(self.labels))
        # Total number of classes
        self.dataInfo['NumClass'] = int(self.labels[-1] + 1)
        # Detailed information
        self.dataInfo['nStart'] = np.zeros((self.dataInfo['NumClass'], ), dtype=int)
        self.dataInfo['nClassNum'] = np.zeros((self.dataInfo['NumClass'],), dtype=int)
        self.dataInfo['nStart'][0] = 0
        for n in range(1, self.dataInfo['NumClass']):
            x = tm.arg_find_array(self.labels[self.dataInfo['nStart'][n-1] + 1:] == n,
                                  1, 'first')
            self.dataInfo['nClassNum'][n-1] = x + 1
            self.dataInfo['nStart'][n] = x + self.dataInfo['nStart'][n-1] + 1
        self.dataInfo['nClassNum'][-1] = \
            self.dataInfo['NumTotalTrain'] - self.dataInfo['nStart'][-1]

    def show_image(self, n):
        # n can be a number or an image
        if type(n) is int:
            l_side = int(np.sqrt(self.length))
            bf.plot_surf(np.arange(0, l_side), np.arange(0, l_side),
                         self.images[:, n].reshape(l_side, l_side))
        else:
            l_side = n.shape[0]
            bf.plot_surf(np.arange(0, l_side), np.arange(0, l_side), n)

    def report_dataset_info(self):
        print('There are ' + str(self.dataInfo['NumClass']) + ' classes in the dataset')
        print('Total number of samples: ' + str(self.dataInfo['NumTotalTrain']))
        for n in range(0, self.dataInfo['NumClass']):
            print('\t There are ' + str(self.dataInfo['nClassNum'][n]) +
                  ' samples in the ' + str(n) + '-th class')

    def check_consistency(self):
        if self.dataInfo['NumTotalTrain'] != sum(self.dataInfo['nClassNum']):
            bf.print_error('The total number in the dataset is NOT consistent '
                           'with the sum of the samples of all classes')
        for n in range(0, self.dataInfo['NumClass']):
            start = self.dataInfo['nStart'][n]
            end = start + self.dataInfo['nClassNum'][n]
            tmp = self.labels[start:end]
            if not np.prod(tmp == tmp[0]):
                bf.print_error('In the ' + str(tmp[0]) + '-th labels, not all labels are '
                               + str(tmp[0]))
                print(bf.arg_find_array(tmp != tmp[0]))


class MachineLearningFeatureMap(MachineLearningBasic):

    def __init__(self, d, dataset='mnist', data_path='..\\..\\..\\MNIST\\', file_sample='train-images.idx3-ubyte',
                 file_label='train-labels.idx1-ubyte', is_normalize=True, par_pool=None):
        MachineLearningBasic.__init__(self, dataset, data_path=data_path, file_sample=file_sample, file_label=file_label,
                                      is_normalize=is_normalize, par_pool=par_pool)
        self.d = d
        self.vec_classes = list()
        self.vecsImages = np.zeros(0)
        self.LabelNow = np.zeros(0)
        self.vecsLabels = np.zeros(0)
        self.numVecSample = 0

    def multiple_images2vecs(self, theta_max=np.pi/2):
        # Put the data of images in self.tmp!!!
        # The pixels should have been normalized to [0, 1)
        s = self.tmp.shape
        self.tmp *= theta_max
        self.vecsImages = np.zeros((self.d, ) + s)
        for nd in range(1, self.d+1):
            self.vecsImages[nd-1, :, :] = (np.sqrt(tm.combination(self.d-1, nd-1)) * (
                    np.cos(self.tmp)**(self.d-nd)) * (np.sin(self.tmp)**(nd-1)))

    def images2vecs(self, classes, numbers, how='random'):
        self.vec_classes = classes
        num_class = classes.__len__()
        ntot = 0
        if numbers is None:
            numbers = ['all'] * num_class
        for n in range(0, num_class):
            if numbers[n] is 'all':
                ntot += self.dataInfo['nClassNum'][n]
            else:
                ntot += min(numbers[n], self.dataInfo['nClassNum'][n])
        self.numVecSample = ntot
        self.LabelNow = np.zeros((ntot,), dtype=int)
        self.tmp = np.zeros((self.length, ntot))

        n_now = 0
        for n in range(0, num_class):
            if numbers[n] is 'all':
                start = self.dataInfo['nStart'][n]
                end = start + self.dataInfo['nClassNum'][n]
                self.tmp[:, n_now:self.dataInfo['nClassNum'][n]] = self.images[:, start:end]
                self.LabelNow[n_now:self.dataInfo['nClassNum'][n]] = self.labels[start:end]
                n_now += self.dataInfo['nClassNum'][n]
            else:
                n_sample = numbers[n]
                start = self.dataInfo['nStart'][classes[n]]
                if n_sample >= self.dataInfo['nClassNum'][classes[n]]:
                    rand_p = range(start, self.dataInfo['nClassNum'][classes[n]] + start)
                elif how is 'random':
                    rand_p = np.random.permutation(self.dataInfo['nClassNum'][classes[n]])[
                             :n_sample] + start
                elif how is 'first':
                    rand_p = range(start, n_sample + start)
                else:
                    rand_p = range(self.dataInfo['nClassNum'][classes[n]] - n_sample + start,
                                   self.dataInfo['nClassNum'][classes[n]] + start)
                for ns in rand_p:
                    self.tmp[:, n_now] = self.images[:, ns]
                    self.LabelNow[n_now] = self.labels[ns]
                    n_now += 1
        self.multiple_images2vecs()
        self.clear_tmp_data()
        if is_debug and n_now != ntot:
            bf.print_error('In images2vecs_train_samples: total number of vectorized '
                           'images NOT consistent')

    def label2vectors(self, num_channels=None):
        # num_channels is the dimension of the label bonds; it doesn't has to be the number of classes
        num_class = self.vec_classes.__len__()
        if num_channels is None:
            num_channels = num_class
        dn1 = int(num_channels/num_class)
        dn0 = int((num_channels - dn1) / (num_class - 1))
        self.vecsLabels = np.zeros((num_channels, self.numVecSample))
        for n in range(0, self.numVecSample):
            which_c = self.vec_classes.index(self.LabelNow[n])
            n_start = dn0 * which_c
            self.vecsLabels[n_start:n_start + dn1, n] = np.ones((dn1,)) / np.sqrt(dn1)

    def fidelity_mps_image(self, mps_ref, ni):
        # Calculate the fidelity between an MPS and one image
        fid = 0
        length = mps_ref.__len__()
        v0 = np.ones((1, ))
        image = self.vecsImages[ni]
        for n in range(0, self.length):
            v0 = tm.absorb_vectors2tensors(mps_ref[n], (v0, image[:, n]), (0, 1))
            norm = np.linalg.norm(v0)
            v0 /= norm
            fid -= np.log(norm) / length
        return fid


class MachineLearningMPS(MachineLearningFeatureMap):

    def __init__(self, d, chi, dataset='mnist', data_path='..\\..\\..\\MNIST\\',
                 file_sample='train-images.idx3-ubyte', file_label='train-labels.idx1-ubyte',
                 is_normalize=True, par_pool=None, mps=None):
        MachineLearningFeatureMap.__init__(self, d=d, dataset=dataset, data_path=data_path,
                                           file_sample=file_sample, file_label=file_label,
                                           is_normalize=is_normalize, par_pool=par_pool)
        self.chi = chi
        self.vecsLeft = list()
        self.vecsRight = list()
        if mps is None:
            self.mps = MPS(self.length, d, chi, is_eco_dims=True)
        else:
            self.mps = mps

    def initialize_virtual_vecs_train(self):
        self.vecsLeft = bf.empty_list(self.length, list())
        self.vecsRight = bf.empty_list(self.length, list())
        for n in range(0, self.length):
            self.vecsLeft[n] = np.ones((self.mps.virtual_dim[n], self.numVecSample))
            self.vecsRight[n] = np.ones((self.mps.virtual_dim[n+1], self.numVecSample))

    def update_virtual_vecs_train(self, which_t, which_side):
        if (which_side is 'left') or (which_side is 'both'):
            tmp = tm.khatri(self.vecsLeft[which_t], self.vecsImages[:, which_t, :].squeeze())
            self.vecsLeft[which_t + 1] = np.tensordot(
                self.mps.mps[which_t], tmp, ([0, 1], [0, 1]))
        if (which_side is 'right') or (which_side is 'both'):
            tmp = tm.khatri(self.vecsRight[which_t], self.vecsImages[:, which_t, :].squeeze())
            self.vecsRight[which_t - 1] = np.tensordot(
                self.mps.mps[which_t], tmp, ([2, 1], [0, 1]))

    def env_tensor(self, nt, way):
        s = self.mps.mps[nt].shape
        env = tm.khatri(tm.khatri(
            self.vecsLeft[nt], self.vecsImages[:, nt, :].squeeze()).reshape(
            self.mps.virtual_dim[nt] * self.d, self.numVecSample), self.vecsRight[nt])
        if way is 'mera':
            env = env.dot(np.ones((self.numVecSample, )))
        elif way is 'gradient':
            weight = self.mps.mps[nt].reshape(1, -1).dot(env)
            env = env.dot(1 / weight)
        return env.reshape(s)

    def update_tensor_gradient(self, nt, step):
        # for n in self.mps.mps:
        #     print(n)
        # input()
        self.mps.correct_orthogonal_center(nt)
        env = self.env_tensor((nt, 'all', 'gradient'))
        env = tm.normalize_tensor(env)[0]
        # env /= np.linalg.norm(env.reshape(-1, ))
        self.mps.mps[nt] = self.mps.mps[nt] * (1 - step) + step * env.reshape(
            self.mps.mps[nt].shape)
        self.mps.mps[nt] /= np.linalg.norm(self.mps.mps[nt].reshape(-1, ))


class DecisionTensorNetwork(MachineLearningFeatureMap):

    def __init__(self, dataset, d, chi, tn, classes, numbers=None, if_reducing_samples=False, par_pool=None):
        MachineLearningFeatureMap.__init__(self, dataset=dataset, d=d, par_pool=par_pool)
        self.tn = tn  # 'mps' or 'ttn'
        self.classes = classes
        self.num_classes = classes.__len__()
        self.chi = chi
        self.tensors = list()
        self.num_tensor = 0
        self.if_reducing_samples = if_reducing_samples
        self.initialize_decision_tree()

        self.vLabel = [[] for _ in range(0, self.num_classes)]
        self.images2vecs(classes, numbers)
        # self.train_label2vectors(self.chi)
        self.generate_vector_labels()
        self.remaining_samples_train = list(range(0, self.numVecSample))
        self.remaining_samples_test = list(range(0, self.numSampleTest))
        self.v_ctr_train = [np.ones(1) for _ in range(0, self.numVecSample)]
        self.v_ctr_test = [np.ones(1) for _ in range(0, self.numSampleTest)]
        self.lm = [np.zeros(1) for _ in range(0, self.length)]
        self.intermediate_accuracy_train = np.ones((self.length,))
        self.intermediate_accuracy_test = np.ones((self.length,))

    def initialize_decision_tree(self):
        if self.tn is 'mps':
            self.num_tensor = self.length
            self.tensors = [np.zeros(0) for _ in range(0, self.num_tensor)]
        elif self.tn is 'tree':
            pass  # to be added

    def generate_vector_labels(self):
        dn1 = int(self.chi / self.num_classes)
        dn0 = int((self.chi - dn1) / (self.num_classes - 1))
        for n in range(0, self.num_classes):
            v = np.zeros((self.chi, ))
            n_start = dn0 * n
            v[n_start:n_start + dn1] = np.ones((dn1,)) / np.sqrt(dn1)
            self.vLabel[n] = v.copy()

    def update_tensor_decision_mps_svd(self, nt):
        env = 0
        d0 = self.v_ctr_train[0].shape[0]
        d1 = self.vecsImages[0][:, nt].shape[0]
        for n in self.remaining_samples_train:
            env += np.kron(np.kron(self.v_ctr_train[n], self.vecsImages[n][:, nt]),
                           self.vLabel[self.classes.index(self.LabelNow[n])])
        u, self.lm[nt], v = np.linalg.svd((env / np.linalg.norm(env.reshape(-1, ))).reshape(
            d0 * d1, self.chi), full_matrices=False)
        self.tensors[nt] = u.dot(v).reshape([d0, d1, self.chi])
        self.lm[nt] /= np.linalg.norm(self.lm[nt])

    def update_tensor_decision_mps_svd_threshold_algo(self, nt, time_r=5, threshold=0.9):
        self.update_tensor_decision_mps_svd(nt)
        env = 0
        d0 = self.v_ctr_train[0].shape[0]
        d1 = self.vecsImages[0][:, nt].shape[0]
        for t in range(0, time_r):
            for n in self.remaining_samples_train:
                v1 = tm.absorb_vectors2tensors(
                    self.tensors[nt], (self.v_ctr_train[n], self.vecsImages[n][:, nt]), (0, 1))
                norm = np.linalg.norm(v1)
                fid = self.fun_fidelity(v1 / norm)
                fid_now = fid[self.classes.index(self.LabelNow[n])]
                fid = [fid[nn] / fid_now for nn in range(0, self.num_classes)]
                fid.pop(self.classes.index(self.LabelNow[n]))
                if max(fid) > threshold:
                    env += np.kron(np.kron(self.v_ctr_train[n], self.vecsImages[n][:, nt]),
                                   self.vLabel[self.classes.index(self.LabelNow[n])])
            u, self.lm[nt], v = np.linalg.svd((env / np.linalg.norm(env.reshape(-1, ))).reshape(
                d0 * d1, self.chi), full_matrices=False)
            self.tensors[nt] = u.dot(v).reshape([d0, d1, self.chi])
            self.lm[nt] /= np.linalg.norm(self.lm[nt])

    def update_tensor_decision_mps_gradient_algo(self, nt, time_r=5, threshold=0, step=0.2):
        self.update_tensor_decision_mps_svd(nt)
        for t in range(0, time_r):
            d_tensor = np.zeros(self.tensors[nt].shape)
            for n in self.remaining_samples_train:
                v1 = tm.absorb_vectors2tensors(
                    self.tensors[nt], (self.v_ctr_train[n], self.vecsImages[n][:, nt]), (0, 1))
                norm = np.linalg.norm(v1)
                fid = self.fun_fidelity(v1 / norm)
                fid_now = fid[self.classes.index(self.LabelNow[n])]
                fid = [fid[nn] / fid_now for nn in range(0, self.num_classes)]
                fid.pop(self.classes.index(self.LabelNow[n]))
                if max(fid) > threshold:
                    tmp = np.kron(np.kron(self.v_ctr_train[n], self.vecsImages[n][:, nt]),
                                  self.vLabel[self.classes.index(self.LabelNow[n])]) \
                          / (fid_now * norm)
                    d_tensor += tmp.reshape(self.tensors[nt].shape)
            d_tensor -= self.tensors[nt]
            norm = np.linalg.norm(d_tensor.reshape(-1, ))
            if norm > 1e-10:
                d_tensor /= norm
            self.tensors[nt] = self.tensors[nt] + step * d_tensor
            self.tensors[nt] /= np.linalg.norm(self.tensors[nt].reshape(-1, ))

    def update_v_ctr_train(self, nt):
        for n in self.remaining_samples_train:
            self.v_ctr_train[n] = tm.absorb_vectors2tensors(
                self.tensors[nt], (self.v_ctr_train[n], self.vecsImages[n][:, nt]), (0, 1))
            self.v_ctr_train[n] /= np.linalg.norm(self.v_ctr_train[n])

    def update_v_ctr_test(self, nt):
        for n in self.remaining_samples_test:
            self.v_ctr_test[n] = tm.absorb_vectors2tensors(
                self.tensors[nt], (self.v_ctr_test[n], self.vecsTest[:, nt, n]), (0, 1))
            self.v_ctr_test[n] /= np.linalg.norm(self.v_ctr_test[n])

    def accuracy_from_one_v(self, v, ni, which_set):
        fid = self.fun_fidelity(v)
        if which_set is 'train':
            if np.argmax(fid) == self.classes.index(self.LabelNow[ni]):
                fid_m = max(fid)
                fid.remove(fid_m)
                if self.if_reducing_samples and abs(fid_m/max(fid)) > 2:
                    self.remaining_samples_train.remove(ni)
                return 1
            else:
                return 0
        else:
            if np.argmax(fid) == self.classes.index(self.TestLabelNow[ni]):
                return 1
            else:
                return 0

    def calculate_intermediate_accuracy_train(self, nt):
        n_right = 0
        for n in self.remaining_samples_train:
            n_right += self.accuracy_from_one_v(self.v_ctr_train[n], n, 'train')
        self.intermediate_accuracy_train[nt] = n_right / self.remaining_samples_train.__len__()

    def calculate_intermediate_accuracy_test(self, nt):
        n_right = 0
        for n in self.remaining_samples_train:
            n_right += self.accuracy_from_one_v(self.v_ctr_train[n], n, 'train')
        self.intermediate_accuracy_train[nt] = n_right / self.remaining_samples_train.__len__()

    def fun_fidelity(self, v):
        fid = list()
        for nc in range(0, self.num_classes):
            # fid.append(np.linalg.norm(v - self.vLabel[nc]))
            # fid.append(v.reshape(1, -1).dot(self.vLabel[nc])[0])
            fid.append(abs(v.reshape(1, -1).dot(self.vLabel[nc]))[0])
        return fid
