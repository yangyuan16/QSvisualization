import numpy as np
from library import HamiltonianModule as hm
from library.BasicFunctions import print_error, print_options, print_dict


def common_parameters_dmrg():
    # common parameters for finite-size DMRG
    para = dict()
    para['chi'] = 30  # Virtual bond dimension cut-off
    para['sweep_time'] = 100  # sweep time
    # Fixed parameters
    para['if_print_detail'] = False
    para['tau'] = 1e-4  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-5
    para['break_tol'] = 1e-8  # tolerance for breaking the loop
    para['is_real'] = True
    para['dt_ob'] = 4  # in how many sweeps, observe to check the convergence
    para['ob_position'] = 0  # to check the convergence, chose a position to observe

    para['eigWay'] = 1
    para['isParallel'] = False
    para['isParallelEnvLMR'] = False
    para['is_save_op'] = True
    para['data_path'] = '.\\data_dmrg\\'
    return para


def parameter_dmrg_arbitrary():
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 0
    para['hx'] = 0
    para['hz'] = 0

    # The numbers of the spin operators are defined in op
    # For each row (say the m-th), it means one one-body terms where the site index1[m, 0] is acted by the
    #   operator op[index1[m, 1]]
    # The example below means the sz terms on the sites 0, 1, and 2
    para['index1'] = [
        [0, 6],
        [1, 6],
        [2, 6]
    ]
    para['index1'] = np.array(para['index1'])
    para['coeff1'] = [1, 1, 1]
    para['coeff1'] = np.array(para['coeff1']).reshape(-1, 1)

    # The numbers of the spin operators are defined in op
    # For each row (say the m-th), it means one two-body terms, where the site index1[m, 0] is with the
    #   operator op[index1[m, 2]], and the site index1[m, 1] is with the op[index1[m, 3]]
    # The example below means the sz.sz interactions between the 1st and second, as well as the
    #   second and third spins.
    para['index2'] = [
        [0, 1, 3, 3],
        [1, 2, 3, 3]
    ]
    para['index2'] = np.array(para['index2'])
    para['coeff2'] = [1, 1]
    para['coeff2'] = np.array(para['coeff2']).reshape(-1, 1)
    para['lattice'] = 'arbitrary'
    para['data_exp'] = 'Put_here_your_file_name_to_save_data'
    return para
    # ====================================================================


def parameter_dmrg_chain():
    para = dict()
    para['spin'] = 'half'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 18  # Length of MPS and chain
    para['spin'] = 'half'
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'chain'
    return para
    # ====================================================================

def parameter_dmrg_chain_Ising_stagger_hz():
    para = dict()
    para['lattice'] = 'chain_Ising_stagger_hz'
    para['spin'] = 'half'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 18  # Length of MPS and chain
    para['spin'] = 'half'
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['hz_up'] = 0
    para['hz_down'] = 0
    return para
    # =========================================================



def parameter_dmrg_jigsaw():
    para = dict()
    para['spin'] = 'one'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 21  # Length of MPS; odd for open boundary and even for periodic boundary
    # The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['jxy1'] = 1
    para['jz1'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'jigsaw'
    return para
    # ====================================================================


def parameter_dmrg_zigzag():
    para = dict()
    para['spin'] = 'one'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 21  # Length of MPS; odd for open boundary and even for periodic boundary
    # The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['jxy1'] = 1
    para['jz1'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'zigzag'
    return para
    #=============================================================

def parameter_dmrg_ladder_J1J2J3J4():
    para = dict()
    para['spin'] = 'one'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 20  # Length of MPS; odd for open boundary and even for periodic boundary
    # The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy1'] = 1
    para['jz1'] = 1
    para['jxy2'] = 1
    para['jz2'] = 1
    para['jxy3']=1
    para['jz3']=1
    para['jxy4']=1
    para['jz4']=1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'ladderJ1J2J3J4'
    return para
    #=======================================================


def parameter_dmrg_sawtooth_ladder():
    para = dict()
    para['spin'] = 'one'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 18  # Length of MPS; length/2 = odd for open boundary and even for periodic boundary
    # The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy_nn'] = 1
    para['jz_nn'] = 1
    para['jxy_nnn'] = 1
    para['jz_nnn'] = 1
    para['jxy_rung']=1
    para['jz_rung']=1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'sawtooth_ladder'
    return para
    # ==================================================
def parameter_dmrg_sawtooth_ladder_v2():
    para = dict()
    para['spin'] = 'one'
    para['bound_cond'] = 'open'  # open or periodic
    para['l'] = 18  # Length of MPS; length/2 = odd for open boundary and even for periodic boundary
    # The interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy_nn'] = 1
    para['jz_nn'] = 1
    para['jxy_nnn'] = 1
    para['jz_nnn'] = 1
    para['jxy_rung'] = 1
    para['jz_rung'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'sawtooth_ladder_v2'
    return para
    #=========================================================
def parameter_dmrg_square_yy():
    para = dict()
    para['spin'] = 'half'
    para['bound_cond'] = 'open'  # open or periodic
    para['Lx'] = 4   # 4行
    para['Ly'] = 4   # 5列
    para['l'] = para['Lx'] * para['Ly']  # Length of MPS;

    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'square_yy'
    return para
    #=============================================
def parameter_dmrg_ladder_simple():
    para = dict()
    para['spin'] = 'half'
    para['bound_cond'] = 'open'  # open or
    para['l'] = 20  # Length of MPS;

    para['jxy'] = 1
    para['jz'] = 1
    para['jxy_rung'] = 1
    para['jz_rung'] =1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'ladder_simple'
    return para
    #=============================================
def parameter_dmrg_square():
    para = dict()
    para['bound_cond'] = 'open'
    para['square_width'] = 4  # width of the square lattice
    para['square_height'] = 4  # height of the square lattice
    para['l'] = para['square_width'] * para['square_height']
    para['spin'] = 'half'
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    # in square, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['lattice'] = 'square'
    return para
# ====================================================================


def parameter_dmrg_full():
    para = dict()
    para['spin'] = 'half'
    para['l'] = 6  # Length of MPS and chain
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['bound_cond'] = 'open'
    para['is_pauli'] = True
    para['lattice'] = 'full'
    return para


def parameter_dmrg_long_range():
    para = dict()
    para['spin'] = 'half'
    para['alpha'] = 1
    para['l'] = 6  # Length of MPS and chain
    # in chain, the interactions are assumed to be uniform; if not, use parameter_dmrg_arbitrary instead
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['bound_cond'] = 'open'
    para['is_pauli'] = True
    para['lattice'] = 'longRange'
    return para
    # ====================================================================


def generate_parameters_dmrg(lattice):
    # =======================================================
    # No further changes are needed for these codes
    model = ['chain', 'square', 'arbitrary', 'jigsaw', 'full', 'longRange','chain_Ising_stagger_hz',
             'zigzag', 'ladderJ1J2J3J4','sawtooth_ladder']
    if lattice is 'chain':
        para = parameter_dmrg_chain()
    elif lattice is 'chain_Ising_stagger_hz':
        para = parameter_dmrg_chain_Ising_stagger_hz()
    elif lattice is 'square':
        para = parameter_dmrg_square()
    elif lattice is 'arbitrary':
        para = parameter_dmrg_arbitrary()
    elif lattice is 'jigsaw':
        para = parameter_dmrg_jigsaw()
    elif lattice is 'zigzag':
        para = parameter_dmrg_zigzag()
    elif lattice is 'ladderJ1J2J3J4':
        para = parameter_dmrg_ladder_J1J2J3J4()
    elif lattice is 'sawtooth_ladder':
        para = parameter_dmrg_sawtooth_ladder()
    elif lattice is 'square_yy':
        para = parameter_dmrg_square_yy()
    elif lattice is 'ladder_simple':
        para = parameter_dmrg_ladder_simple()
    elif lattice is 'sawtooth_ladder_v2':
        para = parameter_dmrg_sawtooth_ladder_v2()
    elif lattice is 'full':
        para = parameter_dmrg_full()
    elif lattice is 'longRange':
        para = parameter_dmrg_long_range()
    else:
        para = dict()
        print_error('Wrong input of lattice!')
        print_options(model, welcome='Set lattice as one of the following:\t', quote='\'')
    para1 = common_parameters_dmrg()
    para = dict(para, **para1)  # combine with the common parameters
    para = make_consistent_parameter_dmrg(para)
    return para
# =======================================================


def make_consistent_parameter_dmrg(para):
    if para['lattice'] is 'chain':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'chainN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'chain_Ising_stagger_hz':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['op'].append(-para['hx'] * para['op'][1] - para['hz_up'] * para['op'][3])
        para['op'].append(-para['hx'] * para['op'][1] - para['hz_down'] * para['op'][3])

        para['index1'] = np.zeros((para['l'],2),dtype=int)
        for it in range(0, para['l']):
            para['index1'][it,0] = np.int(it)
            if np.mod(it,2) == 0:
                para['index1'][it,1] = np.int(7)
            elif np.mod(it,2) ==  1:
                para['index1'][it,1] = np.int(8)

        para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'chainN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']

    elif para['lattice'] is 'square':
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_nearest_neighbor_square(
            para['square_width'], para['square_height'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'square' + '(%d,%d)' % (para['square_width'], para['square_height']) + \
                           'N%d_j(%g,%g)_h(%g,%g)_chi%d' % (para['l'], para['jxy'], para['jz'], para['hx'],
                                                            para['hz'], para['chi']) + para['bound_cond']
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'arbitrary':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        # para['coeff1'] = np.array(para['coeff1']).reshape(-1, 1)
        # para['coeff2'] = np.array(para['coeff2']).reshape(-1, 1)
        # para['index1'] = np.array(para['index1'], dtype=int)
        # para['index2'] = np.array(para['index2'], dtype=int)
        para['l'] = max(max(para['index1'][:, 0]), max(para['index2'][:, 0]), max(para['index2'][:, 1])) + 1
        para['positions_h2'] = from_index2_to_positions_h2(para['index2'])
        check_continuity_pos_h2(pos_h2=para['positions_h2'])
    elif para['lattice'] is 'jigsaw':
        op = hm.spin_operators(para['spin'])
        if para['bound_cond'] is 'open':
            if para['l'] % 2 == 0:
                print('Note: for OBC jigsaw, l has to be odd. Auto-change l = %g to %g'
                      % (para['l'], para['l'] + 1))
                para['l'] += 1
        else:
            if para['l'] % 2 == 1:
                print('Note: for PBC jigsaw, l has to be even. Auto-change l = %g to %g'
                      % (para['l'], para['l'] + 1))
                para['l'] += 1
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_jigsaw_1d(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['l'] - (para['bound_cond'] is 'open')):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
        for n in range(para['l'] - (para['bound_cond'] is 'open'), para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy1'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy1'] / 2
            para['coeff2'][n * 3 + 2] = para['jz1']
        para['data_exp'] = 'JigsawN%d_j(%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['jxy1'], para['jz1'], para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']
    elif para['lattice'] is 'zigzag':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2']=hm.positions_zigzag(para['l'],para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['l']):
            para['coeff2'][n * 3] = para['jxy'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy'] / 2
            para['coeff2'][n * 3 + 2] = para['jz']
        for n in range(para['l'] , para['positions_h2'].shape[0]):
            para['coeff2'][n * 3] = para['jxy1'] / 2
            para['coeff2'][n * 3 + 1] = para['jxy1'] / 2
            para['coeff2'][n * 3 + 2] = para['jz1']
        para['data_exp'] = 'ZigzagN%d_j(%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['jxy1'], para['jz1'], para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']

    elif para['lattice'] is 'ladderJ1J2J3J4':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int) #### 指向要加的磁场的位置
        para['positions_h2']=hm.positions_ladderJ1J2J3J4(para['l'],para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))  ###磁场项前面的系数，若要给系统只加部分磁场的话 要看para['coeff1']是否进了哈密顿
        nh1=round(para['l']/2)
        nh2=round((para['l']-2)/2)
        nh3=round(para['l']-2)
        nh4=round((para['l']-2)/2)
        coeff2_1=np.zeros((nh1*3, 1))
        coeff2_2=np.zeros((nh2*3, 1))
        coeff2_3=np.zeros((nh3*3, 1))
        coeff2_4=np.zeros((nh4*3, 1))
        for n in range(0,nh1):
            coeff2_1[n * 3] = para['jxy1']/2
            coeff2_1[n * 3 + 1] = para['jxy1']/2
            coeff2_1[n * 3 + 2] = para['jz1']
        for n in range(0,nh2):
            coeff2_2[n * 3] = para['jxy2']/2
            coeff2_2[n * 3 + 1] = para['jxy2']/2
            coeff2_2[n * 3 + 2] = para['jz2']
        for n in range(0,nh3):
            coeff2_3[n * 3] = para['jxy3']/2
            coeff2_3[n * 3 + 1] = para['jxy3']/2
            coeff2_3[n * 3 + 2] = para['jz3']
        for n in range(0,nh4):
            coeff2_4[n * 3] = para['jxy4']/2
            coeff2_4[n * 3 +1] = para['jxy4']/2
            coeff2_4[n * 3 +2] = para['jz4']
        para['coeff2'] = np.vstack((coeff2_1,coeff2_2,coeff2_3,coeff2_4))
        para['data_exp'] = 'ladderJ1J2J3J4N%d_j(%g,%g,%g,%g,%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
            (para['l'], para['jxy1'], para['jz1'], para['jxy2'], para['jz2'],para['jxy3'], para['jz3'],
             para['jxy4'], para['jz4'],para['hx'],
             para['hz'], para['chi']) + para['bound_cond']
    elif para['lattice'] is 'sawtooth_ladder':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz']*para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)  #### 指向要加的磁场的位置
        para['positions_h2'] = hm.positions_sawtooth_ladder(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))  ###磁场项前面的系数，若要给系统只加部分磁场的话 要看para['coeff1']是否进了哈密顿
        if para['bound_cond'] is 'periodic':
            nh1 = round(para['l'])
            nh2 = round(para['l']/2)
            nh3 = round(para['positions_h2'].shape[0] - nh1 -nh2)
        elif para['bound_cond'] is 'open':
            nh1 = round(para['l']-2)
            nh2 = round(para['l']/2 - 1)
            nh3 = round(para['positions_h2'].shape[0] - nh1 - nh2)
        coeff2_1 = np.zeros((nh1 * 3, 1))
        coeff2_2 = np.zeros((nh2 * 3, 1))
        coeff2_3 = np.zeros((nh3 * 3, 1))
        for n in range(0, nh1):
            coeff2_1[n * 3] = para['jxy_nn'] / 2
            coeff2_1[n * 3 + 1] = para['jxy_nn'] / 2
            coeff2_1[n * 3 + 2] = para['jz_nn']
        for n in range(0, nh2):
            coeff2_2[n * 3] = para['jxy_nnn'] / 2
            coeff2_2[n * 3 + 1] = para['jxy_nnn'] / 2
            coeff2_2[n * 3 + 2] = para['jz_nnn']
        for n in range(0, nh3):
            coeff2_3[n * 3] = para['jxy_rung'] / 2
            coeff2_3[n * 3 + 1] = para['jxy_rung'] / 2
            coeff2_3[n * 3 + 2] = para['jz_rung']
        para['coeff2'] = np.vstack((coeff2_1, coeff2_2, coeff2_3))
        para['data_exp'] = 'sawtooth_ladderN%d_J_(%g,%g,%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
                   (para['l'], para['jxy_nn'], para['jz_nn'], para['jxy_nnn'], para['jz_nnn'],
                    para['jxy_rung'], para['jz_rung'], para['hx'], para['hz'], para['chi']) + para['bound_cond']

    elif para['lattice'] is 'sawtooth_ladder_v2':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)  #### 指向要加的磁场的位置
        para['positions_h2'] = hm.positions_sawtooth_ladder_v2(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))  ###磁场项前面的系数，若要给系统只加部分磁场的话 要看para['coeff1']是否进了哈密顿
        if para['bound_cond'] is 'periodic':
            nh1 = round(para['l'])
            nh2 = round(para['l'] / 2)
            nh3 = round(para['positions_h2'].shape[0] - nh1 - nh2)
        elif para['bound_cond'] is 'open':
            nh1 = round(para['l'] - 2)
            nh2 = round(para['l'] / 2 - 1)
            nh3 = round(para['positions_h2'].shape[0] - nh1 - nh2)
        coeff2_1 = np.zeros((nh1 * 3, 1))
        coeff2_2 = np.zeros((nh2 * 3, 1))
        coeff2_3 = np.zeros((nh3 * 3, 1))
        for n in range(0, nh1):
            coeff2_1[n * 3] = para['jxy_nn'] / 2
            coeff2_1[n * 3 + 1] = para['jxy_nn'] / 2
            coeff2_1[n * 3 + 2] = para['jz_nn']
        for n in range(0, nh2):
            coeff2_2[n * 3] = para['jxy_nnn'] / 2
            coeff2_2[n * 3 + 1] = para['jxy_nnn'] / 2
            coeff2_2[n * 3 + 2] = para['jz_nnn']
        for n in range(0, nh3):
            coeff2_3[n * 3] = para['jxy_rung'] / 2
            coeff2_3[n * 3 + 1] = para['jxy_rung'] / 2
            coeff2_3[n * 3 + 2] = para['jz_rung']
        para['coeff2'] = np.vstack((coeff2_1, coeff2_2, coeff2_3))
        para['data_exp'] = 'sawladderv2_N%d_J_(%g,%g,%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy_nn'], para['jz_nn'], para['jxy_nnn'], para['jz_nnn'],
                            para['jxy_rung'], para['jz_rung'], para['hx'], para['hz'], para['chi']) + para['bound_cond']


    elif para['lattice'] is 'square_yy':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)  #### 指向要加的磁场的位置

        para['positions_h2'] = hm.positions_square_yy(para['Lx'],para['Ly'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))  ###磁场项前面的系数，若要给系统只加部分磁场的话 要看para['coeff1']是否进了哈密顿

        coeff2 = np.zeros((3*para['positions_h2'].shape[0], 1))
        for n in range(para['positions_h2'].shape[0]):
            coeff2[n * 3] = para['jxy'] / 2
            coeff2[n * 3 + 1] = para['jxy'] / 2
            coeff2[n * 3 + 2] = para['jz']
        para['coeff2'] = coeff2
        para['data_exp'] = 'square_yyN(%d,%d)_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['Lx'],para['Ly'],para['jxy'], para['jz'],para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']

    elif para['lattice'] is 'ladder_simple':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)  #### 指向要加的磁场的位置

        para['positions_h2'] = hm.positions_ladder_simple(para['l'], para['bound_cond'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['coeff1'] = np.ones((para['l'], 1))  ###磁场项前面的系数，若要给系统只加部分磁场的话 要看para['coeff1']是否进了哈密顿

        coeff2 = np.zeros((3*para['positions_h2'].shape[0], 1))
        for n in range(para['l']-2):
            coeff2[n * 3] = para['jxy'] / 2
            coeff2[n * 3 + 1] = para['jxy'] / 2
            coeff2[n * 3 + 2] = para['jz']
        for n in range(para['l']-2,para['l']-2+np.int(para['l']/2)):
            coeff2[n * 3] = para['jxy_rung'] / 2
            coeff2[n * 3 + 1] = para['jxy_rung'] / 2
            coeff2[n * 3 + 2] = para['jz_rung']
        para['coeff2'] = coeff2
        para['data_exp'] = 'ladder_simpleN%d_j(%g,%g,%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'],para['jxy'], para['jz'],para['jxy_rung'],para['jz_rung'],para['hx'],
                            para['hz'], para['chi']) + para['bound_cond']

    elif para['lattice'] is 'full':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_fully_connected(para['l'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'fullConnectedN%d_j(%g,%g)_h(%g,%g)_chi%d' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'])
        if para['is_pauli']:
            para['coeff1'] = np.ones((para['l'], 1)) * 2
        else:
            para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            if para['is_pauli']:
                para['coeff2'][n * 3] = para['jxy'] * 2
                para['coeff2'][n * 3 + 1] = para['jxy'] * 2
                para['coeff2'][n * 3 + 2] = para['jz'] * 4
            else:
                para['coeff2'][n * 3] = para['jxy'] / 2
                para['coeff2'][n * 3 + 1] = para['jxy'] / 2
                para['coeff2'][n * 3 + 2] = para['jz']
    elif para['lattice'] is 'longRange':
        op = hm.spin_operators(para['spin'])
        para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
        para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
        para['index1'] = np.mat(np.arange(0, para['l']))
        para['index1'] = np.vstack((para['index1'], 6 * np.ones((1, para['l'])))).T.astype(int)
        para['positions_h2'] = hm.positions_fully_connected(para['l'])
        para['index2'] = hm.interactions_position2full_index_heisenberg_two_body(para['positions_h2'])
        para['data_exp'] = 'longRange' + para['bound_cond'] + 'N%d_j(%g,%g)_h(%g,%g)_chi%d_alpha%g' % \
                           (para['l'], para['jxy'], para['jz'], para['hx'],
                            para['hz'], para['chi'], para['alpha'])
        if para['is_pauli']:
            para['coeff1'] = np.ones((para['l'], 1)) * 2
        else:
            para['coeff1'] = np.ones((para['l'], 1))
        para['coeff2'] = np.zeros((para['positions_h2'].shape[0] * 3, 1))
        for n in range(0, para['positions_h2'].shape[0]):
            if para['bound_cond'] is 'open':
                dist = abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1])
            else:  # periodic
                dist = min(abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1]),
                           para['l'] - abs(para['positions_h2'][n, 0] - para['positions_h2'][n, 1]))
            const = dist**(para['alpha'])
            if para['is_pauli']:
                para['coeff2'][n * 3] = para['jxy'] * 2 / const
                para['coeff2'][n * 3 + 1] = para['jxy'] * 2 / const
                para['coeff2'][n * 3 + 2] = para['jz'] * 4 / const
            else:
                para['coeff2'][n * 3] = para['jxy'] / 2 / const
                para['coeff2'][n * 3 + 1] = para['jxy'] / 2 / const
                para['coeff2'][n * 3 + 2] = para['jz'] / const
    para['d'] = physical_dim_from_spin(para['spin'])
    para['nh'] = para['index2'].shape[0]  # number of two-body interactions
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_finite_tebd_gs(lattice='chain'):
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['l'] = 12
    para['chi'] = 32
    para['bound_cond'] = 'open'

    para['tau0'] = 1e-1
    para['dtau'] = 0.1
    para['taut'] = 3
    para['dt_ob'] = 10
    para['iterate_time'] = 5000

    para['lattice'] = lattice
    para['save_mode'] = 'final'  # 'final': only save the converged result; 'all': save all results
    para['if_break'] = True  # if break with certain tolerance
    para['break_tol'] = 1e-7
    return make_para_consistent_tebd(para)


def make_para_consistent_tebd(para):
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
    para['num_h2'] = para['positions_h2'].shape[0]
    para['d'] = physical_dim_from_spin(para['spin'])
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    para['ob_time'] = int(para['iterate_time'] / para['dt_ob'])
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_infinite_dmrg():
    para = dict()
    para['dmrg_type'] = 'mpo'

    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0

    para['n_site'] = 2
    para['chi'] = 16  # Virtual bond dimension cut-off
    para['sweep_time'] = 1000  # sweep time
    # Fixed parameters
    para['tau'] = 1e-4  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-15
    para['break_tol'] = 2e-10  # tolerance for breaking the loop
    para['is_symme_env'] = False
    para['is_real'] = True
    para['form'] = 'center_ort'
    para['dt_ob'] = 10  # in how many sweeps, observe to check the convergence
    para['d'] = physical_dim_from_spin(para['spin'])
    para['data_path'] = '.\\data_idmrg\\'
    return make_para_consistent_idmrg(para)


def make_para_consistent_idmrg(para):
    if para['dmrg_type'] not in ['mpo', 'white']:
        print('Bad para[\'d,rg_type\']. Set to \'white\'')
        para['dmrg_type'] = 'white'
    if para['dmrg_type'] is 'white':
        print('Warning: dmrg_type==\'white\' only suits for nearest-neighbor chains')
        print('In this mode, self.is_symme_env is set as True')
        para['is_symme_env'] = True
    para['model'] = 'heisenberg'
    para['hamilt_index'] = hm.hamiltonian_indexes(
        para['model'], (para['jxy'], para['jz'], -para['hx']/2, -para['hz']/2))
    return para


# =================================================================
# Parameters of infinite DMRG
def generate_parameters_deep_mps_infinite():
    para = dict()
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0

    para['n_site'] = 2 # n-site DMRG algorithm
    para['chi'] = 8  # Virtual bond dimension cut-off
    para['chib0'] = 4  # dimension cut-off of the uMPO (maximal para['chi'])
    para['chib'] = 4  # Virtual bond dimension for the secondary MPS
    para['d'] = 4  # Physical bond dimension (2-sites in one tensor)
    # Fixed parameters
    para['sweep_time'] = 200  # sweep time
    para['tau'] = 1e-3  # shift to ensure the GS energy has the largest magnitude
    para['eigs_tol'] = 1e-12
    para['break_tol'] = 1e-9  # tolerance for breaking the loop
    para['is_symme_env'] = False
    para['is_real'] = True
    para['dt_ob'] = 5  # in how many sweeps, observe to check the convergence
    para['form'] = 'center_ort'
    para['data_path'] = '.\\data_idmrg\\'

    para['chib0'] = min(para['chi'], para['chib0'])
    return para


# =================================================================
# Parameters of super-orthogonalization of honeycomb model
def generate_parameters_so_honeycomb():
    para = dict()
    para['lattice'] = 'honeycomb0'
    para['state_type'] = 'mixed'
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0

    para['chi'] = 12
    para['so_time'] = 20
    if para['state_type'] is 'pure':
        para['tau'] = [1e-1, 1e-2, 1e-3]
        para['beta'] = 10
        para['tol'] = 1e-7
        para['dt_ob'] = 10
        para['ini_way'] = 'random'
    elif para['state_type'] is 'mixed':
        para['tau'] = 1e-2
        para['beta'] = np.arange(0.1, 1.1, 0.1)
        para['ini_way'] = 'id'

    para['d'] = physical_dim_from_spin(para['spin'])
    if para['state_type'] is 'mixed':
        para['d'] *= 2
    para['if_print'] = True
    para['is_debug'] = False
    para['data_path'] = '.\\data_ipeps\\'
    return para


# =================================================================
# Parameters of tree DMRG of honeycomb lattice (square TN)
def generate_parameters_tre_dmrg_honeycomb_lattice():
    para = dict()
    para['lattice'] = 'honeycomb0'
    para['state_type'] = 'pure'
    para['spin'] = 'half'
    para['jxy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0

    para['chi'] = 12
    para['sweep_time'] = 20
    para['dt_ob'] = 4
    para['tau'] = [1e-1, 1e-2, 1e-3]
    para['tol'] = 1e-7

    para['d'] = physical_dim_from_spin(para['spin'])
    para['if_print'] = True
    para['data_path'] = '.\\data_ipeps\\'
    return para


def parameters_gcmpm():
    para = dict()
    para['dataset'] = 'mnist'
    para['classes'] = [0, 1]
    para['chi'] = [8, 8]
    para['d'] = 2
    para['step'] = 1e-2  # gradient step
    para['step_ratio'] = 0.5  # how the step is reduced
    para['step_min'] = 1e-4  # minimal gradient step
    para['ratio_step_tol'] = 0.2  # Suggested value: 0.2
    # para['break_tol'] = para['step'] * para['step_ratio']
    para['sweep_time'] = 20
    para['check_time0'] = 0
    para['check_time'] = 1
    para['data_path'] = '.\\data_tnml\\gcmpm\\'

    para['parallel'] = False
    para['n_nodes'] = 4

    para['if_save'] = True
    para['if_load'] = True
    para['if_print_detail'] = True
    return para


def parameters_gcmpm_one_class():
    para = dict()
    para['dataset'] = 'mnist'
    para['class'] = 0
    para['d'] = 2
    para['chi'] = 8
    para['step'] = 1e-2  # gradient step
    para['step_ratio'] = 0.5  # how the step is reduced
    para['step_min'] = 1e-4  # minimal gradient step
    para['ratio_step_tol'] = 5
    # para['break_tol'] = para['step'] * para['ratio_step_tol']
    para['sweep_time'] = 20
    para['check_time0'] = 0
    para['check_time'] = 2
    para['break_tol'] = 1e-6
    para['data_path'] = '.\\data_tnml\\gcmpm\\'

    para['parallel'] = False
    para['n_nodes'] = 4

    para['if_save'] = True
    para['if_load'] = True
    para['if_print_detail'] = True
    return para


def parameters_decision_mps():
    para = dict()
    para['dataset'] = 'mnist'
    para['classes'] = [0, 1]
    para['numbers'] = [5, 5]
    para['chi'] = 2
    para['if_reducing_samples'] = False

    para['n_local'] = 1  # how many pixels are input in each tensor
    para['order'] = 'normal'  # 'normal' or 'random'
    return para


# =================================================================
# Parameters of ED algorithms
def parameters_ed_time_evolution(lattice):
    para = dict()
    para['spin'] = 'half'
    para['jx'] = 1
    para['jy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['l'] = 12
    para['tau'] = 1e-2
    para['time'] = 10
    para['dt_ob'] = 0.2
    para['bound_cond'] = 'open'

    para['lattice'] = lattice
    para['task'] = 'TE'  # Time Evolution
    para = make_para_consistent_ed(para)
    return para


def parameters_ed_ground_state(lattice):
    para = dict()
    para['spin'] = 'half'
    para['jx'] = 1
    para['jy'] = 1
    para['jz'] = 1
    para['hx'] = 0
    para['hz'] = 0
    para['l'] = 12
    para['tau'] = 1e-4  # Hamiltonian will be shifted as I-tau*H
    para['bound_cond'] = 'open'

    para['task'] = 'GS'
    para['lattice'] = lattice
    para = make_para_consistent_ed(para)
    return para


def make_para_consistent_ed(para):
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l'], para['bound_cond'])
    para['num_h2'] = para['positions_h2'].shape[0]
    tmp = np.zeros((para['num_h2'], 1), dtype=int)
    para['couplings'] = np.hstack((para['positions_h2'], tmp))
    para['d'] = physical_dim_from_spin(para['spin'])
    para['pos4corr'] = np.hstack((np.zeros((para['l'] - 1, 1), dtype=int), np.arange(
        1, para['l'], dtype=int).reshape(-1, 1)))
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    if para['task'] is not 'GS':
        para['iterate_time'] = int(para['time'] / para['tau'])
    return para


def parameter_qes_by_ed():
    para = generate_parameters_infinite_dmrg()
    para['spin'] = 'half'
    para['jxy'] = 0
    para['jz'] = 1
    para['hx'] = 0.5
    para['hz'] = 0
    para['l_phys'] = 6  # number of physical sites in the bulk

    para['chi'] = 16  # Virtual bond dimension cut-off
    para['if_load_bath'] = True
    para = make_para_consistent_qes(para)
    return para


def make_para_consistent_qes(para):
    para = make_para_consistent_idmrg(para)
    para['phys_sites'] = list(range(1, para['l_phys']+1))
    para['bath_sites'] = [0, para['l_phys']+1]
    para['positions_h2'] = hm.positions_nearest_neighbor_1d(para['l_phys']+2, 'open')
    para['num_h2'] = para['positions_h2'].shape[0]
    tmp = np.zeros((para['num_h2'], 1), dtype=int)
    tmp[0] = 1
    tmp[-1] = 2
    para['couplings'] = np.hstack((para['positions_h2'], tmp))
    para['d'] = physical_dim_from_spin(para['spin'])
    para['pos4corr'] = np.hstack((np.ones((para['l_phys']-1, 1), dtype=int), np.arange(
        2, para['l_phys']+1, dtype=int).reshape(-1, 1)))
    op = hm.spin_operators(para['spin'])
    para['op'] = [op['id'], op['sx'], op['sy'], op['sz'], op['su'], op['sd']]
    para['op'].append(-para['hx'] * para['op'][1] - para['hz'] * para['op'][3])
    para['data_path'] = '../data_qes/results/'
    para['data_exp'] = 'QES_ED_chainL(%d,2)_j(%g,%g)_h(%g,%g)_chi%d' % \
                       (para['l_phys'], para['jxy'], para['jz'], para['hx'],
                        para['hz'], para['chi'])
    para['bath_path'] = '../data_qes/bath/'
    para['bath_exp'] = 'bath_chain_j(%g,%g)_h(%g,%g)_chi%d' % \
                       (para['jxy'], para['jz'], para['hx'],
                        para['hz'], para['chi'])
    if para['dmrg_type'] is 'white':
        para['data_exp'] += '_white'
        para['bath_exp'] += '_white'
    para['data_exp'] += '.pr'
    para['bath_exp'] += '.pr'
    return para


# =================================================================
# Some function used here that need not be modified
def from_index2_to_positions_h2(index2):
    from algorithms.DMRG_anyH import sort_positions
    pos_h2 = index2[:, :2]
    pos_h2 = sort_positions(pos_h2)
    new_pos = pos_h2[0, :].reshape(1, -1)
    for n in range(1, pos_h2.shape[0]):
        if not (pos_h2[n, 0] == new_pos[-1, 0] and pos_h2[n, 1] == new_pos[-1, 1]):
            new_pos = np.vstack((new_pos, pos_h2[n, :]))
    return np.array(new_pos, dtype=int)


def check_continuity_pos_h2(pos_h2):
    p0 = np.min(pos_h2)
    if p0 != 0:
        exit('The numbering of sites should start with 0, not %d. Please revise the numbering.' % p0)
    p1 = np.max(pos_h2)
    missing_number = list()
    for n in range(p0+1, p1):
        if n not in pos_h2:
            missing_number.append(n)
    if missing_number.__len__() > 0:
        print_error('The pos_h2 is expected to contain all numbers from 0 to %d. The following numbers are missing:' % p1)
        print(str(missing_number))
        exit('Please check and revise you numbering')


def show_parameters(para):
    print_dict(para, welcome='The parameters are: \n', style_sep=':\n')


def physical_dim_from_spin(spin):
    if spin is 'half':
        return 2
    elif spin is 'one':
        return 3
    else:
        return False
