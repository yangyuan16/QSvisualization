import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'Yuan Yang'
    info['email'] = 'yangyuan@mails.ucas.ac.cn'
    info['affiliation'] = 'University of Chinese Academy of Sciences'
    info['Tel']='+86-18810220137'
    return info

def distance_matrix(fidelity, way = 'mps_fidelity'):
    if way == 'mps_fidelity':
        Distance = fidelity
    elif way == 'log_mps_fidelity':
        Distance = np.abs(fidelity)
    elif way == 'euclidean':
        Distance = np.sqrt(2*(1-fidelity))
    return Distance

def image_distance_matrix(D, lengend, file, title='',Issave_fig_D=True):
    fig = plt.figure()
    for it in range(len(D)):
        ax = fig.add_subplot(2,2,it+1)
        ax.imshow(D[it])
        lable_x = lengend[it]
        ax.set_xlabel(lable_x)
        plt.title(title)
    if Issave_fig_D is True:
        plt.savefig(file, dpi=400, bbox_inches='tight')
    plt.show()

def image_distance_matrix_mnist(D, lengend, file, title='',Issave_fig_D=True):
    fig = plt.figure()
    for it in range(len(D)):
        ax = fig.add_subplot(2,3,it+1)
        ax.imshow(D[it])
        lable_x = lengend[it]
        ax.set_xlabel(lable_x)
        plt.title(title)
    if Issave_fig_D is True:
        plt.savefig(file, dpi=400, bbox_inches='tight')
    plt.show()



def fidelity_curve(D, P_label,lengend,title=''):
    fig = plt.figure()
    P_label = np.array(P_label)
    for it in range(len(D)):
        ax = fig.add_subplot(2,2,it+1)
        plt.sca(ax)
        L = plt.plot(P_label[:],D[it][:,0], label=lengend[it],ls='--',lw=1.5,color='black',
                 marker='h',alpha=1,markersize=2,markeredgewidth=3, markeredgecolor='brown',markerfacecolor='w')
        label_x = r'$h_z$'
        legfont = {'family' : 'Times New Roman','weight' : 'normal','size': 12, }###图例字体的大小###ncol 设置列的数量，使显示扁平化，当要表示的线段特别多的时候会有用
        plt.legend(handles=L, loc = 4, bbox_to_anchor=(0.88, 0.65),
        ncol = 1,prop=legfont,markerscale=1,fancybox=None,shadow=None,frameon=False)
        ax.set_xlabel(label_x)
        plt.title(title)
    plt.show()
    return


def fidelity_curve_mnist(D, P_label,lengend,title=''):
    fig = plt.figure()
    P_label = np.array(P_label)
    for it in range(len(D)):
        ax = fig.add_subplot(2,3,it+1)
        plt.sca(ax)
        L = plt.plot(P_label[:],D[it][:,0], label=lengend[it],ls='--',lw=1.5,color='black',
                 marker='h',alpha=1,markersize=2,markeredgewidth=3, markeredgecolor='brown',markerfacecolor='w')
        label_x = r'$h_z$'
        legfont = {'family' : 'Times New Roman','weight' : 'normal','size': 12, }###图例字体的大小###ncol 设置列的数量，使显示扁平化，当要表示的线段特别多的时候会有用
        plt.legend(handles=L, loc = 4, bbox_to_anchor=(0.88, 0.65),
        ncol = 1,prop=legfont,markerscale=1,fancybox=None,shadow=None,frameon=False)
        ax.set_xlabel(label_x)
        plt.title(title)
    plt.show()
    return

#=======================================================================
def plot_in_2d(x, y,legend='fe'):
    # this function is not good, need update
    x_min, x_max = x.min(0), x.max(0)
    x_norm = (x - x_min) / (x_max - x_min)  # 归一化
    plt.figure()
    for i in range(x_norm.shape[0]):
        if y[i]>0:
            c='r'
        elif y[i]==0:
            c='green'
        else:
            c='blue'
        plt.text(x_norm[i,0],x_norm[i,1],str(y[i]),color=c,
                 fontdict={'weight': 'bold', 'size': 9})
    plt.text(0.5,0.1,legend)
    plt.show()

def plot_4subfig(x, y, legend, perplexity,title='HTSNE', colorcut=[0,0] ):
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 2, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut[1]:
                color = 'red'
            elif y[j] < colorcut[0]:
                color = 'blue'
            elif y[j] >= colorcut[0] and y[j] < colorcut[1]:
                color = 'green'
            ax.text(x_norm[j, 0], x_norm[j, 1], str(round(y[j],3)), color=color,
                     fontdict={'weight': 'bold', 'size': 7})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
        plt.title(title)
    plt.show()

def plot_4subfigP4(x, y, legend, perplexity, colorcut=[0,0,0] ):
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 2, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut[2]:
                c = 'red'
            elif y[j] < colorcut[0]:
                c = 'green'
            elif y[j] >= colorcut[0] and y[j] < colorcut[1]:
                c = 'blue'
            elif y[j] >= colorcut[1] and y[j] < colorcut[2]:
                c = 'magenta'
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 7})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()

def plot_1subfigP4(x, y, legend, perplexity, colorcut=[0,0,0] ):
    fig = plt.figure()
    for i in range(1):
        ax = fig.add_subplot(1, 1, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut[2]:
                c = 'red'
                size = 10
            elif y[j] < colorcut[0]:
                c = 'green'
                size = 10
            elif y[j] >= colorcut[0] and y[j] < colorcut[1]:
                c = 'blue'
                size = 10
            elif y[j] >= colorcut[1] and y[j] < colorcut[2]:
                c = 'magenta'
                size = 10
            if y[j] == 100:
                c = 'black'
                size=10
            if y[j] == 0:
                c = 'red'
                size = 14
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': size})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()

#=========================================================================
#=========================================================================
def plot_1subfigP2(x, y, legend, perplexity, colorcut=[0,0,0] ):
    '''
    plot of one figure, P2 means two phase or two clusters
    '''
    fig = plt.figure()
    for i in range(1):
        ax = fig.add_subplot(1, 1, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            #x_norm = x[i]
            if y[j] > colorcut[2]:
                c = 'red'
                size = 10
            elif y[j] < colorcut[0]:
                c = 'green'
                size = 10
            elif y[j] >= colorcut[0] and y[j] < colorcut[1]:
                c = 'blue'
                size = 10
            elif y[j] >= colorcut[1] and y[j] < colorcut[2]:
                c = 'magenta'
                size = 10
            if y[j] == 100:
                c = 'black'
                size=10
            if y[j] == 0:
                c = 'red'
                size = 14
            ax.text(x_norm[j, 0], x_norm[j, 1], str(round(y[j],2)), color=c,
                     fontdict={'weight': 'bold', 'size': size})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()


def plot3d_1subfigP2(x, y, legend, perplexity, colorcut=[0,0,0] ):
    '''
    plot of one figure, P2 means two phase or two clusters
    '''
    fig = plt.figure()
    ax = plt.figure().add_subplot(111, projection='3d')
    for i in range(1):

        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut[2]:
                c = 'red'
                size = 10
                marker = 'o'
            elif y[j] < colorcut[0]:
                c = 'green'
                size = 10
                marker = '^'
            elif y[j] >= colorcut[0] and y[j] < colorcut[1]:
                c = 'blue'
                size = 10
                marker = 's'
            elif y[j] >= colorcut[1] and y[j] < colorcut[2]:
                c = 'magenta'
                size = 10
            if y[j] == 100:
                c = 'black'
                size=10
            if y[j] == 0:
                c = 'red'
                size = 14
            ax.scatter(x_norm[j, 0], x_norm[j, 1], x_norm[j,2], marker= marker, color=c,)
        #ax.text(0.5, 0.1, legend[i],)
        #label_x = 'perplexity=%g' % perplexity
        #ax.set_xlabel(label_x)
    plt.show()

def plot3d_1subfigP2_text(x, y, legend, perplexity, colorcut=[0,0,0] ):
    '''
    plot of one figure, P2 means two phase or two clusters
    '''
    fig = plt.figure()
    ax = plt.figure().add_subplot(111, projection='3d')
    for i in range(1):

        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut[2]:
                c = 'red'
                size = 10
                marker = 'o'
            elif y[j] < colorcut[0]:
                c = 'green'
                size = 10
                marker = '^'
            elif y[j] >= colorcut[0] and y[j] < colorcut[1]:
                c = 'blue'
                size = 10
                marker = 's'
            elif y[j] >= colorcut[1] and y[j] < colorcut[2]:
                c = 'magenta'
                size = 10
            if y[j] == 100:
                c = 'black'
                size=10
            if y[j] == 0:
                c = 'red'
                size = 14
            ax.text(x_norm[j, 0], x_norm[j, 1], x_norm[j,2], str(round(y[j],2)), color=c,)
        #ax.text(0.5, 0.1, legend[i],)
        #label_x = 'perplexity=%g' % perplexity
        #ax.set_xlabel(label_x)
    plt.show()


#==========================================================================
#=================###############==========================================
#==========================================================================
def plot_4subfig_v2(x, y, legend, perplexity, colorcut=[0,0] ):
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 2, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut:
                c = 'r'
            elif y[j] == colorcut:
                c = 'green'
            else:
                if (y[j]*100)%2 == 1:
                    c = 'blue'
                else:
                    c = 'yellow'
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 9})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()

def plot_4subfig_3phase(x, y, legend, perplexity, colorcut=[-1,1]):
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 2, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] >= colorcut[1]:
                c = 'r'
            elif y[j] <= colorcut[0]:
                c = 'green'
            else:
                c = 'blue'
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 9})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()
#========================================================================
def plot_4subfig_mnist(x, y, legend, perplexity, color ):
    # 画的是 fe fe_log fe_eu fe_log_eu 的四种情况
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 2, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            c= color[str(np.int(y[j]))]
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 9})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()

def plot_5subfig_mnist(x, y, legend, perplexity, color ):
    # 画的是 针对Msnist dataset 的五种距离矩阵的情况
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 3, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            c= color[str(np.int(y[j]))]
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 9})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' % perplexity
        ax.set_xlabel(label_x)
    plt.show()

def plot_1subfig_mnist(x, y, legend, perplexity, color ):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_min, x_max = x.min(0), x.max(0)
    x_norm = (x - x_min) / (x_max - x_min)  # 归一化
    for j in range(x.shape[0]):
        c = color[str(np.int(y[j]))]
        print(c)
        ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                fontdict={'weight': 'bold', 'size': 9})
    ax.text(0.5, 0.1, legend)
    label_x = 'perplexity=%g' % perplexity
    ax.set_xlabel(label_x)
    plt.show()

def plot_9subfig_mnist_logfe(Data,label,title,color):
    fig = plt.figure()
    for it in range(9):
        ax = fig.add_subplot(3,3,it+1)
        x_min, x_max = Data[it].min(0), Data[it].max(0)
        Data[it] = (Data[it] - x_min) / (x_max - x_min)
        for it1 in range(len(Data[it])):
            ax.text(Data[it][it1,0],Data[it][it1,1],str(label[it][it1]),color=color[it][it1],
                    fontdict = {'weight':'bold', 'size':9})
        plt.title(title[it], size=8)
    plt.show()

#========================================================================
def plot_6subfig_Ising(x, y, legend, per,pp=0.5):
    # 画的是 'euler', 'cossim_log', 'fe_eu', 'cossim_log_map','fe_log_image','fe_log_mps' 的六种情况
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2, 3, i + 1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] < pp:
                c = 'red'
            else:
                c= 'blue'
            #c= color[str(np.int(y[j]))]
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 9})
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g' %(per)
        ax.set_xlabel(label_x)
    plt.show()

#========================================================================
def plot_4subfig_scatter(x, y, legend, perplexity, title = 'HTSNE',colorcut=[0,0] ):

    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2,2,i+1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            #print(x_norm.shape)
            if y[j] > colorcut[1]:
                color = 'red'
            elif y[j] < colorcut[0]:
                color = 'green'
            else:
                color = 'blue'
            ax.scatter(x_norm[j, 0], x_norm[j, 1], c=color)
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g'%perplexity
        ax.set_xlabel(label_x)
        plt.title(title)
    plt.show()

def plot_4subfig_scatterP4(x, y, legend, perplexity, colorcut=[0,0,0] ):

    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(2,2,i+1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            #print(x_norm.shape)
            if y[j] > colorcut[2]:
                color = 'red'
            elif y[j] < colorcut[0]:
                color = 'green'
            elif y[j]>=colorcut[0] and y[j] <colorcut[1]:
                color = 'blue'
            elif y[j]>=colorcut[1] and y[j] <colorcut[2]:
                color = 'magenta'
            ax.scatter(x_norm[j, 0], x_norm[j, 1], c=color)
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g'%perplexity
        ax.set_xlabel(label_x)
    plt.show()


def plot_1subfig_scatterP4(x, y, legend, perplexity, colorcut=[0,0,0] ):

    fig = plt.figure()
    for i in range(1):
        ax = fig.add_subplot(1,1,i+1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            #print(x_norm.shape)
            if y[j] > colorcut[2]:
                color = 'red'
            elif y[j] < colorcut[0]:
                color = 'green'
            elif y[j]>=colorcut[0] and y[j] <colorcut[1]:
                color = 'blue'
            elif y[j]>=colorcut[1] and y[j] <colorcut[2]:
                color = 'magenta'
            ax.scatter(x_norm[j, 0], x_norm[j, 1], c=color)
        ax.text(0.5, 0.1, legend[i])
        label_x = 'perplexity=%g'%perplexity
        ax.set_xlabel(label_x)
    plt.show()
#=======










def plot_8subfig(x, y, legend, perplexity, niter, colorcut=0 ):
    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(3, 3, i + 1)
        print(x[i])
        print(x[i].dtype)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            # print(x_norm.shape)
            if y[j] > colorcut:
                c = 'r'
            elif y[j] == colorcut:
                c = 'green'
            else:
                c = 'blue'
            ax.text(x_norm[j, 0], x_norm[j, 1], str(y[j]), color=c,
                     fontdict={'weight': 'bold', 'size': 9})
        ax.text(0.5, 0.1, legend)
        label_x = 'p=%g_niter=%g' % (perplexity,niter[i])
        ax.set_xlabel(label_x)
    plt.show()


def plot_8subfig_scatter(x, y, legend, perplexity, niter, colorcut=0 ):

    fig = plt.figure()
    for i in range(len(x)):
        ax = fig.add_subplot(3,3,i+1)
        for j in range(x[i].shape[0]):
            x_min, x_max = x[i].min(0), x[i].max(0)
            x_norm = (x[i] - x_min) / (x_max - x_min)  # 归一化
            #print(x_norm.shape)
            ax.scatter(x_norm[j, 0], x_norm[j, 1], c=plt.cm.Set1(y[j]))
        ax.text(0.5, 0.1, legend)
        label_x = 'p=%g_niter=%g' % (perplexity,niter[i])
        ax.set_xlabel(label_x)
    plt.show()

#=================================================================================
# 画3d的图
def plot_in_3d(x, y,per,legend='fe',colorcut=[-1,1]):
    # this function is not good, need update
    x_min, x_max = x.min(0), x.max(0)
    x_norm = (x - x_min) / (x_max - x_min)  # 归一化
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(x_norm.shape[0]):
        if y[i] >= colorcut[1]:
            c = 'red'
        elif y[i] <= colorcut[0]:
            c = 'green'
        else:
            c = 'blue'
        ax.scatter(x_norm[i,0],x_norm[i,1],x_norm[i,2],c=c)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    title = legend + '_per%g'%per
    plt.title(title,size=14)
    plt.show()


def Variance(A,B):
    # A is the fidelity matrix
    #B is the data in 2d space
    NA=A.shape[0]
    sumA=np.sum(A)
    sigma_h=(NA/(NA-1))*(1-sumA/(NA*NA))

    NB=B.shape[0]
    aveB=sum(B)/NB
    diffB=B-aveB
    sigma_2d=np.sum(diffB*diffB)/NB
    return sigma_h,sigma_2d


