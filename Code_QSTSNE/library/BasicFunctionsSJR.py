# Notes (about some frequently used BIF):
# np.hstack((m1,m2)): 合并两个矩阵，成（d, d1+d2）
# np.vstack((m1,m2))：合并两个矩阵， 成（d1+d2，d）

import os
import re
import pickle
import inspect
import numpy as np
import matplotlib.pyplot as mp
from termcolor import cprint, colored
from math import factorial
import struct


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'S.J. Ran'
    info['email'] = 'ranshiju10@mail.s ucas.ac.cn'
    info['affiliation'] = 'ICFO – The Institute of Photonic Sciences'
    return info


def save_pr(path, file, data, names):
    """
    Save the data as a dict in a file located in the path
    :param path: the location of the saved file
    :param file: the name of the file
    :param data: the data to be saved
    :param names: the names of the data
    Notes: 1. Conventionally, use the suffix \'.pr\'. 2. If the folder does not exist, system will
    automatically create one. 3. use \'load_pr\' to load a .pr file
    Example:
    >>> x = 1
    >>> y = 'good'
    >>> save_pr('.\\test', 'ok.pr', [x, y], ['name1', 'name2'])
      You have a file '.\\test\\ok.pr'
    >>> z = load_pr('.\\test\\ok.pr')
      z = {'name1': 1, 'name2': 'good'}
      type(z) is dict
    """
    mkdir(path)
    # print(os.path.join(path, file))
    s = open(os.path.join(path, file), 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load_pr(path_file, names=None):
    """
    Load the file saved by save_pr as a dict from path
    :param path_file: the path and name of the file
    :param names: the specific names of the data you want to load
    :return  the file you loaded
    Notes: the file you load should be a  \'.pr\' file.
    Example:
        >>> x = 1
        >>> y = 'good'
        >>> z = [1, 2, 3]
        >>> save_pr('.\\test', 'ok.pr', [x, y, z], ['name1', 'name2', 'name3'])
        >>> A = load_pr('.\\test\\ok.pr')
          A = {'name1': 1, 'name2': 'good'}
        >>> y, z = load_pr('\\test\\ok.pr', ['y', 'z'])
          y = 'good'
          z = [1, 2, 3]
    """
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
            elif type(names) is list or type(names) is tuple:
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
            s.close()
            return tuple(data)
    else:
        return False


def mkdir(path):
    """
       Create a folder at your path
       :param path: the path of the folder you wish to create
       :return: the path of folder being created
       Notes: if the folder already exist, it will not create a new one.
    """
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def search_file(path, exp):
    content = os.listdir(path)
    exp = re.compile(exp)
    result = list()
    for x in content:
        if re.match(exp, x):
            result.append(os.path.join(path, x))
    return result


def output_txt(x, filename='data'):
    np.savetxt(filename + '.txt', x)


def sort_list(a, order):
    """
    Return the elements sorted in the given order
    :param a: an iterable object
    :param order: the order of elements you want to sort
    :return: the new list contains only elements in the order
    Example:
        >>> a = [1, 2, 'a', 'b']
        >>> order = [1,3]
        >>> z = sort_list(a, order)
          z = [2, 'b']
    """
    return [a[i] for i in order]


def empty_list(n, content=None):
    """
    Create a list of size n with elements as None or content
    :param n: the size of list
    :param content: the content of all elements
    :return: a size n list with all elements are content
    Example:
        >>> z = empty_list(3)
          z = [None, None, None]
        >>> z = empty_list(4, 'a')
          z = ['a', 'a', 'a', 'a']
    """
    # make a empty list of size n
    return [content for _ in range(0, n)]


def remove_element_from_list(x, element):
    """
    Remove an element from a list
    :param x: a list
    :param element: an element to be removed
    :return: a list without 'element'
    Example:
       >>>x = [1, 2, 3]
       >>>print(arg_find_list(x, 3))
         [1, 2]
    """
    return list(filter(lambda a: a != element, x))


def arg_find_array(arg, n=1, which='first'):
    """
    Find the position of positions of elements required
    :param arg:  requirement of the elements needed to fulfil
    :param n:  number of how many elements you need
    :param which:  the first n elements or last elements
    :return:  the position of elements you need
    Notes: 1.arg should be boolean type, 2. if can't find n elements to suit your need, it will return all it can find
    Example:
        >>> x = np.array([-1, 2, -3])
        >>> z = arg_find_array(x < 0, 1, 'last')
          z = 2
    """
    x = np.nonzero(arg)
    length = x[0].size
    if length == 0:
        y = np.zeros(0)
    else:
        num = min(length, n)
        dim = arg.ndim
        if dim > 1 and (not (dim == 2 and arg.shape[1] == 1)):
            y = np.zeros((dim, num), dtype=int)
            if which == 'last':
                for i in range(0, dim):
                    y[i, :] = x[i][length-num:length]
            else:
                for i in range(0, dim):
                    y[i, :] = x[i][:num]
        else:
            if which == 'last':
                y = x[0][length - num:length]
            else:
                y = x[0][:num]
            if n == 1:
                y = y[0]
    return y


def arg_find_list(x, target, n=1, which='first'):
    """
    Find the position of target elements in list
    :param x: a list
    :param target: target element
    :param n: how much elements needed to be find
    :param which: first or last
    :return: position
    Example:
       >>>x = [1, 2, 1, 3]
       >>>print(arg_find_list(x, 3, which='last'))
         [2]
    """
    # x should be a list or tuple (of course '1D')
    # for array or ndarray, please use arg_find_array
    n_found = 0
    n_start = 0
    ind = list()
    if which is 'last':
        x = x[::-1]
    for i in range(0, n):
        try:
            new_ind = x.index(target, n_start)
        except ValueError:
            break
        else:
            ind.append(new_ind)
            n_found += 1
            n_start = new_ind+1
    if which is 'last':
        length = x.__len__()
        ind = [length - tmp - 1 for tmp in ind]
    return ind


def sort_vecs(mat, order, which):
    s = mat.shape
    mat1 = np.zeros(s)
    if which == 0:  # sort as row vectors
        for n in range(0, s[0]):
            mat1[n, :] = mat[order[n], :]
    else:
        for n in range(0, s[1]):
            mat1[:, n] = mat[:, order[n]]
    return mat1


def arrangement(n, m):
    return factorial(n) / factorial(n-m)


def combination(n, m):
    return arrangement(n, m) / factorial(m)


def generate_indexes(ndim):
    key0 = {'0', '1'}
    key = set()
    if ndim == 1:
        return key0
    else:
        for x1 in key0:
            for x2 in generate_indexes(ndim - 1):
                key.add(x1 + x2)
        return key


def get_z2_indexes(ndim):
    indexes_z2 = set()
    indexes = generate_indexes(ndim)
    for x in indexes:
        x1 = [int(m) for m in x]
        if sum(x1) % 2 == 0:
            indexes_z2.add(x)
    return indexes_z2

# ========================================
# MNIST functions
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    Downloaded from: https://blog.csdn.net/jiede1/article/details/77099326
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images.T


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    Downloaded from: https://blog.csdn.net/jiede1/article/details/77099326
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


# =========================================
# Print or Check functions
def trace_stack(level0=2):
    """
    Print the line and file name where this function is used
    :param level0:  previous level0 level in files
    :return: previous level0 line and file name
    Example
        in fileA.py
        >>> def fucntion1():
        >>>    print(trace_stack(2))
        in fileB.py
        if import fileA
        >>> def function2():
        >>>    fileA.function1()
        in fileC.py
        if import fileB
        >>> def function3():
        >>>    fileB.function2()
        >>>function3()
          in file_path\fileC.py at line 2
    """
    # print the line and file name where this function is used
    info = inspect.stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def print_dict(a, keys=None, welcome='', style_sep=': ', color='white', end='\n'):
    """
    Print dictionary
    :param a: dictionary
    :param keys: names in dictionary
    :param welcome:  front words of dictionary
    :param style_sep:  separator
    :param color: print in what color
    :param end: how to end each line
    :return: what need to be print
    Example:
        >>>A = {'name1': 1, 'name2': 'a'}
        >>>print_dict(A, 'this is an example', '-')
          this is an example
          name1-1
          name2-2
    """
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    cprint(express, color)
    return express


def print_error(string, if_trace_stack=True):
    """
    Print an error
    :param string: error information
    :param if_trace_stack: if need to print file name and line
    Example:
        >>>print_error('error: this is an example', 0)
          error: this is an example
    """
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)


def print_sep(info='', style='=', length=40, color='cyan'):
    """
    Print a separator
    :param info:  information
    :param style:  separator type
    :param length:  total length
    :param color:  color
    Example:
        >>>print_sep('This is an example', '@', '20')
          @@@@@@@@@@ This is an example @@@@@@@@@@
    """
    if info == '':
        cprint(style * (length * 2), color)
    else:
        l_info = info.__len__()
        l_new = length * 2 - 2 - l_info
        dl = l_new % 2
        l_new = int(l_new/2)
        l_new = max(l_new, 0)
        mes = style*max(l_new, 0) + ' ' + info + ' ' + style*((l_new + dl)*(l_new > 0))
        cprint(mes, color)


def print_options(options, start=None, welcome='', style_sep=': ', end='    ', color='cyan', quote=None):
    """
    Print the options
    :param options: possible options
    :param start: options count start with
    :param welcome: explaining of options
    :param style_sep:  separator between counts and options
    :param end: end
    :param color: color
    Example:
        >>>a = ['left', 'right']
        >>>print_options(a, [1, 2], 'Where to go:')
          Where to go:1: left    2: right
    """
    message = welcome
    length = len(options)
    if start is None:
        start = list(range(0, options.__len__()))
    for i in range(0, length):
        if quote is None:
            message += colored(str(start[i]) + style_sep + options[i], color)
        elif type(quote) is str:
            message += colored(str(start[i]) + style_sep + quote + options[i] + quote, color)
        if i < length-1:
            message += end
    print(message)


def input_and_check_type(right_type, name, print_result=True, dict_name='para'):
    """
    Input and check input type
    :param right_type: allowed types for input
    :param name: name of input
    :param print_result: if print out the input
    :param dict_name: dictionary of input belongs to
    :return: input
    Example:
        >>>input_and_check_type(int, 'number',True, 'input')
          Please input the value of number:
        >>> a
          number should be int, please input again:
        >>> 2
          You have set input 'number' = 2
    """
    # right_type should be a tuple
    ok = False
    some_error = True
    while some_error:
        try:
            while not ok:
                value = eval(input('Please input the value of ' + name + ': '))
                if isinstance(value, right_type):
                    ok = True
                else:
                    print(name + ' should be ' + str(right_type) + ', please input again.')
            some_error = False
        except (NameError, ValueError, SyntaxError):
            cprint('The input is illegal, please input again ...', 'magenta')
    if print_result:
        print('You have set ' + colored(dict_name + '[\'' + name + '\'] = ' + str(value), 'cyan'))
    return value


def input_and_check_value(right_value, values_str, names='', dict_name=''):
    """
    Input and check the value of input
    :param right_value:  allowed values of input
    :param values_str: describe of input
    :param names:  name of input
    :param dict_name: dictionary name of input
    :param start_ind: start with 1
    :return: input
    Example:
        >>>input_and_check_value([1, 2, 3], ('one', 'two', 'three'), names='Example', dict_name='Only an')
          Please input your choice:
        >>> 2
          You have set Only an['Example'] = 'two'
    """
    # right_value should be an array
    ok = False
    some_error = True
    while some_error:
        try:
            while not ok:
                value = eval(input('Please input your choice: '))
                if value in right_value:
                    ok = True
                else:
                    print('Input should be ' + colored(str(right_value), 'cyan') + ', please input again: ')
            some_error = False
        except (NameError, ValueError, SyntaxError):
            cprint('The input is illegal, please input again ...', 'magenta')
    ind = right_value.index(value)
    print('You have set ' + colored(dict_name + '[\'' + names + '\'] = \'' + str(values_str[ind]) + '\'', 'cyan'))
    return value


def check_condition(x, cond):
    """
    check if x satisfied condition
    :param x: a variable
    :param cond: a function that return boolean variable
    :return: true or false
    Example:
        >>>y = check_condition(3, lambda x: x > 0)
        >>> print(y)
          True
    """
    from inspect import isfunction
    if not isfunction(cond):
        return False
    try:
        return cond(x)
    except (TypeError, IndexError, ValueError):
        cprint('Wrong input in check_condition')
        return False


def input_and_check_type_multiple_items(right_type0, cond=None, name='your terms', max_len=100,
                                        key_stop=-1, key_clear=-3, is_print=False):
    """
    Input multiple items and check type
    :param right_type0:  allowed input type
    :param cond:  condition of input
    :param name:  name of input
    :param max_len:  maximal number inputs
    :param key_stop:  keyword to end inputs
    :param key_clear:  keyword to clean all inputs
    :param is_print:  if print your inputs
    :return:  all inputs
    Example:
        >>>y = input_and_check_type_multiple_items(int, lambda x: x > 0, 'int',key_stop='stop', key_clear='clean')
          Please input the value of int:
        >>> 2
          Please input the value of int:
        >>> -1
          The input is invalid since it does not satisfy the condition
          Please input the value of int:
        >>> 3
          Please input the value of int:
        >>>'stop'
          You input the key to stop. Input completed.
        >>> print(y)
          {2, 3}
    """
    # cond(inout) is True or False, a function to judge if the input is satisfactory
    if is_print:
        cprint('To finish inputting, input -1', 'cyan')
        cprint('To clear all the inputs and start over, input -3', 'cyan')
    output = set()
    # add the type of the key_stop in the tuple of the right types
    if type(right_type0) is type:
        right_type = {right_type0, type(key_stop), type(key_clear)}
    else:
        if type(right_type0) is tuple:
            right_type0 = set(right_type0)
        right_type = right_type0 | {type(key_stop), type(key_clear)}
    right_type = tuple(right_type)
    not_stop = True
    while not_stop:
        new = input_and_check_type(right_type, name, False)
        if new == key_stop:
            cprint('You input the key to stop. Input completed.', 'cyan')
            not_stop = False
        elif new == key_clear:
            output.clear()
            cprint('You have cleared all the inputs.', 'cyan')
        elif (cond is not None) and (not check_condition(new, cond)):
            cprint('The input is invalid since it does not satisfy the condition', 'magenta')
        elif not isinstance(new, right_type0):
            if is_print:
                cprint('This input is invalid since its type is incorrect (should be %s or stop key)'
                       % str(right_type0), 'magenta')
        elif new in output:
            if is_print:
                cprint('This input is invalid since it already exists', 'magenta')
        else:
            output.add(new)
            if output.__len__() > max_len:
                cprint('Number if items exceeds the maximum. Stop the input', 'magenta')
                not_stop = False
    return output


# =========================================
# Plot functions
def plot_square_lattice(width, height, numbered=False, title='', save_path=None):
    """
    Plot a figure of square lattice
    :param width: width of the square lattice
    :param height:  height of the square lattice
    :param numbered:  if show each each lattice dot a number
    :param title:  title of the figure
    :param save_path:  if save the figure
    Example:
        >>>plot_square_lattice(2, 2)
          show a figure of a 2x2 square lattice
    """
    from HamiltonianModule import positions_nearest_neighbor_square
    pos_1d = np.arange(0, width*height, dtype=int).reshape(height, width)
    index = positions_nearest_neighbor_square(width, height)
    for n in range(0, index.shape[0]):
        pos1 = arg_find_array(pos_1d == index[n, 0])
        pos2 = arg_find_array(pos_1d == index[n, 1])
        mp.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '-ob', markersize=8)
    mp.axis('equal')
    if numbered:
        for w in range(0, width):
            for h in range(0, height):
                mp.text(h+0.06, w-0.06, str(pos_1d[h, w]), horizontalalignment='left',
                        verticalalignment='top', fontsize=15)
    mp.axis('off')
    mp.title(title)
    if save_path is not None:
        mkdir(save_path)
        mp.savefig(os.path.join(save_path, 'square(%d,%d).png' % (width, height)))
    mp.show()


def plot_connections_polar(positions, numbered=False, title='', save_path=None):
    """
    Plot a figure of points on polar coordinate with connections
    :param positions: position of points
    :param numbered: if show each each lattice dot a number
    :param title: title of the figure
    :param save_path: if save the figure
    Example:
        >>>x = np.array([[1, 3], [1, 4], [2, 4]])
        >>>plot_connections_polar(x, True)
          plot a figure with [1, 3] are connected, [1, 4] are connected, [2, 4] are connected
    """
    nb = positions.shape[0]
    ax = mp.subplot(111, projection='polar')
    n_site = np.max(positions) + 1
    theta = np.linspace(0, 2*np.pi, n_site+1)
    x1 = np.zeros((nb, 1))
    x2 = np.zeros((nb, 1))
    for n in range(0, nb):
        x1[n] = theta[positions[n, 0]]
        x2[n] = theta[positions[n, 1]]
        ax.plot([x1[n], x2[n]], [1, 1], '-ob')
    if numbered:
        for n in range(0, n_site):
            mp.text(theta[n]+0.05, 1.1, str(n), horizontalalignment='center',
                    verticalalignment='top', fontsize=15)
    mp.axis('off')
    mp.title(title)
    if type(save_path) is str:
        mkdir(save_path)
        mp.savefig(os.path.join(save_path, 'arbitrary.png'))
    mp.show()
