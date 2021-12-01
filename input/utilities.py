import scipy.io


def read_input_file(filepath, filename):
    mat = scipy.io.loadmat(filepath + filename)
    return mat


def save_input_file():
    pass