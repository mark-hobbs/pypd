import scipy.io


def read_input_file(filepath, filename):
    mat = scipy.io.loadmat(filepath + filename, squeeze_me=True)
    return mat


def save_input_file():
    pass