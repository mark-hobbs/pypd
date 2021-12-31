
import scipy.io


def read_input_file(filepath, filename):
    mat = scipy.io.loadmat(filepath + filename, squeeze_me=True)
    return mat


def save_input_file():
    pass


def read_mat_file(filepath, filename):
    """
    Read Matlab file and extract data
    """
    mat = scipy.io.loadmat(filepath + filename, squeeze_me=True)

    # Build particle set
    particles = ParticleSet(mat['undeformedCoordinates'])

    # Build bond set
    bonds = BondSet()

    # Build penetrators

    # Build constitutive model

    # Build solver

