import scipy.io

from .particles import ParticleSet
from .bonds import BondSet
from .model import Model


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
    particles = ParticleSet(mat["undeformedCoordinates"], mat["DX"])

    # Build bond set
    bonds = BondSet(particles.nlist)

    # Build penetrators

    # Build constitutive model

    # Build solver / time integrator

    # Build model
    model = Model(particles, bonds)

    return model
