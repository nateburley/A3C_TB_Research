import numpy as np

def transform_h(z, eps=10**-2):
    return (np.sign(z) * (np.sqrt(np.abs(z) + 1.) - 1.)) + (eps * z)


def transform_h_inv(z, eps=10**-2):
    return np.sign(z) * (np.square((np.sqrt(1 + 4 * eps * (np.abs(z) + 1 + eps)) - 1) / (2 * eps)) - 1)

print("Original score: {}\n Transformed score: {}".format(350, transform_h(350 + 0.99 * transform_h_inv(350))))