# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u, s, v = np.linalg.svd(I, full_matrices=False)
    s[3:] = 0.

    B = v[:3]
    Lt = u[:, :3] * s[:3]
    L = Lt.T
    # import pdb
    # pdb.set_trace()
    L = L / np.linalg.norm(L, axis=0)
    return B, L


if __name__ == "__main__":

    # Put your main code here
    I, L_gt, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    # q2.b
    # albedos, normals = estimateAlbedosNormals(B)
    # albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    # plt.imshow(albedoIm, cmap='gray')
    # # # plt.show()
    # plt.savefig('../out/q2b_albedo.png')
    # plt.close()
    # plt.imshow(normalIm, cmap='rainbow')
    # # # plt.show()
    # plt.savefig('../out/q2b_normal.png')
    # plt.close()

    # q2.c
    # print(f'Groundtruth Light: {L_gt}')
    # print(f'Estimated Light: {L}')

    # q2.d
    # albedos, normals = estimateAlbedosNormals(B)
    # surface = estimateShape(normals, s)
    # plotSurface(surface)

    # q2.e
    # albedos, normals = estimateAlbedosNormals(B)
    # normals = enforceIntegrability(normals, s)
    # surface = estimateShape(normals, s)
    # plotSurface(surface)

    # q2.f
    mu = 0
    v = 0
    lambd = 2
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, v, lambd]])
    G = np.linalg.inv(G)
    G = G.T
    print(G)
    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    normals = G @ normals

    surface = estimateShape(normals, s)
    # surface = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))
    # surface = surface * 100
    # surface = surface / 100
    plotSurface(surface)




