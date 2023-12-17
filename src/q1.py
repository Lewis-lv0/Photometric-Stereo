# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from utils import integrateFrankot
import sys
import cv2
from skimage.color import rgb2xyz
import os
def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    x, y = np.meshgrid(np.arange(res[0]), np.arange(start=res[1]-1, stop=-1, step=-1)) # build meshgrid

    # shift the center of pixel coordinates to 0 -> orthographic -> same x and y for the object
    x = (x - (res[0]/2)) * pxSize
    y = (y - (res[1]/2)) * pxSize
    # shift the coordinate relative to the sphere center
    x, y = x - center[0], y - center[1]
    # get z^2 value (relative to sphere center) 
    z2 = rad ** 2 - x ** 2 - y ** 2
    # mask out the area outside the sphere
    mask = (z2 < 0)
    z2[mask] = 0
    # get z value
    z = np.sqrt(z2) 

    # shift xyz value relative to the origin
    x, y, z = x + center[0], y + center[1], z + center[2]

    # normals
    n = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
    # normalize
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    # fully reflective -> albedo = 1
    image = n @ light
    image = image.reshape(res[1], res[0])
    image[mask] = 0
    return image


def loadData(path = "../data/"):
    """
    Question 1 (c)
    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory
    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    I = None
    L = None
    s = None
    # # read images
    # for i in range(7):
    #     input_path = path + f'input_{i+1}.tif'
    #     img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    #     h, w = img.shape[:2]
    #     if I is None:
    #         I = np.zeros((7, h*w))
    #     if s is None:
    #         s = (h, w)
    #     img = rgb2xyz(img) # RGB -> XYZ
    #     luminance = img[:, :, 1] # luminance channel
    #     I[i,] = luminance.reshape(-1)

    # # lighting directions
    # L = np.load(path + 'sources.npy')
    # L = L.T
    # return I, L, s
    for i in range(1, 8):
        img_path = os.path.join(path, f"input_{i}.tif")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # cv2.imshow(' ', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_xyz = rgb2xyz(img)
        luminance = img_xyz[:, :, 1]
        if I is None:
            s = luminance.shape
            I = np.zeros((7, luminance.size))

        I[i - 1, :] = luminance.flatten()

    sources_path = os.path.join(path, "sources.npy")
    # import pdb; pdb.set_trace()
    L = np.load(sources_path).T
    return I, L, s



def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.inv(L @ L.T) @ L @ I
    # B = np.linalg.lstsq(L.T, I, rcond=None)[0]
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    assert B.shape[0] == 3
    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    # albedo values should be in range [0, 1]
    albedoIm = (albedoIm - np.min(albedoIm)) / (np.max(albedoIm) - np.min(albedoIm))
    normalIm = normals.T.reshape(s[0], s[1], 3)
    # shift range from [-1, 1] -> [0, 1]
    normalIm = (normalIm + 1) / 2
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    dx = normals[0] / -normals[2]
    dy = normals[1] / -normals[2]
    dx, dy = dx.reshape(s), dy.reshape(s)
    surface = integrateFrankot(dx, dy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    h, w = surface.shape

    x, y = np.linspace(0, w - 1, w), np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(x, y, surface, cmap='coolwarm')
    plt.show()
    # plt.savefig('../out/q2e.png')
    plt.close()


if __name__ == '__main__':
    # Put your main code here
    
    # Q1.b
    # center = np.array([0, 0, 0])
    # rad = 7.5
    # lights = np.array([[1, 1, 1] / np.sqrt(3), [1, -1, 1] / np.sqrt(3), [-1, -1, 1] / np.sqrt(3)])
    # pxSize = 7e-3
    # res = np.array([3840, 2160])
    # for i in range(len(lights)):
    #     image = renderNDotLSphere(center, rad, lights[i], pxSize, res)
    #     plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    #     plt.savefig(f'../out/render_result_{i}.png')
 

    
    # Q1.c Q1.d
    I, L, s = loadData()

    # U, S, Vh = np.linalg.svd(I, full_matrices=False)
    # print(S)
    # [72.40617702 12.00738171  8.42621836  2.23003141  1.51029184  1.17968677 0.84463311]
    
    
    # Q1.e
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    # albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # plt.imshow(albedoIm, cmap='gray')
    # # plt.show()
    # plt.savefig('../out/albedo.png')
    # plt.close()
    # plt.imshow(normalIm, cmap='rainbow')
    # # plt.show()
    # plt.savefig('../out/normal.png')
    # plt.close()

    # Q1.i
    surface = estimateShape(normals, s)
    plotSurface(surface)
