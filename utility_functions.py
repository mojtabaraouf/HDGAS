import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def create_masks(image, res, rad50, rad100):
    hdu = fits.open(image, memmap=True)
    Map = hdu[0].data
    Map_H2_50 = Map.copy()
    Map_H2_100 = Map.copy()
    Map_H2_300 = Map.copy()
    z, h, w = Map.shape

    mask_50 = create_circular_mask(h, w, center=(res, res), radius=rad50)
    mask_100 = create_circular_mask(h, w, center=(res, res), radius=rad100)
    mask_300 = create_circular_mask(h, w, center=(res, res), radius=500)

    Map_H2_50[-1, ~mask_50] = np.nan # for observations only using [~mask_50]
    Map_H2_100[-1, ~mask_100] = np.nan # for observations only using [~mask_100]
    Map_H2_300[-1, ~mask_300] = np.nan # for observations only using [~mask_300]

    return mask_50, mask_100, mask_300

def radial_profile(data, center):
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def plot_profile(data, mask, jj):
    center = (res, res)
    rad_profile = radial_profile(data, center)

    if jj < 6:
        plt.plot(rad_profile[0:100], '-', c='blue', label='AGN')
    elif jj < 12:
        plt.plot(rad_profile[0:100], '--', c='red', label='NoAGN')
    elif jj < 18:
        plt.plot(rad_profile[0:100], '-', c='grey', label='AGN(equilibrium)')
    else:
        plt.plot(rad_profile[0:100], '--', c='grey', label='NoAGN(equilibrium)')

    plt.xlabel('Radius')
    plt.ylabel('Profile')
    plt.title('Radial Profile')
    plt.legend()
    plt.show()

# fns = [image1,image2] res = 512,512 in simualations
def process_profile(fns, res, rad, rad1):
    for i, fn in enumerate(fns):
        h = fits.getheader(fn)
        hdu_GK = fits.open(fn, memmap=True)
        hdu_GK.info()

        Map_GK = hdu_GK[0].data
        Map_GK_H = Map_GK.copy()
        z, h, w = Map_GK.shape
        print('h:', h, 'w:', w)
        mask_vel = create_circular_mask(h, w, center=(res, res), radius=rad)
        mask_vel1 = create_circular_mask(h, w, center=(res, res), radius=rad1)
        Map_GK_H[-1, ~mask_vel] = np.nan
        masked_vel_GK = Map_GK.copy()
        mean_vel_GK, median_vel_GK, std_vel_GK = sigma_clipped_stats(masked_vel_GK[-1, mask_vel], sigma=3)
        Map_GK_H1 = Map_GK_H
        kernel = Gaussian2DKernel(3, mode='center')
        Map_GK_H1_conv = scipy_convolve(Map_GK_H1[-1, :, :], kernel, mode='same', method='direct')

        plot_profile(Map_GK_H1_conv, mask_vel1, i)
