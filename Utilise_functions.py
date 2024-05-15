import numpy as np
#from numpy import mean
import matplotlib.pyplot as plt
import matplotlib as mpl
#from astropy.visualization import astropy_mpl_style
#from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
#from astropy.wcs import WCS
#from matplotlib.colors import LogNorm
#import pyfits
# import copy
# from astropy.table import Table
# import glob, os
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
# import math as ma
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from matplotlib.cbook import get_sample_data
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib import colors
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
# from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
# from scipy.optimize import curve_fit
# from mpl_toolkits.axes_grid1 import AxesGrid
# from matplotlib.ticker import AutoMinorLocator
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import cm
# minorLocator = AutoMinorLocator()
import pandas as pd
# from astropy.utils.data import get_pkg_data_filename
from scipy import stats
# from matplotlib import cm
visible_ticks = {"top": True, "right": True}
from scipy.stats import gaussian_kde
# from pafit.fit_kinematic_pa import fit_kinematic_pa
# from plotbin.symmetrize_velfield import symmetrize_velfield
# from plotbin.plot_velfield import plot_velfield

# run with ipython2
import yt
import h5py
# from yt.units import kpc
# import unyt
# import plotmedian as pm
from scipy.stats import skew
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from sklearn.cluster import KMeans
# Functions
# -----------------------------------------------------------------------------------------------------------------------------
# Skewness Analysis:
# Compute the skewness of each distribution. Skewness measures the asymmetry of a distribution. A positive skewness indicates a longer tail on the right side of the distribution, while a negative skewness indicates a longer tail on the left side.
# Compare the skewness values of the two distributions. If one distribution has a significantly higher (or lower) skewness value compared to the other, it suggests a difference in the shape and asymmetry between the distributions.
def compute_skewness(data):
    return skew(data)

def compare_skewness(distribution1, distribution2):
    skewness1 = compute_skewness(distribution1)
    skewness2 = compute_skewness(distribution2)

    if skewness1 > skewness2:
        print("Distribution 1 is more positively skewed.")
    elif skewness1 < skewness2:
        print("Distribution 2 is more positively skewed.")
    else:
        print("Both distributions have similar skewness.")

    return skewness1, skewness2
 # Bimodality Analysis:


#from scipy.stats import norm
from sklearn.neighbors import KernelDensity
def plot_kde(data, bins, color, linestyle, label):
    kde = KernelDensity(kernel='gaussian').fit(data[:, np.newaxis])
    x = np.linspace(min(data), max(data), 100)
    log_density = kde.score_samples(x[:, np.newaxis])
    density = np.exp(log_density)
    plt.plot(x, density, color=color, ls=linestyle, label=label)

def compare_distributions_violin(distribution1, distribution2):
    plt.violinplot([distribution1, distribution2], showmedians=True)
    plt.xticks([1, 2], ['Distribution 1', 'Distribution 2'])
    plt.show()

def compare_distributions_qqplot(distribution1, distribution2):
    sorted_data1 = np.sort(distribution1)
    sorted_data2 = np.sort(distribution2)
    quantiles = np.linspace(0, 1, min(len(distribution1), len(distribution2)))
    plt.plot(sorted_data1, sorted_data2, 'o')
    plt.plot([np.min(sorted_data1), np.max(sorted_data1)], [np.min(sorted_data2), np.max(sorted_data2)], 'r--')

# Statistical Tests:
def test_distribution_difference(distribution1, distribution2):
    ks_statistic, ks_p_value = ks_2samp(distribution1, distribution2)
    ad_statistic, ad_critical_values, ad_significance_levels = anderson_ksamp([distribution1, distribution2])

    if ks_p_value < 0.05:
        print("The distributions significantly differ based on the Kolmogorov-Smirnov test.")
    else:
        print("The distributions do not significantly differ based on the Kolmogorov-Smirnov test.")

    if ad_statistic > ad_critical_values[2]:
        print("The distributions significantly differ based on the Anderson-Darling test.")
    else:
        print("The distributions do not significantly differ based on the Anderson-Darling test.")
    return ad_critical_values[2],ad_statistic

def perform_cluster_analysis(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    return labels

limit_H2 = 0.5
limit_CO = 1e-9
limit_CI = 1e-9
limit_CII = 1e-9
limit_H2O = 1e-10
limit_HCO = 1e-12
def PDF_g(abundance1_CO,abundance1_HCO,abundance1_H2O,abundance1_CI,abundance1_CII,abundance1_H2,density1,abundance2_CO,abundance2_HCO,abundance2_H2O,abundance2_CI,abundance2_CII,abundance2_H2,density2,name,snap,rad1,limit=None):
    x1,x2 = -9.5,-1
    y1,y2 = 0,1000

    print('CO:',min(abundance1_CO),'CI',min(abundance1_CI),'H2O',min(abundance1_H2O),'HCO+',min(abundance1_HCO[abundance1_HCO>0]))
    # data1_H2=np.log10(density1[index1]* abundance1_H2[index1]/np.mean(density1[index1]))
    # # index2=(abundance2_H2*2>limit_H2)
    # data2_H2=np.log10(density2[index2]* abundance2_H2[index2]/np.mean(density2[index2]))
    #
    index1=(abundance1_CO*2>limit_CO)
    data1_CO=np.log10(density1[index1]* abundance1_CO[index1]/np.median(density1[index1]))
    index2=(abundance2_CO*2>limit_CO)
    data2_CO=np.log10(density2[index2]* abundance2_CO[index2]/np.median(density2[index2]))

    index1=(abundance1_HCO*2>limit_HCO)
    data1_HCO=np.log10(density1[index1]* abundance1_HCO[index1]/np.median(density1[index1]))
    index2=(abundance2_HCO*2>limit_HCO)
    data2_HCO=np.log10(density2[index2]* abundance2_HCO[index2]/np.median(density2[index2]))

    index1=(abundance1_H2O*2>limit_H2O)
    data1_H2O=np.log10(density1[index1]* abundance1_H2O[index1]/np.median(density1[index1]))
    index2=(abundance2_H2O*2>limit_H2O)
    data2_H2O=np.log10(density2[index2]* abundance2_H2O[index2]/np.median(density2[index2]))

    index1=(abundance1_CI*2>limit_CI)
    data1_CI=np.log10(density1[index1]* abundance1_CI[index1]/np.median(density1[index1]))
    index2=(abundance2_CI*2>limit_CI)
    data2_CI=np.log10(density2[index2]* abundance2_CI[index2]/np.median(density2[index2]))

    index1=(abundance1_CII*2>limit_CII)
    data1_CII=np.log10(density1[index1]* abundance1_CII[index1]/np.median(density1[index1]))
    index2=(abundance2_CII*2>limit_CII)
    data2_CII=np.log10(density2[index2]* abundance2_CII[index2]/np.median(density2[index2]))

    time = snap/10.
    CO_Skew_NoAGN,CO_Skew_AGN = compare_skewness(data1_CO[np.isfinite(data1_CO)], data2_CO[np.isfinite(data2_CO)])
    ad_critical_values_CO,ad_statistic_CO = test_distribution_difference(data1_CO[np.isfinite(data1_CO)], data2_CO[np.isfinite(data2_CO)])

    HCO_Skew_NoAGN,HCO_Skew_AGN = compare_skewness(data1_HCO[np.isfinite(data1_HCO)], data2_HCO[np.isfinite(data2_HCO)])
    ad_critical_values_HCO,ad_statistic_HCO = test_distribution_difference(data1_HCO[np.isfinite(data1_HCO)], data2_HCO[np.isfinite(data2_HCO)])

    H2O_Skew_NoAGN,H2O_Skew_AGN = compare_skewness(data1_H2O[np.isfinite(data1_H2O)], data2_H2O[np.isfinite(data2_H2O)])
    ad_critical_values_H2O,ad_statistic_H2O = test_distribution_difference(data1_H2O[np.isfinite(data1_H2O)], data2_H2O[np.isfinite(data2_H2O)])

    CI_Skew_NoAGN,CI_Skew_AGN = compare_skewness(data1_CI[np.isfinite(data1_CI)], data2_CI[np.isfinite(data2_CI)])
    ad_critical_values_CI,ad_statistic_CI = test_distribution_difference(data1_CI[np.isfinite(data1_CI)], data2_CI[np.isfinite(data2_CI)])

    CII_Skew_NoAGN,CII_Skew_AGN = compare_skewness(data1_CII[np.isfinite(data1_CII)], data2_CII[np.isfinite(data2_CII)])
    ad_critical_values_CII,ad_statistic_CII = test_distribution_difference(data1_CII[np.isfinite(data1_CII)], data2_CII[np.isfinite(data2_CII)])
    plt.figure(figsize=(8,5))
    # plt.hist(data1_H2[np.isfinite(data1_H2)], bins='auto',histtype='step', lw = 2, ls='--',color='grey')
    # plt.hist(data2_H2[np.isfinite(data2_H2)], bins='auto',histtype='step', ls='-', lw = 2,color='grey',label='H2')
    counts, bins,patches = plt.hist(data1_CO[np.isfinite(data1_CO)], bins='auto',histtype='step', lw = 2, ls='--',color='red')

    # plot_kde(data1_CO[np.isfinite(data1_CO)], bins, 'red', '--', 'Fit')

    plt.hist(data2_CO[np.isfinite(data2_CO)], bins='auto',histtype='step', ls='-', lw = 2,color='red',label='CO(Skew:%0.2f,%0.2f)' % (CO_Skew_NoAGN,CO_Skew_AGN))

    # plt.hist(data1_HCO[np.isfinite(data1_HCO)], bins='auto',histtype='step', lw = 2, ls='--',color='green')
    #
    # plt.hist(data2_HCO[np.isfinite(data2_HCO)], bins='auto',histtype='step', ls='-', lw = 2,color='green',label=r'$HCO^+$(Skew:%0.2f,%0.2f)' % (HCO_Skew_NoAGN,HCO_Skew_AGN))
    #
    # plt.hist(data1_H2O[np.isfinite(data1_H2O)], bins='auto',histtype='step', lw = 2, ls='--',color='orange')
    #
    # plt.hist(data2_H2O[np.isfinite(data2_H2O)], bins='auto',histtype='step', ls='-', lw = 2,color='orange',label=r'$H_2O$(Skew:%0.2f,%0.2f)' % (H2O_Skew_NoAGN,H2O_Skew_AGN))

    plt.hist(data1_CI[np.isfinite(data1_CI)], bins='auto',histtype='step', lw = 2, ls='--',color='indigo')

    plt.hist(data2_CI[np.isfinite(data2_CI)], bins='auto',histtype='step', ls='-', lw = 2,color='indigo',label='CI(Skew:%0.2f,%0.2f)' % (CI_Skew_NoAGN,CI_Skew_AGN))

    plt.hist(data1_CII[np.isfinite(data1_CII)], bins='auto',histtype='step', lw = 2, ls='--',color='green')

    plt.hist(data2_CII[np.isfinite(data2_CII)], bins='auto',histtype='step', ls='-', lw = 2,color='green',label='CII(Skew:%0.2f,%0.2f)' % (CII_Skew_NoAGN,CII_Skew_AGN))


    data = np.array([time,CI_Skew_NoAGN,CI_Skew_AGN,CII_Skew_NoAGN,CII_Skew_AGN, CO_Skew_NoAGN,CO_Skew_AGN,HCO_Skew_NoAGN,HCO_Skew_AGN,H2O_Skew_NoAGN,H2O_Skew_AGN])

    xx = [10,11,12]
    yy = [max(counts),max(counts),max(counts)]
    plt.plot(xx,yy,'k-',lw = 2,label='AGN')
    plt.plot(xx,yy,'k--',lw = 2,label='NoAGN')
    #plt.text(-10,max(counts)*0.75, r'$-$   AGN', size = 15)
    #plt.text(-10,max(counts)*0.7, r'$--$ NoAGN', size = 15)
    #plt.text(-12,400, 't = %0d Myr' % time, fontsize=18,color='black', bbox=dict(boxstyle='square', facecolor='linen', alpha=0.8))
    leg = plt.legend(loc='lower right')
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('large')
    plt.ylabel('PDF', size = 20)
    plt.title(r'$\rho_{g}$(r < %02d pc, t = %0d Myr)' % (rad1,time) , size = 20)
    # plt.xlabel(r'log($\rho_{g}/\rho_0$)', size =18)
    plt.xlabel(r'log($n_{i}/n_{H}$)', size =20)
    plt.tick_params(which = 'both',direction = 'in',**visible_ticks)
    plt.tick_params(labelsize=17)

    # plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(50))
    plt.gca().xaxis.set_major_locator(MultipleLocator(3))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))

    plt.xlim(x1, x2)
    # plt.ylim(y1, y2)
    plt.yscale('log')
    plt.subplots_adjust(wspace=0.1,hspace=0.35, bottom=0.13, right=0.96,left=0.123, top=0.93)
    plt.savefig('./plots/PDF/'+name+'_%03d_%0dMyr_Cp_CI_CO.png' %(rad1,time))
    return data

ProtonMass = yt.YTQuantity(1.67e-24, "g")
# UnitDensity_in_cgs = UnitMass_in_g / (UnitLength_in_cm**3)
# Add up volum density to Simulations data
def Density(field, data):
    return (data['gas', 'density']/ProtonMass)


def plot_profile_Obs(csv_file, ID, Iv, dist):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Group the data by ID
    grouped_data = data.groupby(ID)

    # Define colors and legends for each group
    colors = ['blue', 'green', 'red', 'brown']
    marker = ['o','*','d','v']
    legends = ['NGC 1808', 'NGC 3627', 'NGC 4321', 'NGC 7469']

    # Iterate over each group
    i = 0
    for group_name, group_data in grouped_data:
        # Filter data where dist is less than 150
        filtered_data = group_data[group_data[dist] < 300]

        # Extract scatter intensity and distance values
        scatter_intensity = filtered_data[Iv]
        distance = filtered_data[dist]

        # Remove NaN and inf values
        scatter_intensity = scatter_intensity.dropna()
        scatter_intensity = scatter_intensity[~np.isinf(scatter_intensity)]

        # Calculate kernel density estimate (KDE)
        kde = gaussian_kde(scatter_intensity, bw_method='silverman')

        # Get a range of values for smoother plotting
        x = np.linspace(distance.min(), distance.max(), 100)
        y = kde(x)

        # Calculate quartiles
        q25 = np.percentile(scatter_intensity, 25)
        q75 = np.percentile(scatter_intensity, 75)

        # Calculate mean within each bin
        mean_values = []
        for bin_start, bin_end in zip(x[:-1], x[1:]):
            indices = (distance >= bin_start) & (distance < bin_end)
            bin_mean = scatter_intensity[indices].mean()
            mean_values.append(bin_mean)
        # yy = np.array(mean_values)
        # Plot scatter intensity vs distance with error bars
        # plt.errorbar(x[:-1], mean_values, yerr=[[mean_values[i]-q25],[q75-mean_values[i]]],
                     # fmt='o', color=colors[i], label=legends[i])
        # plt.plot(bin_cent,ymean, color=colors[i], linestyle='solid', marker='o')
        plt.scatter(x[:-1], mean_values, color=colors[i], marker=marker[i], s=40, label=legends[i], alpha = 0.99)
        # plt.legend(loc='lower right')
        # plt.yscale("log")
        i += 1


# Profile plot functions
def create_profile_plot(image, xlabel, ylabel,rad1, name, xmin1, xmin2, ymin1, ymin2, snap,leg_loc, model, color='red', xi=None, xj=None, yi=None, yj=None):
    x1, x2, y1, y2 = xmin1, xmin2, ymin1, ymin2
    res = 512
    pixel = 1024

    time = snap/10.
      # Load the data and create a single plot
    h = fits.getheader(image)
    hdu_GK = fits.open(image, memmap=True)
    hdu_GK.info()
    Map_GK = hdu_GK[0].data
    if ((snap==30)&(model=='AGN')&(name=='CII_CO_profile')|(name=='CI_CO_profile')|(name=='CII_CI_profile')):
       Map_GK[Map_GK>1000] = np.mean(Map_GK[Map_GK<1000])
    Map_GK_H = Map_GK.copy()
    z, h, w = Map_GK.shape
    print('h:',h,'w:',w)
    mask_vel = create_circular_mask(h, w, center=(res,res),radius = rad1)
    # mask_vel1 = create_circular_mask(h, w, center=(res,res),radius = rad2)
    Map_GK_H[-1,~mask_vel] = np.nan
    masked_vel_GK = Map_GK.copy()
    mean_vel_GK, median_vel_GK, std_vel_GK = sigma_clipped_stats(masked_vel_GK[-1,mask_vel],sigma = 3)
    Map_GK_H1 = Map_GK_H
    kernel = Gaussian2DKernel(3,mode = 'center') #'oversample') #'integrate') # 'linear_interp')
    Map_GK_H1_conv = scipy_convolve(Map_GK_H1[-1,:,:], kernel, mode='same', method='direct')

    # Fit the global kinematic position-angle
    x = np.arange(pixel) - res
    y = np.arange(pixel) - res
    # (X,Y) = np.meshgrid(x, y)
    # X,Y = X[mask_vel1],Y[mask_vel1]

    center=(res,res)
    rad_profile = radial_profile(Map_GK_H1_conv, center)

    xx = [1000,1100,1200]
    yy = [10,11,12]
    if model == 'AGN':
        # c1 = next(color1)
        plt.plot(rad_profile[0:300], '-',c=color,label='%0d Myr' % time)
        if snap == 80:
            plt.plot(xx,yy,'k-', label = 'AGN')
            x50 = [50,50,50]
            x100 = [100,100,100]
            y50 = [0,50,20000]
            plt.plot(x50,y50,'k:')
            plt.plot(x100,y50,'k:')
    elif model == 'NoAGN':
        # c2 = next(color2)
        plt.plot(rad_profile[0:300], '--',c=color)
        if snap == 80:
            plt.plot(xx,yy,'k--', label = 'NoAGN')


    plt.ylabel(ylabel,size=15)
    plt.xlabel(xlabel,size=15)
    plt.tick_params(which='both', direction='in', **visible_ticks)
    plt.tick_params(labelsize=14)
    plt.ylim([ymin1, ymin2])
    # plt.yscale("log")
    plt.xlim([xmin1, xmin2])
    color_m ='black'
    # plt.legend(loc=leg_loc)


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

def radial_profile(data, center):
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def deg_to_parsec(cdelt2):
    """
    Converts the pixel scale in degrees per pixel to parsecs per pixel.

    Parameters:
    cdelt2 (float): The pixel scale in the y-direction (degrees/pixel).

    Returns:
    float: The pixel scale in parsecs/pixel.
    """
    # Convert degrees to radians
    deg_to_rad = np.pi / 180

    # Convert radians to parsecs
    rad_to_parsec = 1 / 206265

    # Calculate the pixel scale in parsecs/pixel
    parsec_per_pixel = cdelt2 * deg_to_rad * rad_to_parsec

    return parsec_per_pixel

def calculate_radii_in_parsec(x0, y0, map_width_pixels, map_height_pixels, cdelt1, cdelt2, distance_mpc, inclination_deg):
    """
    Calculates the radii in parsecs for each pixel in the map.
    Mojtaba Raouf
    Parameters:
    x0 (int): The x-coordinate of the central pixel.
    y0 (int): The y-coordinate of the central pixel.
    map_width_pixels (int): The width of the map in pixels.
    map_height_pixels (int): The height of the map in pixels.
    cdelt1 (float): The pixel scale in the x-direction (degrees/pixel).
    cdelt2 (float): The pixel scale in the y-direction (degrees/pixel).
    distance_mpc (float): The distance to the object in Megaparsecs.
    inclination_deg (float): The inclination angle in degrees.

    Returns:
    list: A list of radii in parsecs for each pixel in the map.
    """
    # Convert inclination from degrees to radians
    inclination_rad = np.deg2rad(inclination_deg)

    # Initialize list to store radii in parsecs
    radii_in_parsec = []

    # Loop through each pixel in the map
    for i in range(map_width_pixels):
        for j in range(map_height_pixels):
            # Calculate radial distance from the center using pixel scale (cdelt1 and cdelt2)
            r = np.sqrt((i - x0)**2 + (1 / np.cos(inclination_rad) * (j - y0))**2) * cdelt2

            # Convert radius from degrees to radians
            r_rad = np.deg2rad(r)

            # Convert radius from radians to parsecs
            radius_in_parsec = r_rad * distance_mpc * 1000000

            # Append to radii list
            radii_in_parsec.append(radius_in_parsec)

    return radii_in_parsec


def runningmedian(x,y,xlolim=-1.e20,ylolim=-1.e20,bins=10,stat='median'):
        xp = x[(x>xlolim)&(y>ylolim)]
        yp = y[(x>xlolim)&(y>ylolim)]
        if bins < 0:	# bins<0 sets bins such that there are equal numbers per bin
            bin_edges = histedges_equalN(xp,-bins)
            bin_means, bin_edges, binnumber = stats.binned_statistic(xp,yp,bins=bin_edges,statistic=stat)
        else:
            bin_means, bin_edges, binnumber = stats.binned_statistic(xp,yp,bins=bins,statistic=stat)
        bin_cent = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ymed = []
        ymean = []
        ysigma = []
        for i in range(0,len(bin_edges[:-1])):
                xsub = xp[xp>bin_edges[i]]
                ysub = yp[xp>bin_edges[i]]
                ysub = ysub[xsub<bin_edges[i+1]]
                ymed.append(np.median(10**ysub))
                ymean.append(np.mean(10**ysub))
                ysigma.append(np.std(10**ysub))
        if stat=='median': ymean = np.asarray(ymed)
        else: ymean = np.asarray(ymean)
        ysiglo = np.maximum(ymean-ysigma,ymean*0.1)
        ysiglo = np.log10(ymean)-np.log10(ysiglo)
        ysighi = np.log10(ymean+ysigma)-np.log10(ymean)
        ymean = np.log10(ymean)
        return bin_cent,ymean,ysiglo,ysighi

def plot_equation(x1,x2,a,b,ls=None,label=None):
    X = np.linspace(10**x1, 10**x2,1000)  # Generate 100 evenly spaced values for X between 0 and 10
    Y = b * X ** a  # Compute Y values using the equation Y = X ^ 1.45
    if label is None:
        plt.plot(np.log10(X), np.log10(Y),ls='-', color='k',label='Obs')  # Plot the X and Y values
    else:
        plt.plot(np.log10(X), np.log10(Y),ls=ls, color ='k',label=label)  # Plot the X and Y values


def plot_Intensity_Obs(csv_file, ID, Iv1,Iv2, dist):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Group the data by ID
    grouped_data = data.groupby(ID)

    # Define colors and legends for each group
    colors = ['blue', 'green', 'red', 'brown']
    marker = ['o','*','d','v']
    legends = ['NGC 1808', 'NGC 3627', 'NGC 4321', 'NGC 7469']

    # Iterate over each group
    i = 0
    for group_name, group_data in grouped_data:
        # Filter data where dist is less than 150
        filtered_data = group_data[group_data[dist] < 300]

        # Extract scatter intensity and distance values
        scatter_intensity1 = filtered_data[Iv1]
        scatter_intensity2 = filtered_data[Iv2]
        distance = filtered_data[dist]

        # Remove NaN and inf values
        scatter_intensity1 = scatter_intensity1.dropna()
        scatter_intensity1 = scatter_intensity1[~np.isinf(scatter_intensity1)]
        scatter_intensity2 = scatter_intensity2.dropna()
        scatter_intensity2 = scatter_intensity2[~np.isinf(scatter_intensity2)]
        # Plot scatter intensity vs distance with error bars
        plt.scatter(np.log10(scatter_intensity1), np.log10(scatter_intensity2), color=colors[i], marker=marker[i], s=50, label=legends[i], alpha = 0.99)
        
     
        
        # ax = plt.gca()
        leg2  =plt.legend(loc='lower right',ncol = 4)
        # ax.add_artist(leg2)
        for t in leg2.get_texts():
            t.set_fontsize('x-small')
        # plt.gca().add_artist(leg2)
        # plt.yscale("log")
        i += 1
        # return leg2

#Scatter plot from intensity maps
color1 = iter(cm.rainbow(np.linspace(0, 1, 6)))
color2 = iter(cm.rainbow(np.linspace(0, 1, 6)))
# Mac path
path_Obs_Liu2022 =  '/Users/raouf/Work_space/HDGAS/Obs_Liu2022/'
def create_scatter_intensity(image1, image2, xlabel, ylabel, name, rad50, rad100, xmin1, xmin2, ymin1, ymin2, snap, model, xi=None, xj=None, yi=None, yj=None):

    x1, x2, y1, y2 = xmin1, xmin2, ymin1, ymin2
    res = 512

    hdu_1 = fits.open(image1, memmap=True)
    hdu_2 = fits.open(image2, memmap=True)

    Map_1 = hdu_1[0].data
    Map_H1_50 = Map_1.copy()
    Map_H1_100 = Map_1.copy()
    Map_H1_300 = Map_1.copy()
    z1, h1, w1 = Map_1.shape

    mask_vel1_50 = create_circular_mask(h1, w1, center=(res, res), radius=rad50)
    mask_vel1_100 = create_circular_mask(h1, w1, center=(res, res), radius=rad100)
    mask_vel1_300 = create_circular_mask(h1, w1, center=(res, res), radius=500)
    Map_H1_50[-1, ~mask_vel1_50] = np.nan
    Map_H1_100[-1, ~mask_vel1_100] = np.nan
    Map_H1_300[-1, ~mask_vel1_300] = np.nan

    Map_2 = hdu_2[0].data
    Map_H2_50 = Map_2.copy()
    Map_H2_100 = Map_2.copy()
    Map_H2_300 = Map_2.copy()
    z2, h2, w2 = Map_2.shape
    mask_vel2_50 = create_circular_mask(h2, w2, center=(res, res), radius=rad50)
    mask_vel2_100 = create_circular_mask(h2, w2, center=(res, res), radius=rad100)
    mask_vel2_300 = create_circular_mask(h2, w2, center=(res, res), radius=500)
    Map_H2_50[-1, ~mask_vel2_50] = np.nan
    Map_H2_100[-1, ~mask_vel2_100] = np.nan
    Map_H2_300[-1, ~mask_vel2_300] = np.nan

    flattened_image2_50 = np.log10(Map_H2_50.flatten())
    flattened_image2_100 = np.log10(Map_H2_100.flatten())

    time = snap / 10.
    print(time, name)
    x = np.log10(Map_H1_300.flatten())
    y = np.log10(Map_H2_300.flatten())

    radii_in_parsec = calculate_radii_in_parsec(x0=512, y0=512, map_width_pixels=1024, map_height_pixels=1024, cdelt1=-1.998317230538E-06, cdelt2=1.998317230538E-06, distance_mpc=14, inclination_deg=41)

    if snap == 40:
        plt.scatter(x, y, c=radii_in_parsec, cmap='jet', marker='.', alpha=0.2, s=2)
        cbar = plt.colorbar(label='r [pc]')


    bin_cent,ymean,ysiglo,ysighi = runningmedian(x,y,xlolim=-2,ylolim=-2,bins=20,stat='median')
    # plt.plot(bin_cent,ymean,color = 'red',lw=2,alpha=.99, markersize=1, label='median')
    # Upe = ymean + ysighi
    # Lwo = ymean - ysiglo
    # plt.fill_between(bin_cent, Upe, Lwo, facecolor='grey', alpha=0.4)

    if model == 'AGN':
        c1 = next(color1)
        plt.plot(bin_cent,ymean,color = c1,lw=2,alpha=.99, markersize=1, label='t = %0d Myr' % time)

    elif model == 'NoAGN':
        c2 = next(color2)
        plt.plot(bin_cent,ymean,color = c2,lw=2,ls='-',alpha=.99, markersize=1)

    if xi is not None:
        obsx = [xi, xj]
        obsy = [yi, yj]
        obseq = [0, 0]
        plt.plot(obsx, obsy, 'k:', lw=2)
        plt.plot(obsx, obseq, 'k:', lw=2)

    if name == 'CII_CI':
        plt.text(1.3, -1.5, model, size=20) #, transform=plt.gcf().transFigure)
    elif name == 'CII_CO':
        plt.text(2.3, 0.1, model, size=20) #, transform=plt.gcf().transFigure)
    elif ((name == 'CI_CO')|(name == 'NOAGN_CI_CO')):
        plt.text(2.3, 0.3, model, size=20) #, transform=plt.gcf().transFigure)
        # ------------------------------------------------------------------------
        NGC7469= path_Obs_Liu2022 + 'NGC7469.csv'
        data1 = pd.read_csv(NGC7469)
        CO21 = data1['CO']
        CI_CO21 =  data1['CI_CO']
        plt.plot(np.log10(CO21),np.log10(CI_CO21), 'k--',alpha=0.6)
        plt.text(2.7, -0.3, 'NGC7469', size=10)
        # ------------------------------------------------------------------------
        NGC1808= path_Obs_Liu2022 + 'NGC1808.csv'
        data1 = pd.read_csv(NGC1808)
        CO21 = data1['CO']
        CI_CO21 =  data1['CI_CO']
        plt.plot(np.log10(CO21),np.log10(CI_CO21), 'k--',alpha=0.6)
        plt.text(2.7, -0.7, 'NGC1808', size=10)
        # ------------------------------------------------------------------------
        NGC3627= path_Obs_Liu2022 + 'NGC3627.csv'
        data1 = pd.read_csv(NGC3627)
        CO21 = data1['CO']
        CI_CO21 =  data1['CI_CO']
        plt.plot(np.log10(CO21),np.log10(CI_CO21), 'k--',alpha=0.6)
        plt.text(2.7, -0.9, 'NGC3627', size=10)
        # ------------------------------------------------------------------------
        NGC4321= path_Obs_Liu2022 + 'NGC4321.csv'
        data1 = pd.read_csv(NGC4321)
        CO21 = data1['CO']
        CI_CO21 =  data1['CI_CO']
        plt.plot(np.log10(CO21),np.log10(CI_CO21), 'k--',alpha=0.6)
        plt.text(2.7, -1.2, 'NGC34321', size=10)
    elif name == 'ICI_ICO':
        plt.text(-0.5, 1.5, model, size=20) #, transform=plt.gcf().transFigure)
        if snap == 80:
            plot_equation(-2,4,1.05,1,ls=':',label='SB(Salak+19)')
            plot_equation(-2,4,0.8,0.035,ls='--',label='CND(Bolatto+13)')
        # plot_equation(x1,x2,1.05,lable='Salak+19')
    elif name == 'ICII_ICO':
        plt.text(2.3, -0.5, model, size=20) #, transform=plt.gcf().transFigure)
    elif name == 'ICII_ICI':
        plt.text(-1.5, 1.0, model, size=20) #, transform=plt.gcf().transFigure)
    else:
        plt.text(2.3, 0.5, model, size=20) #, transform=plt.gcf().transFigure)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)

    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.tick_params(which='both', direction='in', **visible_ticks)
    plt.tick_params(labelsize=14)
    leg = plt.legend(loc='upper right', ncol=4)
    for t in leg.get_texts():
        t.set_fontsize('small')
    


#Scatter plot from intensity maps vs NH2
def create_X_NH2_plot(image1, image2, xlabel, ylabel, name, rad50, rad100, xmin1, xmin2, ymin1, ymin2, snap, model, xi=None, xj=None, yi=None, yj=None):
    x1, x2, y1, y2 = xmin1, xmin2, ymin1, ymin2
    res = 512

    # CO & H2 column density
    # -------------------------------------------------------
    # fn_AGN =  '/data2/mojtaba/gizmo-public/output_Disk_0103_AGN/BH3V5B6_CH/data_AMRmaps_H2CO_N1024_%03d.hdf5' % snap
    f_agn = h5py.File(image1, 'r')
    df_CO_agn = f_agn['AMRmaps']['NCO'][:]
    df_H2_agn = f_agn['AMRmaps']['NH2'][:]
    UnitLength_in_cm   = 3.085678e21 # / 3e18 for convert unit to pc-2
    X=np.ones(shape=f_agn['AMRmaps']['NCO'].shape,dtype=np.float64)
    X *= UnitLength_in_cm
    df_CO_agn = df_CO_agn * X
    df_H2_agn = df_H2_agn * X
    Map_NH2_50 = df_H2_agn.copy()
    Map_NH2_100 = df_H2_agn.copy()
    Map_NH2_300 = df_H2_agn.copy()
    # print(df_H2_agn.shape())

    hdu_2 = fits.open(image2, memmap=True)

    Map_2 = hdu_2[0].data
    Map_H2_50 = Map_2.copy()
    Map_H2_100 = Map_2.copy()
    Map_H2_300 = Map_2.copy()
    z2, h2, w2 = Map_2.shape
    mask_vel2_50 = create_circular_mask(h2, w2, center=(res, res), radius=rad50)
    mask_vel2_100 = create_circular_mask(h2, w2, center=(res, res), radius=rad100)
    mask_vel2_300 = create_circular_mask(h2, w2, center=(res, res), radius=500)
    Map_H2_50[-1, ~mask_vel2_50] = np.nan
    Map_H2_100[-1, ~mask_vel2_100] = np.nan
    Map_H2_300[-1, ~mask_vel2_300] = np.nan

    Map_NH2_300[~mask_vel2_300] = np.nan

    Map_NH2_100[~mask_vel2_100] = np.nan
    Map_NH2_50[~mask_vel2_50] = np.nan
    time = snap / 10.
    print(time, name)
    # x = np.log10(Map_H1_300.flatten())
    y = np.log10(Map_H2_100.flatten())
    z = np.log10(Map_NH2_100.flatten())

    radii_in_parsec = calculate_radii_in_parsec(x0=512, y0=512, map_width_pixels=1024, map_height_pixels=1024, cdelt1=-1.998317230538E-06, cdelt2=1.998317230538E-06, distance_mpc=14, inclination_deg=41)

    bin_cent,ymean,ysiglo,ysighi = runningmedian(z, y-z,xlolim=10,ylolim=-26,bins=20,stat='median')

    xx = [0,1,2]
    yy = [0,1,1]
    if model == 'AGN':
        c1 = next(color1)
        plt.plot(bin_cent,ymean,color = c1,lw=2,alpha=.99, markersize=1, label='t = %0d Myr' % time)
        if snap == 80:
            plt.plot(xx,yy,'k-', label = 'AGN')
    elif model == 'NoAGN':
        c2 = next(color2)
        plt.plot(bin_cent,ymean,color = c2,lw=2,ls='--',alpha=.99, markersize=1)
        if snap == 80:
            plt.plot(xx,yy,'k--', label = 'NoAGN')

    if xi is not None:
        obsx = [xi, xj]
        obsy = [yi, yj]
        obseq = [0, 0]
        plt.plot(obsx, obsy, 'k:', lw=2)
        plt.plot(obsx, obseq, 'k:', lw=2)

    # plt.text(0.2, 0.8, model, size=20, transform=plt.gcf().transFigure)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=13)
    #
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.tick_params(which='both', direction='in', **visible_ticks)
    plt.tick_params(labelsize=14)
    leg = plt.legend(loc='upper right')
    for t in leg.get_texts():
        t.set_fontsize('large')
