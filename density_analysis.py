# -*- coding: utf-8 -*-
'''script with functions to calculate the turbulence velocity plots'''
from scipy.optimize import curve_fit
from scipy.special import iv, gamma, gammaln

# =============================================================
def hopkins_pdf(s, sigma_s=1.0, theta=0.1,C=1.0):
    log_pdf = hopkins_pdf_log(s, sigma_s, theta)
    return np.exp(log_pdf)*C

# ============ see Hopkins (2013) =============================
def hopkins_pdf_log(s, sigma_s=1.0, theta=0.1):
    if theta < 1e-7: theta = 1e-7 # approximate limit of zero intermittency
    s = np.array(s)
    # turn into list in case of single value input
    if s.size == 1:
        s = [s]
        s = np.array(s)
    lamb = sigma_s**2 / (2.0*theta**2)
    u = -s/theta + lamb / (1.0+theta)
    # init return values
    ret = np.zeros(s.size) - 9999999999
    for i in range(0, s.size):
        # only work on u > 0 points
        if u[i] > 0:
            log_bessel = -9999999999
            # the argument of the 1st-order modified Bessel function of the 1st kind
            arg_bessel = 2.0*np.sqrt(lamb*u[i])
            if arg_bessel < 700: # we call the scipy iv function
                log_bessel = np.log( iv(1, arg_bessel) )
            else: # approximate log(iv) for large argument
                log_bessel = arg_bessel - np.log( np.sqrt(2*np.pi*arg_bessel) )
            # now define the log(PDF)
            ret[i] = log_bessel - lamb - u[i] + np.log(np.sqrt(lamb/u[i]/theta**2))
    return ret


def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def hk(s,sigma_s,theta,C) :                # Function from Hopkins(2013b): Eq 4 in Federrath & Banarjee 2015    
    #sigma_s = -2*np.mean(s)*(1.+theta)
    lamb = ((sigma_s**2)/(2*(theta**2)))
    omega = lamb/(1+theta) - s/theta
    return (iv(1,2*np.sqrt(lamb*omega))*np.exp(-(lamb+omega))*np.sqrt(lamb/((theta**2) * omega)))*C

def fit_hopkins(data, bins=10) :

    hist = np.histogram(data,bins=bins,density=True,range=(-2.5, 2.5))
    b = hist[1]
    bins = (b[1:] + b[:-1])/2
    bins_data = bins[np.nonzero(hist[0])]
    diff=np.diff(bins_data)
    bins_data=np.append([bins_data[0]-2*diff[0],bins_data[0]-diff[0]],bins_data)
    bins_data=np.append(bins_data,[bins_data[-1]+diff[0]])
    #data
    eta_data = hist[0][np.nonzero(hist[0])]
    eta_data=np.append([0,0],eta_data,axis=None)
    eta_data=np.append(eta_data,[0],axis=None)
    
    sigma_fit = np.std(data)     
    #popt , pcov = curve_fit(hopkins_pdf,bins_data,eta_data,p0=(sigma_fit,0.1,1.0))
    popt , pcov = curve_fit(hopkins_pdf,bins_data,eta_data,p0=(sigma_fit,0.1,1.0),
                           bounds=((0,0,0), (np.inf,np.inf,np.inf)))
    perr = np.sqrt(np.diag(pcov))
    
    return popt[0], popt[1], popt[2], perr[0], perr[1], bins,hist[0]


def select_region(coor_r, rad1, which='less'):
    '''Function to select a spherical region'''

    if which == 'less':
        ind_r = np.where(coor_r < rad1)[0]
    elif which == 'lessthanequalto':
        ind_r = np.where(coor_r <= rad1)[0]
    elif which == 'greater':
        ind_r = np.where(coor_r > rad1)[0]
    elif which == 'greaterthanequalto':
        ind_r = np.where(coor_r >= rad1)[0]
    else:
        raise ValueError('Invalid value of which!')
    return ind_r

def center_of_mass(masses, x_coordinates, y_coordinates, z_coordinates, mass_BH, coordinates_BH):
    '''Function to calculate the center of mass'''
    total_mass = np.sum(masses) + mass_BH
    # Ca:wqlculate the weighted sum of coordinates
    weighted_sum_x = np.sum(masses * x_coordinates) + (mass_BH*coordinates_BH[0])
    weighted_sum_y = np.sum(masses * y_coordinates) + (mass_BH*coordinates_BH[1])
    weighted_sum_z = np.sum(masses * z_coordinates) + (mass_BH*coordinates_BH[2])

    # Calculate the center of mass coordinates/velocities
    center_of_mass_x = weighted_sum_x / total_mass
    center_of_mass_y = weighted_sum_y / total_mass
    center_of_mass_z = weighted_sum_z / total_mass

    return [center_of_mass_x, center_of_mass_y, center_of_mass_z]
def density_pdf(density_agn, density_noagn,rad1,snap):

    #fit agn
    logscaled_density_agn = np.log(density_agn/np.mean(density_agn))
    sigma_fit_agn, T_fit_agn, scale_fit_agn, error_fit_agn, error_T_agn, bins_eta_agn, hist_eta_agn  = fit_hopkins(logscaled_density_agn)

    scaled_dispersion_agn = np.sqrt(np.exp(sigma_fit_agn**2/(1.+3.*T_fit_agn +2*(T_fit_agn**2)))-1.0)
    print('fitted width AGN', scaled_dispersion_agn,sigma_fit_agn,T_fit_agn)
    
    logscaled_density_noagn = np.log(density_noagn/np.mean(density_noagn))
    sigma_fit_noagn, T_fit_noagn, scale_fit_noagn, error_fit_noagn, error_T_noagn, bins_eta_noagn, hist_eta_noagn  = fit_hopkins(logscaled_density_noagn)

    scaled_dispersion_noagn = np.sqrt(np.exp(sigma_fit_noagn**2/(1.+3.*T_fit_noagn +2*(T_fit_noagn**2)))-1.0)
    print('fitted width NOAGN', scaled_dispersion_noagn,sigma_fit_noagn,T_fit_noagn)


    plt.figure(figsize=(8,5))
    #hist_agn=plt.hist(logscaled_density_agn, bins=10, histtype='step', density=True, color='tab:blue', label='AGN')
    

    fit_bins = np.linspace(np.min(logscaled_density_agn), np.max(logscaled_density_agn), 10)
    plt.plot(fit_bins,hopkins_pdf(fit_bins,sigma_fit_agn,T_fit_agn,scale_fit_agn),color='tab:blue',linestyle='dotted',label=r'H. AGN')
    
    #hist_noagn=plt.hist(logscaled_density_noagn, bins=10, histtype='step', density=True, color='tab:orange', label='NOAGN')
    fit_bins = np.linspace(np.min(logscaled_density_noagn), np.max(logscaled_density_noagn), 10)
    plt.plot(fit_bins,hopkins_pdf(fit_bins,sigma_fit_noagn,T_fit_noagn,scale_fit_noagn),color='tab:orange',linestyle='dotted',label=r'H. No AGN')
    log_vec=np.logspace(0.4,0.8,4)
    plt.title('Hopkins fit')
    plt.text(-4.9,0.7, r'H. NoAGN sd='+str(np.round(scaled_dispersion_noagn,2))+', $\sigma_\eta$='+str(np.round(sigma_fit_noagn,2)), fontsize = 15)
    plt.text(-4.9,0.55, r'H. AGN sd='+str(np.round(scaled_dispersion_agn,2))+', $\sigma_\eta$='+str(np.round(sigma_fit_agn,2)), fontsize = 15)
    plt.grid()
    plt.xlim([-5,5])
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.grid()
    plt.ylim([0,1.5])
    plt.legend(loc='upper right',fontsize="16")
    plt.ylabel('PDF',fontsize='14')
    plt.xlabel(r'log($\rho/\rho_0$)',fontsize='14')

    label_inf='_r%03d_t%03d.png' %(rad1,snap) #adding a label with radius and time information. Important time last
    label_dir='./results_den/'  #label for the directory
    return scaled_dispersion_agn,scaled_dispersion_noagn,sigma_fit_agn,sigma_fit_noagn

    #plt.savefig(label_dir+'pdf_den_hopkins_'+label_inf)


from lmfit.models import LinearModel,PowerLawModel, GaussianModel#, PolynomialModel
from lmfit import Model
from numpy import exp, linspace, random
def NL_PL_new(x,N,alpha,A,sigma,so,st):
    fun_val=[]
    #C=(exp(0.5*(alpha-1)*alpha*sigma**2))/(sigma*np.sqrt(2*np.pi))
    #st=(alpha-0.5)*sigma**2
    #print(st)
    #so=-0.5*sigma**2
    #N=1/((C*exp(-alpha*st))/alpha+0.5+0.5*special.erf((2*st+sigma**2)/(2*np.sqrt(2)*sigma)))
    for s in x:
        if s<st:
           fun_val.append(N*(1/(np.sqrt(2*np.pi)*sigma))*exp(-(s-so)**2/(2*sigma**2)))
        else:
           fun_val.append(A*exp(-alpha*s))
    return np.array(fun_val)

def PDF_fit(density1,density2,title,name,snap,rad1):

    '''Function that creates PDF of specific data '''
    

    data_hist1=density1
    
    data_hist1=data_hist1[np.isfinite(data_hist1)]
    data_hist1=data_hist1[np.isfinite(data_hist1)]
    
    data1=np.log(data_hist1/np.mean(data_hist1))


    data_hist2=density2
    data_hist2=data_hist2[np.isfinite(data_hist2)]
    data2=np.log(data_hist2/np.mean(data_hist2))
    #plt.figure(figsize=(8,5))

    time = snap/10
    
    title_info=' '  #'%0.1f Myr %d pc limit %.4f' %(time,rad1,limit)
    plt.title(title+title_info,fontsize='20')


    #creating the histogram
    n_bins1=np.max([len(data1)*20//3000,9])
    n_bins2=np.max([len(data2)*20//3000,9])
    data1_hist=np.histogram(data1, bins=10, density=True)      #'auto', density=True)
    data2_hist=np.histogram(data2, bins=10, density=True)
    data1_hist_val=np.append([0,0],data1_hist[0],axis=None)
    data1_hist_val=np.append(data1_hist_val,[0,0],axis=None)
    data2_hist_val=np.append([0,0],data2_hist[0],axis=None)
    data2_hist_val=np.append(data2_hist_val,[0,0],axis=None)
    dif1=np.diff(data1_hist[1])
    dif2=np.diff(data2_hist[1])
    data1_hist_bin=np.append([data1_hist[1][0]-2*dif1[0],data1_hist[1][0]-dif1[0]],data1_hist[1])
    data1_hist_bin=np.append(data1_hist_bin,[data1_hist[1][-1]+dif1[0]])
    data2_hist_bin=np.append([data2_hist[1][0]-2*dif2[0],data2_hist[1][0]-dif2[0]],data2_hist[1])
    data2_hist_bin=np.append(data2_hist_bin,[data2_hist[1][-1]+dif2[0]])
    #print('data1',np.shape(data1_hist_val))
    #print('data2',np.shape(data2_hist_val))
    #data_total=[data1_hist_val,data2_hist_val]
    bins=[data1_hist_bin,data2_hist_bin]


    center=[]
    sigma=[]
    amp=[]
    C_val=[]
    alpha=[]
    N=[]
    A=[]
    so=[]
    st=[]
    label_plot=['No AGN','AGN']
    colors=['tab:orange','tab:blue']

    #transform nan into 0

    data2_hist_val=np.nan_to_num(np.array(data2_hist_val))
    data1_hist_val=np.nan_to_num(np.array(data1_hist_val))
    data_total=[data1_hist_val,data2_hist_val]
    for i in range(2):
         #comparing with gaussian fit only
        mynan_policy = 'propagate'
        mod1 = GaussianModel(prefix="mod1",nan_policy=mynan_policy)
        pars = mod1.make_params(amplitude=1.0,center=-0.5, sigma=2)

        #set parameters
        #set parameters
        pars['mod1sigma'].set(min=0.3,max=3)
        pars['mod1amplitude'].set(min=0.5,max=3)
        pars['mod1center'].set(min=-4,max=4)

        mod = mod1 #adding both mods
        x = bins[i]
        y = data_total[i]
        out = mod.fit(y, pars, x=x)

        print('gaussian fit: sigma, N, so')
        print(out.best_values['mod1sigma'],out.best_values['mod1amplitude'],out.best_values['mod1center'])

        #plt.plot(x,out.best_fit,color=colors[i], label='Best fit G '+label_plot[i])
        mynan_policy = 'propagate'
        mod2=Model(NL_PL_new)
        pars2 = mod2.make_params(N=out.best_values['mod1amplitude'],sigma=out.best_values['mod1sigma'],
        so=out.best_values['mod1center'],A=1,st=out.best_values['mod1center']+out.best_values['mod1sigma'],alpha=1)
       #set parameters
        #pars2['sigma'].set(vary=False)
        pars2['so'].set(vary=False)
        pars2['N'].set(vary=False)
        if snap==90:
            pars2['alpha'].set(min=1, max=1.3)
        else:
            pars2['alpha'].set(min=1, max=5)
        pars2['A'].set(min=0.05, max=1)
        pars2['sigma'].set(min=0.1, max=10)
        pars2['st'].set(min=out.best_values['mod1center'],max=2.5)


        #pars2['st'].set(min=-3,max=0.5)
        mod = mod2 #adding both mods
        pars=pars2
        x = bins[i]
        y = data_total[i]
        out = mod.fit(y, pars, x=x)

        #save the results
        #center.append(out.best_values['mod1center'])
        sigma.append(out.best_values['sigma'])
        print('sigma_val',out.best_values['sigma'])
        print('out', out.fit_report())
        N.append(out.best_values['N'])
        st.append(out.best_values['st'])
        so.append(out.best_values['so'])
        A.append(out.best_values['A'])
                #amp.append(out.best_values['mod1amplitude'])
        #C_val.append(out.best_values['C'])
        alpha.append(out.best_values['alpha'])
        #C=(exp(0.5*(out.best_values['alpha']-1)*out.best_values['alpha']*out.best_values['sigma']**2))/(out.best_values['sigma']*np.sqrt(2*np.pi))
        #st=(out.best_values['alpha']-0.5)*out.best_values['sigma']**2
        print('sigma, alpha,so, N, A, st')

        print(out.best_values['sigma'],out.best_values['alpha'],out.best_values['so'],out.best_values['N'],out.best_values['A'],out.best_values['st'])
        #plot the results

        #plt.plot(x, out.init_fit, 'k--', label='Inicial '+label_plot[i])
        plt.plot(x,out.best_fit,color=colors[i], label='B. '+label_plot[i])
    #adding the histogram
    plt.hist(data1[np.isfinite(data1)], bins=10,color='tab:orange',density=True,histtype='step',label='No AGN')
    plt.hist(data2[np.isfinite(data2)], bins=10 ,color='tab:blue',density=True,histtype='step',label='AGN')
    #adding text
    #plt.text(-4.9,0.9, r'No AGN N='+str(np.round(N[0],2))+', $\sigma$='+str(np.round(sigma[0],2))+', $s_0$='+str(np.round(so[0],2))+', $s_t$='+str(np.round(st[0],2))+', $A$='+str(np.round(A[0],2))+r', $\alpha$='+str(np.round(alpha[0],2)), fontsize = 15)
    #plt.text(-4.9,0.80, r'AGN N='+str(np.round(N[1],2))+', $\sigma$='+str(np.round(sigma[1],2))+', $s_0$='+str(np.round(so[1],2))+', $s_t$='+str(np.round(st[1],2))+', $A$='+str(np.round(A[1],2))+r', $\alpha$='+str(np.round(alpha[1],2)), fontsize = 15)
    log_vec=np.logspace(0.4,0.8,4)
    plt.text(-4.9,1.2, r'B. NoAGN $\sigma$='+str(np.round(sigma[0],2))+r', $\alpha$='+str(np.round(alpha[0],2)), fontsize = 15)
    plt.text(-4.9,0.95, r'B. AGN $\sigma$='+str(np.round(sigma[1],2))+r', $\alpha$='+str(np.round(alpha[1],2)), fontsize = 15)
    plt.grid()


    plt.xlim([-5,5])
    
    plt.ylim(0.01,)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right',fontsize="16")
    plt.ylabel('PDF',fontsize='14')
    plt.xlabel(r'log($\rho/\rho_0$)',fontsize='14')

    label_inf='_r%03d_t%03d.png' %(rad1,snap) #adding a label with radius and time information. Important time last
    label_dir='./results/'  #label for the directory

    plt.savefig(label_dir+name+label_inf)
    plt.close()


    plt.figure(figsize=(10,5))
    time = snap/10

    title_info='%0.1f Myr %d pc limit %.4f' %(time,rad1,limit)
    plt.title(title+title_info)

    #adding the histogram
    plt.hist(data1[np.isfinite(data1)], bins='auto',color='tab:orange',density=True,histtype='step',label='No AGN')
    plt.hist(data2[np.isfinite(data2)], bins='auto' ,color='tab:blue',density=True,histtype='step',label='AGN')
   #adding text
    plt.text(-3.9,0.9, r'No AGN N='+str(np.round(N[0],2))+', $\sigma$='+str(np.round(sigma[0],2))+', $s_0$='+str(np.round(so[0],2))+', $s_t$='+str(np.round(st[0],2))+', $A$='+str(np.round(A[0],2))+r', $\alpha$='+str(np.round(alpha[0],2)), fontsize = 10)
    plt.text(-3.9,0.7, r'AGN N='+str(np.round(N[1],2))+', $\sigma$='+str(np.round(sigma[1],2))+', $s_0$='+str(np.round(so[1],2))+', $s_t$='+str(np.round(st[1],2))+', $A$='+str(np.round(A[1],2))+r', $\alpha$='+str(np.round(alpha[1],2)), fontsize = 10)




    plt.xlim([-4,4])
    plt.ylim([0,1])
    #plt.grid()
    plt.legend(loc='lower right')
    plt.ylabel('PDF')
    plt.xlabel(r'log($\rho/\rho_0$)')

    label_inf='_r%03d_t%03d_nofit.png' %(rad1,snap) #adding a label with radius and time information. Important time last
    label_dir='./results/'  #label for the directory
    #plt.savefig(label_dir+name+label_inf)
    #plt.close()
    return sigma,alpha,so,N
import yt
import h5py
from yt.units import kpc
import matplotlib.pyplot as plt
import numpy as np
import unyt
from unyt import cm,s
from unyt import unyt_array
import astropy.constants as cons
from astropy import units as u

def simple_plot(y_agn,y_noagn,x_agn,x_noagn,ylabel,xlabel,title,name, scatter=False):
    if scatter==False:
        plt.figure(figsize=(8,5))
        plt.plot(x_agn,y_agn, label=r'$AGN$')
        plt.scatter(x_agn,y_agn,c='tab:blue',s=10)
        plt.plot(x_noagn,y_noagn,label=r'$NoAGN$')
        plt.legend(fontsize=16)
        plt.scatter(x_noagn,y_noagn,c='tab:orange',s=10)
        plt.xlabel(xlabel,fontdict={'size':15})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel(ylabel,fontdict={'size':15})
        plt.title(title,fontsize=20)
        plt.savefig('results/'+name+'.png')
    else:
        plt.figure(figsize=(8,5))
        plt.title(title,fontdict={'size':20})
        plt.scatter(x_agn,y_agn,s=35,label=r'$AGN$',edgecolor='tab:blue',fc='white')
        plt.scatter(x_noagn,y_noagn,s=35,label=r'$NoAGN$',edgecolor='tab:orange',fc='white')
        plt.legend(fontsize=16)
        plt.xlabel(xlabel,fontdict={'size':15})
        #plt.xlim(0,100)
        plt.ylabel(ylabel,fontdict={'size':15})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True)
        if len(x_agn)>0 and len(x_noagn)>0:
            plt.xscale('log')
        plt.minorticks_on()
        plt.tight_layout()

        plt.savefig('results_den/'+name+'.png')

if __name__ == '__main__':

    #packages
    import yt
    import matplotlib.pyplot as plt
    import numpy as np
    import unyt
    from unyt import cm,s
    from unyt import unyt_array
    #import the data
    snaps=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
    snaps=[75]
    snaps=[0]
    #snaps=[80]
    mach_number_agn=[]
    mach_number_noagn=[]
    sigma_agn=[]
    sigma_noagn=[]
    alpha_agn=[]
    alpha_noagn=[]
    sd_agn=[]
    sd_noagn=[]
    sigma_eta_agn=[]
    sigma_eta_noagn=[]
    for snap in snaps:
        label_dir='BH3V5B6/den/'
        file1='BH3V5B6_CH/snapshot_%03d.hdf5' %snap #agn
        file2='BH3V0B0_CH/snapshot_%03d.hdf5' %snap #no agn
        rad1=100*unyt.pc
        limit=0.998/2
        limit_val=0.998

        print('Downloading the data snap', snap)
        snapshot_agn='../'+file1
        snapshot_noagn='../'+file2

    #creating a box for downloading data
        bbox_lim=1000 #kpc
        bbox=[[-bbox_lim,bbox_lim],
        [-bbox_lim,bbox_lim],
        [-bbox_lim,bbox_lim]]


        unit_base = {'UnitMagneticField_in_gauss':  1e+4,
        'UnitLength_in_cm'         : 3.08568e+21,
        'UnitMass_in_g'            :   1.989e+43,
        'UnitVelocity_in_cm_per_s' :      1e5}

        ds_agn=yt.load(snapshot_agn,unit_base=unit_base, bounding_box=bbox)
        ds_no=yt.load(snapshot_noagn,unit_base=unit_base, bounding_box=bbox)

        #saving the data in a variable
        dd_agn=ds_agn.all_data()
        dd_no=ds_no.all_data()


        #saving the AGN coordinates
        agn_coor=dd_agn['PartType5','Coordinates'].to('pc')
        agn_length=dd_agn['PartType5','Coordinates']
        print('AGN coordinates:', agn_coor)


    #calculating center of mass
        masses_agn=dd_agn['PartType0','Masses']
        masses_noagn=dd_no['PartType0','Masses']
        coor_agn=dd_agn['PartType0','Coordinates'].to('pc')
        coor_noagn=dd_no['PartType0','Coordinates'].to('pc')
        vel_agn = dd_agn['PartType0', 'Velocities'].to('km/s')
        vel_noagn = dd_no['PartType0', 'Velocities'].to('km/s')

            #calculating center of mass
        masses_agn=dd_agn['PartType0','Masses']
        masses_noagn=dd_no['PartType0','Masses']
        coor_agn=dd_agn['PartType0','Coordinates'].to('pc')
        coor_noagn=dd_no['PartType0','Coordinates'].to('pc')
        vel_agn = dd_agn['PartType0', 'Velocities'].to('km/s')
        vel_noagn = dd_no['PartType0', 'Velocities'].to('km/s')

        mass_BH_agn = dd_agn['PartType5', 'BH_Mass']
        coor_BH_agn = dd_agn['PartType5', 'Coordinates'][0].to('pc')
        vel_BH_agn = dd_agn['PartType5', 'Velocities'][0].to('km/s')
        mass_BH_noagn = dd_no['PartType5', 'BH_Mass']
        coor_BH_noagn = dd_no['PartType5', 'Coordinates'][0].to('pc')
        vel_BH_noagn = dd_no['PartType5', 'Velocities'][0].to('km/s')

        center_mass_agn=center_of_mass(masses_agn, coor_agn[:,0],coor_agn[:,1], coor_agn[:,2],mass_BH_agn, coor_BH_agn)
        center_mass_noagn=center_of_mass(masses_noagn,coor_noagn[:,0], coor_noagn[:,1], coor_noagn[:,2],mass_BH_noagn, coor_BH_noagn)

        print('Center of mass AGN',center_mass_agn)
        print('Center of mass NoAGN',center_mass_noagn)

        dis_agn = np.sqrt((center_mass_agn[0] - coor_BH_agn[0])**2 + (center_mass_agn[1] - coor_BH_agn[1])**2 + (center_mass_agn[2] - coor_BH_agn[2])**2)
        dis_noagn = np.sqrt((center_mass_noagn[0] - coor_BH_noagn[0])**2 + (center_mass_noagn[1] - coor_BH_noagn[1])**2 + (center_mass_noagn[2] - coor_BH_noagn[2])**2)

        print('Distance from BHs for agn and noagn snapshots: ', dis_agn, dis_noagn)
    #calculating radius from center of mass
    #calculating radius from center of mass
        def _cilindrical_rho_agn(field, data):
            coor_corrected_x = coor_agn[:,0] - center_mass_agn[0]
            coor_corrected_y = coor_agn[:,1] - center_mass_agn[1]
            coor_corrected_z = coor_agn[:,2] - center_mass_agn[2]

            rho=np.sqrt(coor_corrected_x**2 + coor_corrected_y**2 + coor_corrected_z**2)
            return rho

        ds_agn.add_field(("gas", "Rho"),function=_cilindrical_rho_agn,sampling_type="local",units="pc",force_override=True)

        def _cilindrical_rho_noagn(field, data):
            coor_corrected_x = coor_noagn[:,0] - center_mass_noagn[0]
            coor_corrected_y = coor_noagn[:,1] - center_mass_noagn[1]
            coor_corrected_z = coor_noagn[:,2] - center_mass_noagn[2]
            rho = np.sqrt(coor_corrected_x**2 + coor_corrected_y**2 + coor_corrected_z**2)
            return rho

        ds_no.add_field(("gas", "Rho"),function=_cilindrical_rho_noagn,sampling_type="local",units="pc",force_override=True)

    #reload dd_agn and dd_no
        dd_agn = ds_agn.all_data()
        dd_no = ds_no.all_data()

    #    coor_cyl_noagn, vel_cyl_noagn = _cylindrical_vel(center_mass_noagn, coor_noagn, vel_noagn, masses_noagn, mass_BH_noagn, vel_BH_noagn)
    #    coor_cyl_agn, vel_cyl_agn = _cylindrical_vel(center_mass_agn, coor_agn, vel_agn, masses_agn, mass_BH_agn, vel_BH_agn)

    #extracting radius
        radius_noagn=dd_no[('gas','Rho')]
        radius_agn=dd_agn[('gas','Rho')]

        print('Min max all radius NoAGN ', np.min(radius_noagn), np.max(radius_noagn))
        print('Min max all radius AGN ', np.min(radius_agn), np.max(radius_agn))
    #making an spherical region
        ind1a_noagn=select_region(np.array(radius_noagn), rad1, which='lessthanequalto')
        ind1a_agn=select_region(np.array(radius_agn), rad1, which='lessthanequalto')
        ind1b_noagn=select_region(np.array(radius_noagn), 10*unyt.pc, which='greater')
        ind1b_agn=select_region(np.array(radius_agn), 10*unyt.pc, which='greater')
        ind1_noagn = np.intersect1d(ind1a_noagn, ind1b_noagn)
        ind1_agn = np.intersect1d(ind1a_agn, ind1b_agn)
    #ind1_noagn=calc_intersect(ind1a_noagn,ind1b_noagn)
    #ind1_agn=calc_intersect(ind1a_agn,ind1b_agn)

        print('Number of particles selected in the region NoAGN ', len(ind1_noagn), ' out of ', len(radius_noagn))
        print('Number of particles selected in the region AGN ', len(ind1_agn), ' out of ', len(radius_agn))


        print('Maximun radius in NoAGN',np.max(radius_noagn[ind1_noagn]))
        print('Maximun radius in AGN',np.max(radius_agn[ind1_agn]))

        label_reg='sphere'

        H2_abun_agn = dd_agn['PartType0', 'ChimesAbundances'][:,137]
        H2_abun_noagn = dd_no['PartType0', 'ChimesAbundances'][:,137]

    #get all particles within the limit
        ind2_noagn=select_region(H2_abun_noagn, limit, which='greater')
        ind2_agn=select_region(H2_abun_agn, limit, which='greater')

    #combine the limit and the radius
        ind_agn = np.intersect1d(ind1_agn, ind2_agn)
        ind_noagn = np.intersect1d(ind1_noagn, ind2_noagn)
    #combine the limit with radius 100pc
        ind3_agn=np.intersect1d(ind1a_agn, ind2_agn)
        ind3_noagn=np.intersect1d(ind1a_noagn, ind2_noagn)

    #ind_agn=calc_intersect(ind1_agn,ind2_agn)
    #ind_noagn=calc_intersect(ind1_noagn,ind2_noagn)
    #select data within the region and the limit
    #temp_reg_noagn_limit=dd_no['temperature'][ind_noagn]
    #den_reg_noagn_limit=dd_no['density'][ind_noagn]/(1.67E-24*unyt.g)
        den_noagn=dd_no['density']/(1.67E-24*unyt.g)
        temp_noagn=dd_no['temperature']
        den_agn=dd_agn['density']/(1.67E-24*unyt.g)
        temp_agn=dd_agn['temperature']


        #fit density Hopkins
        density_noagn = dd_no['PartType0', 'Density'][ind3_noagn]
        density_agn = dd_agn['PartType0', 'Density'][ind3_agn]
        #scaled_dispersion_agn,scaled_dispersion_noagn,sigma_fit_agn,sigma_fit_noagn=density_pdf(density_agn, density_noagn,rad1,snap)
        #sd_agn.append(scaled_dispersion_agn)
        #sd_noagn.append(scaled_dispersion_noagn)
        #sigma_eta_agn.append(sigma_fit_agn)
        #sigma_eta_noagn.append(sigma_fit_noagn)



        #fit density Burkhart
        sigma,alpha, n, m=PDF_fit(den_noagn[ind3_noagn],den_agn[ind3_agn],' ',label_dir+'den_pdf_both_fits',snap,rad1)
        #sigma_noagn.append(sigma[0])
        #sigma_agn.append(sigma[1])
        #alpha_noagn.append(alpha[0])
        #alpha_agn.append(alpha[1])


    #simple_plot(alpha_agn,alpha_noagn,np.array(snaps)/10,np.array(snaps)/10,r'$\alpha$ Burkhart',r'Myr',r'$\alpha$ ',label_dir+'alpha_evol_B',scatter=False)
    #simple_plot(sigma_agn,sigma_noagn,np.array(snaps)/10,np.array(snaps)/10,r'$\sigma$ Burkhart',r'Myr',r'$\sigma$ ',label_dir+'sigma_evol_B',scatter=False)

    #simple_plot(sigma_eta_agn,sigma_eta_noagn,np.array(snaps)/10,np.array(snaps)/10,r'$\sigma_\eta$ Hopkins',r'Myr',r'$\sigma_\eta$ ',label_dir+'sigma_evol_H',scatter=False)
    #simple_plot(sd_agn,sd_noagn,np.array(snaps)/10,np.array(snaps)/10,r'sd Hopkins',r'Myr',r'sd ',label_dir+'sd_evol_H',scatter=False)
