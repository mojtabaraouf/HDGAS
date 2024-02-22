# -*- coding: utf-8 -*-
'''script with functions to calculate the turbulence velocity plots'''
#import packages
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
#functions
def mach_number(sigma_vel, temp):
    gamma=7/5
    kb = cons.k_B.cgs.value
    mu = 2.2
    mh = cons.m_p.cgs.value + cons.m_e.cgs.value
    c_s=np.sqrt(gamma*kb*temp/(mu*mh))/u.km.to('cm') #km/s
    print('min max cs', np.min(c_s), np.max(c_s))
    print('min max sigma_vel',np.min(sigma_vel), np.max(sigma_vel))
    return sigma_vel/c_s

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

def _cylindrical_vel(center_mass, coordinates, velocities, masses, mass_BH, vel_BH):
    '''Function to transform from cartesians to cilindrical components'''

    x = coordinates[:,0] - center_mass[0]
    y = coordinates[:,1] - center_mass[1]
    z = coordinates[:,2] - center_mass[2]
    
    #calculate velocity of center of mass
    center_vel = center_of_mass(masses, velocities[:,0], velocities[:,1], velocities[:,2], mass_BH, vel_BH)

    #correct velocities from center of mass velocity
    vx = velocities[:,0] - center_vel[0]
    vy = velocities[:,1] - center_vel[1]
    vz = velocities[:,2] - center_vel[2]
    
    #calculate coordinates
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    #save results
    coor_cil = np.empty((len(coordinates), 3))
    coor_cil[:,0] = R
    coor_cil[:,1] = theta
    coor_cil[:,2] = z

         
    #calculate velocities
    R_prim = (x*vx + y*vy)/R
    theta_prim = (vy*x - y*vx)/R**2
    v_R = R_prim
    v_theta = R*theta_prim
    coor_vel = np.empty((len(coordinates), 3))
    coor_vel[:,0] = v_R.to('km/s')
    coor_vel[:,1] = v_theta.to('km/s')
    coor_vel[:,2] = vz.to('km/s')

    return coor_cil, coor_vel

def vel_turb(vel, vel_mean_pos, vel_mean_neg):
    '''Function to extract the mean velocity -calcualte the turbulence velocity components'''
    #divide the velocities into positive and negative
    ind_pos=(vel>0)
    vel_pos=vel[ind_pos]
    ind_neg=(vel<0)
    vel_neg=vel[ind_neg]
    #generate an array to save results
    #extract each mean velocity from each velocity value
    arr=np.empty(len(vel))
    arr[ind_pos]=vel_pos-vel_mean_pos
    arr[ind_neg]=vel_neg-vel_mean_neg
    return arr

def calc_turb(r,theta,z):
    '''Funtion to calculate the turbulence velocity'''
    return np.sqrt(r**2+theta**2+z**2)

def ident_bin(r_coor,bins,v_means):
    '''Function to identify the mean velocity for each point'''
    
    #save the resilts
    v_means_final=[]
    r_coor_final=[]
    num_ind=np.arange(0,len(v_means))
    num_ind_vec=[]
    if len(bins)==0:
        v_means_final=0
    else:
        #for each R calculate its bin
        diff=np.diff(bins)[0]/2
        for r in r_coor:
            #calculate the left edge of the bin
            rest=bins-diff*np.ones(len(bins))

            ind_min=(rest<r)
            #condition for points in the first bin
            if sum(ind_min)==0:
                ind_min=0
                v_means_final.append(v_means[ind_min])
                num_ind_vec.append(0)

            #condition for points in the rest of the bins
            else:
                #take the last left edge lower than R
                bins_min=bins[ind_min][-1]
                #select the respective mean velocity
                v_means_final.append(v_means[ind_min][-1])
                num_ind_vec.append(num_ind[ind_min][-1])
    #return an array with a mean velocity for each R value
    return num_ind_vec
def apply_limit(abundance1,coor1_x,coor1_y, coor1_z,abundance2,coor2_x,coor2_y,coor2_z, limit=0.998):
    '''Function to apply limit in abundance'''
    index1=(abundance1*2>limit)
    coor1_x_new=coor1_x[index1]
    coor1_y_new=coor1_y[index1]

    coor1_z_new=coor1_z[index1]

    index2=(abundance2*2>limit)
    coor2_x_new=coor2_x[index2]
    coor2_y_new=coor2_y[index2]
    coor2_z_new=coor2_z[index2]
    return coor1_x_new,coor1_y_new, coor1_z_new, coor2_x_new, coor2_y_new, coor2_z_new

def simple_plot(y_agn,y_noagn,x_agn,x_noagn,ylabel,xlabel,title,name,snap, scatter=False):
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
        
        
    label_inf='_1d_t%03d.png' %(snap)
    plt.savefig('results/'+name+label_inf)

from lmfit.models import LinearModel,PowerLawModel, GaussianModel#, PolynomialModel
from lmfit import Model
from numpy import exp, linspace, random
from scipy import special
def PDF_simple(data1,data2,sigma_ini,x_label,title,name,ylimit,snap,rad1):
    '''Function to create a simple PDF with a gaussian fit'''

    #create histogram
    data1_hist=np.histogram(data1[np.isfinite(data1)], bins='auto', density=True)
    data2_hist=np.histogram(data2[np.isfinite(data2)], bins='auto', density=True)

    #save initial values
    data_real=[data1,data2]

    #add edges for later gaussian fit
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
    bins=[data1_hist_bin,data2_hist_bin]

    #initialise values
    sigma=[]
    N=[]
    so=[]
    label_plot=['No AGN','AGN']
    colors=['tab:orange','tab:blue']

    #transform nan into num
    data2_hist_val=np.nan_to_num(np.array(data2_hist_val))
    data1_hist_val=np.nan_to_num(np.array(data1_hist_val))

    #sabe histogram values
    data_total=[data1_hist_val,data2_hist_val]

    #initialise plot
    plt.figure(figsize=(8,5))
    
    #loop for both agn and noagn
    for i in range(2):
        
        #fit a gaussian model
        mynan_policy = 'propagate'
        mod1 = GaussianModel(prefix="mod1",nan_policy=mynan_policy)
        pars = mod1.make_params(amplitude=1,center=np.mean(data_real[i]), sigma=sigma_ini)

        #set parameters
        pars['mod1sigma'].set(min=0.01,max=sigma_ini+20)
        pars['mod1amplitude'].set(min=0.5,max=1)
        #pars['mod1center'].set(min=np.mean(data_real[i])-50,max=np.mean(data_real[i])+50)

        mod = mod1
        x = bins[i]
        y = data_total[i]
        out = mod.fit(y, pars, x=x)
        sigma.append(out.best_values['mod1sigma'])
        so.append(out.best_values['mod1center'])
        N.append(out.best_values['mod1amplitude'])
        print('gaussian fit: sigma, N, so')
        print(out.best_values['mod1sigma'],out.best_values['mod1amplitude'],out.best_values['mod1center'])
        
        #add best fit
        plt.plot(x,out.best_fit,color=colors[i], label='Best fit PL '+label_plot[i])

    #generate title and labels
    time = snap/10
    title_info='%0.1f Myr %d pc' %(time,rad1)
    plt.title(title+title_info)
    #add histograms to the plot
    plt.hist(data1[np.isfinite(data1)], bins='auto',color='tab:orange',density=True,histtype='step',label='No AGN')
    plt.hist(data2[np.isfinite(data2)], bins='auto' ,color='tab:blue',density=True,histtype='step',label='AGN')
    plt.text(ylimit[0]+0.1,0.6, r'NoAGN $\sigma$='+str(np.round(sigma[0],2))+r', s$_0$='+str(np.round(so[0],2)), fontsize = 15)
    plt.text(ylimit[0]+0.1,0.30, r'AGN $\sigma$='+str(np.round(sigma[1],2))+r', s$_0$='+str(np.round(so[1],2)), fontsize = 15)

    plt.grid()
    plt.legend(loc='lower right',fontsize='16')
    plt.ylabel('PDF')
    plt.xlim(ylimit)
    plt.ylim([0.001,1])

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel(x_label)
    plt.yscale('log')
    label_inf='_t%03d.png' %(snap) #adding a label with radius and time information. Important time last
    label_dir='./results/'  #label for the directory
    plt.savefig(label_dir+name+label_inf)
    plt.close()
    return sigma, so

def scatter_coordinates(x1_total,y1_total,x2_total,y2_total,x1,y1,x2,y2,x1_limit,y1_limit,x2_limit,y2_limit,xlabel,ylabel,title,name,rad1,snap):

    fig=plt.figure(figsize=(10,5))

    # Subplot 1
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlabel(xlabel, fontdict={'size': 14})
    ax1.set_ylabel(ylabel, fontdict={'size': 14})
    ax1.set_title('AGN', fontdict={'fontsize': 20})
    #ax1.tick_params(axis='both', labelsize=12)
    ax1.scatter(x1_total, y1_total, s=2, c='k',alpha=0.2, label='Total')
    ax1.scatter(x1, y1, s=2, c='b',alpha=0.4, label='Region')
    ax1.scatter(x1_limit, y1_limit, c='r', s=4,label='Selected')
    ax1.scatter([0],[0], c='orange', s=7, label='Center mass')
    #ax1.legend(loc='lower right')
    ax1.set_xlim([-100, 100])
    ax1.set_ylim([-100, 100])
    #ax1.set(adjustable='box')
    


    ax1.tick_params(bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=10)
    plt.text(-90,80, str(snap/10)+' Myr', fontsize = 16)
    plt.tight_layout()
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    # Subplot 2
    ax2 = plt.subplot(1, 2, 2)

    ax2.set_xlabel(xlabel, fontdict={'size': 14})

    ax2.set_title('NoAGN', fontdict={'fontsize': 20})

    ax2.set_xlim([-100, 100])
    ax2.set_ylim([-100, 100])

    ax2.yaxis.set_label_position("right")
    ax2.tick_params(bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False, labelleft=False, labelright=True)
    ax2.scatter(x2_total, y2_total, s=2, c='k',alpha=0.2, label='Total')
    ax2.scatter(x2, y2, s=2, c='b', alpha=0.4, label='Region')
    ax2.scatter(x2_limit, y2_limit, c='r', s=4,label='Selected')
    ax2.scatter([0], [0], c='orange', s=7,label='Center mass')

    ax2.legend(loc='lower right',fontsize=16)
    ax2.yaxis.set_label_position("right")
   # plt.text(-9.5,1.3, str(snap/10)+' Myr', fontsize = 10)
    #ax2.set(adjustable='box')
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
        #generating title
    time=snap/10
    title_info=' %0.1f Myr %d pc' %(time,rad1)
   # fig.suptitle(title+title_info,fontsize=14)
    #saving the image
    label_inf='_r%03d_t%03d.png' %(rad1,snap) #adding a label with radius and snap information
    label_dir='./results/'  #label for the directory
    plt.savefig(label_dir+name+label_inf)
    plt.close()

from scipy import stats



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


def PDF_fit(density1,density2,title,name,snap,rad1,density=True,limit=None):

    '''Function that creates PDF of specific data '''
    if density==False:
        index1=(abundance1*2>limit)
        data_hist1=density1[index1]
        data_hist1=data_hist1[np.isfinite(data_hist1)]
        data1=np.array(data_hist1)

        index2=(abundance2*2>limit)
        data_hist2=density2[index2]
        data_hist2=data_hist2[np.isfinite(data_hist2)]
        data2=np.array(data_hist2)

        plt.figure(figsize=(8,5))
        time = snap/10
        limit=limit/2
        title_info='%0.1f Myr %d pc limit %.4f' %(time,rad1,limit)
        plt.title(title+title_info,fontsize='30')

    else:

        data_hist1=density1
     #       print('max min density')

      #      print(len(data_hist1))
        data_hist1=data_hist1[np.isfinite(data_hist1)]
    #     print('after finiti',len(data_hist1))
        data1=np.log(data_hist1/np.mean(data_hist1))

            
        data_hist2=density2
        data_hist2=data_hist2[np.isfinite(data_hist2)]
        #    print(np.max(data_hist2),np.min(data_hist2))
        data2=np.log(data_hist2/np.mean(data_hist2))
        plt.figure(figsize=(8,5))

        time = snap/10
        limit=limit/2
        title_info='%0.1f Myr %d pc limit %.4f' %(time,rad1,limit)
        plt.title(title+title_info,fontsize='20')


    #creating the histogram
    n_bins1=np.max([len(data1)*20//3000,9])
    n_bins2=np.max([len(data2)*20//3000,9])
    #print('len data1',len(data1))
    #print('len data2', len(data2))
    #if n_bins1>9:
    #    n_bins1='auto'
    #if n_bins2>9:
    #    n_bins2='auto'
    data1_hist=np.histogram(data1, bins='auto', density=True)
    data2_hist=np.histogram(data2, bins='auto', density=True)
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

    #fit the data to a gaussian

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
        plt.plot(x,out.best_fit,color=colors[i], label='Best fit PL '+label_plot[i])



    #adding the histogram
    plt.hist(data1[np.isfinite(data1)], bins='auto',color='tab:orange',density=True,histtype='step',label='No AGN')
    plt.hist(data2[np.isfinite(data2)], bins='auto' ,color='tab:blue',density=True,histtype='step',label='AGN')
    #adding text
    #plt.text(-4.9,0.9, r'No AGN N='+str(np.round(N[0],2))+', $\sigma$='+str(np.round(sigma[0],2))+', $s_0$='+str(np.round(so[0],2))+', $s_t$='+str(np.round(st[0],2))+', $A$='+str(np.round(A[0],2))+r', $\alpha$='+str(np.round(alpha[0],2)), fontsize = 15)
    #plt.text(-4.9,0.80, r'AGN N='+str(np.round(N[1],2))+', $\sigma$='+str(np.round(sigma[1],2))+', $s_0$='+str(np.round(so[1],2))+', $s_t$='+str(np.round(st[1],2))+', $A$='+str(np.round(A[1],2))+r', $\alpha$='+str(np.round(alpha[1],2)), fontsize = 15)

    plt.text(-4.9,0.9, r'NoAGN $\sigma$='+str(np.round(sigma[0],2))+r', $\alpha$='+str(np.round(alpha[0],2)), fontsize = 15)
    plt.text(-4.9,0.80, r'AGN $\sigma$='+str(np.round(sigma[1],2))+r', $\alpha$='+str(np.round(alpha[1],2)), fontsize = 15)
    plt.grid()


    plt.xlim([-5,5])
    plt.ylim([0,1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.grid()
    plt.legend(loc='upper right',fontsize="16")
    plt.ylabel('PDF',fontsize='14')
    plt.xlabel(r'log($\rho/\rho_0$)',fontsize='14')

    label_inf='_r%03d_t%03d.png' %(rad1,snap) #adding a label with radius and time information. Important time last
    label_dir='./results/'  #label for the directory

    plt.savefig(label_dir+name+label_inf)
    plt.close()


    plt.figure(figsize=(10,5))
    time = snap/10
    limit=limit/2
    title_info='%0.1f Myr %d pc limit %.4f' %(time,rad1,limit)
    plt.title(title+title_info)

    #adding the histogram
    plt.hist(data1[np.isfinite(data1)], bins='auto',color='tab:orange',density=True,histtype='step',label='No AGN')
    plt.hist(data2[np.isfinite(data2)], bins='auto' ,color='tab:blue',density=True,histtype='step',label='AGN')

    #adding text
    plt.text(-3.9,0.9, r'No AGN N='+str(np.round(N[0],2))+', $\sigma$='+str(np.round(sigma[0],2))+', $s_0$='+str(np.round(so[0],2))+', $s_t$='+str(np.round(st[0],2))+', $A$='+str(np.round(A[0],2))+r', $\alpha$='+str(np.round(alpha[0],2)), fontsize = 10)
    plt.text(-3.9,0.80, r'AGN N='+str(np.round(N[1],2))+', $\sigma$='+str(np.round(sigma[1],2))+', $s_0$='+str(np.round(so[1],2))+', $s_t$='+str(np.round(st[1],2))+', $A$='+str(np.round(A[1],2))+r', $\alpha$='+str(np.round(alpha[1],2)), fontsize = 10)




    plt.xlim([-4,4])
    plt.ylim([0,1])
    #plt.grid()
    plt.legend(loc='lower right')
    plt.ylabel('PDF')
    plt.xlabel(r'log($\rho/\rho_0$)')

    label_inf='_r%03d_t%03d_nofit.png' %(rad1,snap) #adding a label with radius and time information. Important time last
    label_dir='./results/'  #label for the directory
    plt.savefig(label_dir+name+label_inf)
    plt.close()


    return sigma,alpha, len(data1), len(data2)

def scatter_phase(x1_total,y1_total,x2_total,y2_total,x1,y1,x2,y2,x1_limit,y1_limit,x2_limit,y2_limit,xlabel,ylabel,title,name,rad1,snap):

    fig=plt.figure(figsize=(10,5))

    # Subplot 1
    ax1 = plt.subplot(1, 2, 1)

    ax1.set_xlabel(xlabel, fontdict={'size': 15})
    ax1.set_ylabel(ylabel, fontdict={'size': 15})
    ax1.set_title('AGN', fontdict={'fontsize': 15})
    #ax1.tick_params(axis='both', labelsize=12)
    ax1.scatter(x1_total,y1_total,c='k',s=2, alpha=0.2,label='Total')
    ax1.scatter(x1, y1, s=2, c='b', label='Region')
    ax1.scatter(x1_limit, y1_limit, c='r', s=2,label='Selected')
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim([10**-7,10**6])
    ax1.set_ylim([1,10**6])
    ax1.legend(fontsize=16)
    ax1.tick_params(bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Subplot 2
    plt.grid()
    ax2 = plt.subplot(1, 2, 2)

    ax2.set_xlabel(xlabel, fontdict={'size': 15})

    ax2.set_title('NoAGN', fontdict={'fontsize': 15})
    #ax2.tick_params(axis='both', labelsize=12)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim([10**-7,10**6])
    ax2.set_ylim([1,10**6])

    ax2.yaxis.set_label_position("right")
    ax2.tick_params(bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False, labelleft=False, labelright=True)
    #ax2.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor locator for upper axis
    #ax2.xaxis.tick_top()
    #ax2.yaxis.set_minor_locator(AutoMinorLocator())
    #ax2.set_xticklabels(ax2.get_xticks(), rotation=45, ha='right')  # Rotate x-axis labels

    ax2.scatter(x2_total, y2_total, s=2, c='k', alpha=0.2,label='Total')
    ax2.scatter(x2, y2, s=2, c='b', label='Region')
    
    ax2.scatter(x2_limit, y2_limit, c='r', s=2, label='Selected')
    ax2.yaxis.set_label_position("right")
    ax2.legend(fontsize=16)
    #ax2.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    #generating title
    time=snap/10
    title_info=' %0.1f Myr %d pc' %(time,rad1)
   # fig.suptitle(title+title_info,fontsize=14)
    #saving the image
    label_inf='_r%03d_t%03d.png' %(rad1,snap) #adding a label with radius and snap information
    label_dir='./results/'  #label for the directory
    plt.savefig(label_dir+name+label_inf)
    plt.close()


def twoD_profile_2d(y_data1,y_data2,x_data1,x_data2,val_data1,val_data2,y_data1_limit,y_data2_limit,x_data1_limit,x_data2_limit,val_data1_limit,val_data2_limit,ylabel,xlabel,title,name,snap,nbins=5):
    ''' Function to create velocity profiles'''
    #divide the data into positive and negative values for both agn and noagn
    ind1_pos=(val_data1>0)
    y_data1_pos=y_data1[ind1_pos]
    x_data1_pos=x_data1[ind1_pos]
    val_data1_pos=val_data1[ind1_pos]
    ind1_neg=(val_data1<0)
    y_data1_neg=y_data1[ind1_neg]
    x_data1_neg=x_data1[ind1_neg]
    val_data1_neg=val_data1[ind1_neg]

    ind2_pos=(val_data2>0)
    y_data2_pos=y_data2[ind2_pos]
    x_data2_pos=x_data2[ind2_pos]
    val_data2_pos=val_data2[ind2_pos]
    ind2_neg=(val_data2<0)
    y_data2_neg=y_data2[ind2_neg]
    x_data2_neg=x_data2[ind2_neg]
    val_data2_neg=val_data2[ind2_neg]

    #generate the plot
    plt.figure(figsize=(8,5))
    #plot the selected points within the limit
    #plt.scatter(x_data1_limit,y_data1_limit,s=40, edgecolor='tab:blue',fc='white')
    #plt.scatter(x_data2_limit,y_data2_limit,s=40, edgecolor='tab:orange',fc='white')

    #select the parts positive or negative that has values and plot the points within the region and mean
    if len(x_data1_pos)!=0:
        bin_means1_pos, bin_edges1_r_pos,bin_edges1_theta_pos, binnumber1_pos = stats.binned_statistic_2d(x_data1_pos,y_data1_pos,val_data1_pos, statistic='mean', bins=nbins)
        #plt.scatter(x_data1_pos,y_data1_pos,s=2,alpha=0.5,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
        #plt.plot((bin_edges1_pos[:-1] + bin_edges1_pos[1:])*0.5,bin_means1_pos,label='AGN',alpha=0.5,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], linewidth=4)
    else:
        bin_means1_pos, bin_edges1_r_pos,bin_edges1_theta_pos, binnumber1_pos=np.array([0]),np.array([0]),np.array([0]),np.array([0])
    if len(x_data1_neg)!=0:
        bin_means1_neg, bin_edges1_r_neg,bin_edges1_theta_neg, binnumber1_neg = stats.binned_statistic_2d(x_data1_neg,y_data1_neg,val_data1_neg, statistic='mean', bins=nbins)
        #plt.scatter(x_data1_neg,y_data1_neg,s=2,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],alpha=0.5)
        #plt.plot((bin_edges1_neg[:-1] + bin_edges1_neg[1:])*0.5,bin_means1_neg,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],alpha=0.5, linewidth=4)
    else:
        bin_means1_neg, bin_edges1_r_neg,bin_edges1_theta_neg, binnumber1_neg=np.array([0]),np.array([0]),np.array([0]),np.array([0])
    if len(x_data2_pos)!=0:
        bin_means2_pos, bin_edges2_r_pos,bin_edges2_theta_pos, binnumber2_pos = stats.binned_statistic_2d(x_data2_pos, y_data2_pos,val_data2_pos, statistic='mean', bins=nbins)

        #plt.scatter(x_data2_pos,y_data2_pos,s=2,alpha=0.5,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        #plt.plot((bin_edges2_pos[:-1] + bin_edges2_pos[1:])*0.5,bin_means2_pos,label='No AGN',alpha=0.5,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linewidth=4)
    else:
        bin_means2_pos, bin_edges2_r_pos,bin_edges2_theta_pos, binnumber2_pos=np.array([0]),np.array([0]),np.array([0]),np.array([0])
    if len(x_data2_neg)!=0:
        bin_means2_neg, bin_edges2_r_neg,bin_edges2_theta_neg, binnumber2_neg = stats.binned_statistic_2d(x_data2_neg, y_data2_neg,val_data2_neg, statistic='mean', bins=nbins)
        #plt.scatter(x_data2_neg,y_data2_neg,s=2,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],alpha=0.5)
        #plt.plot((bin_edges2_neg[:-1] + bin_edges2_neg[1:])*0.5,bin_means2_neg,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],alpha=0.5, linewidth=4)
    else:
        bin_means2_neg, bin_edges2_r_neg,bin_edges2_theta_neg, binnumber2_neg=np.array([0]),np.array([0]),np.array([0]),np.array([0])

    #plt.scatter(x_data1_limit,y_data1_limit,s=40,label='AGN', edgecolor='tab:blue',fc='white')
    #plt.scatter(x_data2_limit,y_data2_limit,s=40,label='NoAGN', edgecolor='tab:orange',fc='white')
    
    #plot 2d binned statistics
    #plt.imshow(bin_means1_pos)
    #normalise val_data1

        #return bin edges,and means for both positive and negative and agn and noagn
    bin1_r_pos=(bin_edges1_r_pos[:-1] + bin_edges1_r_pos[1:])*0.5
    bin2_r_pos=(bin_edges2_r_pos[:-1] + bin_edges2_r_pos[1:])*0.5
    bin1_r_neg=(bin_edges1_r_neg[:-1] + bin_edges1_r_neg[1:])*0.5
    bin2_r_neg=(bin_edges2_r_neg[:-1] + bin_edges2_r_neg[1:])*0.5

    bin1_theta_pos=(bin_edges1_theta_pos[:-1] + bin_edges1_theta_pos[1:])*0.5
    bin2_theta_pos=(bin_edges2_theta_pos[:-1] + bin_edges2_theta_pos[1:])*0.5
    bin1_theta_neg=(bin_edges1_theta_neg[:-1] + bin_edges1_theta_neg[1:])*0.5
    bin2_theta_neg=(bin_edges2_theta_neg[:-1] + bin_edges2_theta_neg[1:])*0.5


    val_data1_norm=np.array((val_data1-np.mean(val_data1))/np.std(val_data1)*15,dtype=int)
    plt.scatter(x_data1,y_data1,c='tab:blue',label='NoAGN')
    val_data2_norm=np.array((val_data2-np.mean(val_data2))/np.std(val_data2)*15,dtype=int)
    plt.scatter(x_data2,y_data2,c='tab:orange',label='AGN')
    #plt.scatter(x_data2,y_data2,s=val_data2,c='tab:orange',label='AGN')
    r_agn_range=np.linspace(np.min(x_data1),np.max(x_data1),100)
    r_noagn_range=np.linspace(np.min(x_data2),np.max(x_data2),100)
    theta_agn_range=np.linspace(np.min(y_data1),np.max(y_data1),100)
    theta_noagn_range=np.linspace(np.min(y_data2),np.max(y_data2),100)

    for r in bin1_r_pos:
        plt.plot(r*np.ones(len(theta_agn_range)),theta_agn_range, linestyle='dotted', c='k', alpha=0.4)
    for r in bin2_r_pos:
        plt.plot(r*np.ones(len(theta_noagn_range)),theta_noagn_range, linestyle='dotted', c='k', alpha=0.4)

    for theta in bin1_theta_pos:
        plt.plot(r_agn_range,theta*np.ones(100), linestyle='dotted', c='k', alpha=0.4)
    for theta in bin2_theta_pos:
        plt.plot(r_noagn_range,theta*np.ones(100), linestyle='dotted', c='k', alpha=0.4)
    


    #set leged, labels and title
    plt.legend(loc='upper left',fontsize=16)
    plt.ylabel(ylabel,fontdict={'size':15})
    plt.xlabel(xlabel,fontdict={'size':15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xscale('log')
    plt.minorticks_on()
    #plt.tight_layout()

    #generating title
    time=snap/10
    title_info='%0.1f Myr' %(time)
    plt.title(title+title_info,size=20)
    label_inf='_t%03d.png' %(snap) #adding a label with radius and time information. Important time last
    label_dir='./results/'  #label for the directory
    plt.savefig(label_dir+name+label_inf, facecolor='white')
    plt.close()
    #return bin edges,and means for both positive and negative and agn and noagn
    bin1_r_pos=(bin_edges1_r_pos[:-1] + bin_edges1_r_pos[1:])*0.5
    bin2_r_pos=(bin_edges2_r_pos[:-1] + bin_edges2_r_pos[1:])*0.5
    bin1_r_neg=(bin_edges1_r_neg[:-1] + bin_edges1_r_neg[1:])*0.5
    bin2_r_neg=(bin_edges2_r_neg[:-1] + bin_edges2_r_neg[1:])*0.5

    bin1_theta_pos=(bin_edges1_theta_pos[:-1] + bin_edges1_theta_pos[1:])*0.5
    bin2_theta_pos=(bin_edges2_theta_pos[:-1] + bin_edges2_theta_pos[1:])*0.5
    bin1_theta_neg=(bin_edges1_theta_neg[:-1] + bin_edges1_theta_neg[1:])*0.5
    bin2_theta_neg=(bin_edges2_theta_neg[:-1] + bin_edges2_theta_neg[1:])*0.5
    

    #plt.figure(figsize=(8,5))
    #plt.title('2D statistics AGN positive velocity')
    #plt.xlabel(r'$\theta$')
    #plt.ylabel('r')
    #plt.imshow(bin_means1_pos)
    #plt.savefig('./results_vel/fit_hist.png')

    #plt.figure(figsize=(8,5))
    #plt.title('2D statistics AGN negative velocity')
    #plt.xlabel(r'$\theta$')
    #plt.ylabel('r')
    #plt.imshow(bin_means1_neg)
    #plt.savefig('./results_vel/fit_hist_neg.png')

    return bin1_r_pos,bin1_theta_pos,bin_means1_pos,bin1_r_neg,bin1_theta_neg,bin_means1_neg,bin2_r_pos,bin2_theta_pos,bin_means2_pos,bin2_r_neg,bin2_theta_neg,bin_means2_neg



def gen_v_mean(r,theta,v_means):
    v=[]
    for i in range(len(r)):
        v.append(v_means[r[i],theta[i]])
    return np.array(v)

if __name__ == '__main__':
    
    #packages
    import yt

    #import the data
    snaps=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
    #snaps=[80]
    snaps=np.arange(5,101)
    #snaps=[80]
    mach_number_agn=[]
    mach_number_noagn=[]
    sigma_agn=[]
    sigma_noagn=[]
    alpha_agn=[]
    alpha_noagn=[]
    for snap in snaps:
        label_dir='BH3V5B1/vel_2d/'
        file1='BH3V5B1_CH/snapshot_%03d.hdf5' %snap #agn
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

        coor_cyl_noagn, vel_cyl_noagn = _cylindrical_vel(center_mass_noagn, coor_noagn, vel_noagn, masses_noagn, mass_BH_noagn, vel_BH_noagn)
        coor_cyl_agn, vel_cyl_agn = _cylindrical_vel(center_mass_agn, coor_agn, vel_agn, masses_agn, mass_BH_agn, vel_BH_agn)

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

        print('Number of particles selected in the region NoAGN ', sum(ind1_noagn), ' out of ', len(radius_noagn))
        print('Number of particles selected in the region AGN ', sum(ind1_agn), ' out of ', len(radius_agn))
        

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
    #temp_reg_agn_limit=dd_agn['temperature'][ind_agn]
    #temp_reg_agn=dd_agn['temperature'][ind1_agn]
    #temp_reg_noagn=dd_no['temperature'][ind1_noagn]
    #den_reg_agn_limit=dd_agn['density'][ind_agn]/(1.67E-24*unyt.g)

    #coordinates and velicities
    #coor_reg_agn_limit=dd_agn['PartType0','Coordinates'].to('pc')[ind_agn]
    #coor_reg_noagn_limit=dd_no['PartType0','Coordinates'].to('pc')[ind_noagn]
    #vel_reg_agn_limit=dd_agn['PartType0','Velocities'].to('pc/s')[ind_agn]
    #vel_reg_noagn_limit=dd_no['PartType0','Velocities'].to('pc/s')[ind_noagn]
    
        abun_reg_agn_limit=dd_agn['PartType0', 'ChimesAbundances'][ind_agn]
        H2_abun_reg_agn_limit=abun_reg_agn_limit[:,137]
        CO_abun_reg_agn_limit=abun_reg_agn_limit[:,148]
        Cplus_abun_reg_agn_limit=abun_reg_agn_limit[:,8]
        C1_abun_reg_agn_limit=abun_reg_agn_limit[:,7]
    #abundances no agn
        abun_reg_noagn_limit=dd_no['PartType0', 'ChimesAbundances'][ind_noagn]
        H2_abun_reg_noagn_limit=abun_reg_noagn_limit[:,137]
        CO_abun_reg_noagn_limit=abun_reg_noagn_limit[:,148]
        Cplus_abun_reg_noagn_limit=abun_reg_noagn_limit[:,8]
        C1_abun_reg_noagn_limit=abun_reg_noagn_limit[:,7]

        masses_reg_agn_limit=dd_agn['PartType0','Masses'][ind_agn]
        masses_reg_noagn_limit=dd_no['PartType0','Masses'][ind_noagn]
    

        # region 
#        scatter_coordinates(coor_agn[:,0]-center_mass_agn[0],coor_agn[:,1]-center_mass_agn[1],coor_noagn[:,0]-center_mass_noagn[0],coor_noagn[:,1]-center_mass_noagn[1],coor_agn[ind1_agn][:,0]-center_mass_agn[0],coor_agn[ind1_agn][:,1]-center_mass_agn[1],coor_noagn[ind1_noagn][:,0]-center_mass_noagn[0],coor_noagn[ind1_noagn][:,1]-center_mass_noagn[1],coor_agn[ind_agn][:,0]-center_mass_agn[0],coor_agn[ind_agn][:,1]-center_mass_agn[1],coor_noagn[ind_noagn][:,0]-center_mass_noagn[0],coor_noagn[ind_noagn][:,1]-center_mass_noagn[1],'x (pc)','y (pc)',' ','coordinates_plot',rad1,snap)
        
        #ploting scatter phase diagram
 #       scatter_phase(den_agn,temp_agn,den_noagn,temp_noagn,den_agn[ind1_agn],temp_agn[ind1_agn],den_noagn[ind1_noagn],temp_noagn[ind1_noagn],den_agn[ind_agn],temp_agn[ind_agn],den_noagn[ind_noagn],temp_noagn[ind_noagn],r'Density (cm$^{-3}$)','Temperature (K)',' ','phase_plot',rad1,snap)
        #making pdfs diagrams
        
        #sigma,alpha, n, m=PDF_fit(den_noagn[ind3_noagn],den_agn[ind3_agn],' ','den_pdf_burkhart',snap,rad1,density=True,limit=0.998)
        #sigma_noagn.append(sigma[0])
        #sigma_agn.append(sigma[1])
        #alpha_noagn.append(alpha[0])
        #alpha_agn.append(alpha[1])


#use 2d statistics
        bin_r_agn_pos,bin_theta_agn_pos,bin_means_agn_pos,bin_r_agn_neg,bin_theta_agn_neg,bin_means_agn_neg,bin_r_noagn_pos,bin_theta_noagn_pos,bin_means_noagn_pos,bin_r_noagn_neg,bin_theta_noagn_neg,bin_means_noagn_neg=twoD_profile_2d(coor_cyl_agn[:,1][ind1_agn],coor_cyl_noagn[:,1][ind1_noagn],coor_cyl_agn[:,0][ind1_agn],coor_cyl_noagn[:,0][ind1_noagn],vel_cyl_agn[:,0][ind1_agn],vel_cyl_noagn[:,0][ind1_noagn],coor_cyl_agn[:,1][ind_agn],coor_cyl_noagn[:,1][ind_noagn],coor_cyl_agn[:,0][ind_agn],coor_cyl_noagn[:,0][ind_noagn],vel_cyl_agn[:,0][ind_agn],vel_cyl_noagn[:,0][ind_noagn],r'$\theta$',r'r (pc)','Velocity profile ',label_dir+'vel_r_profile_radial_2d',snap,nbins=10)


        #calcualte the mean velocity of each plot for positive and negatives
        ind_pos=(vel_cyl_agn[:,0][ind_agn]>0)
        ind_neg=(vel_cyl_agn[:,0][ind_agn]<0)

        print(np.shape(bin_means_agn_pos))
        v_r_ind_agn_pos=ident_bin(coor_cyl_agn[:,0][ind_agn][ind_pos],bin_r_agn_pos,bin_means_agn_pos)
        v_r_ind_agn_neg=ident_bin(coor_cyl_agn[:,0][ind_agn][ind_neg],bin_r_agn_neg,bin_means_agn_neg)

        v_theta_ind_agn_pos=ident_bin(coor_cyl_agn[:,1][ind_agn][ind_pos],bin_theta_agn_pos,bin_means_agn_pos)
        v_theta_ind_agn_neg=ident_bin(coor_cyl_agn[:,1][ind_agn][ind_neg],bin_theta_agn_neg,bin_means_agn_neg)
        
        v_means_agn_pos=gen_v_mean(v_r_ind_agn_pos,v_theta_ind_agn_pos,bin_means_agn_pos)
        v_means_agn_neg=gen_v_mean(v_r_ind_agn_neg,v_theta_ind_agn_neg,bin_means_agn_neg)
        
        print('r bins',bin_r_agn_pos)
        print('r coor',coor_cyl_agn[:,0][ind_agn][ind_pos])
        
        print('theta bins',bin_theta_agn_pos)
        print('theta coor',coor_cyl_agn[:,1][ind_agn][ind_pos])

        print('means',bin_means_agn_pos)
        print('vel',vel_cyl_agn[:,0][ind_agn])
        print('vel means', v_means_agn_pos)
        


        #print(v_means_agn_pos)

        vel_turb_r_agn_limit=vel_turb(vel_cyl_agn[:,0][ind_agn], v_means_agn_pos, v_means_agn_neg)
        print('turb vel',vel_turb_r_agn_limit)
        #print(vel_turb_r_agn_limit)
        ind_pos=(vel_cyl_noagn[:,0][ind_noagn]>0)
        ind_neg=(vel_cyl_noagn[:,0][ind_noagn]<0)

        v_r_ind_noagn_pos=ident_bin(coor_cyl_noagn[:,0][ind_noagn][ind_pos],bin_r_noagn_pos,bin_means_noagn_pos)
        v_r_ind_noagn_neg=ident_bin(coor_cyl_noagn[:,0][ind_noagn][ind_neg],bin_r_noagn_neg,bin_means_noagn_neg)

        v_theta_ind_noagn_pos=ident_bin(coor_cyl_noagn[:,1][ind_noagn][ind_pos],bin_theta_noagn_pos,bin_means_noagn_pos)
        v_theta_ind_noagn_neg=ident_bin(coor_cyl_noagn[:,1][ind_noagn][ind_neg],bin_theta_noagn_neg,bin_means_noagn_neg)

        v_means_noagn_pos=gen_v_mean(v_r_ind_noagn_pos,v_theta_ind_noagn_pos,bin_means_noagn_pos)
        v_means_noagn_neg=gen_v_mean(v_r_ind_noagn_neg,v_theta_ind_noagn_neg,bin_means_noagn_neg)

        #print(v_means_agn_pos)

        vel_turb_r_noagn_limit=vel_turb(vel_cyl_noagn[:,0][ind_noagn], v_means_noagn_pos, v_means_noagn_neg)

        simple_plot(vel_turb_r_agn_limit,vel_turb_r_noagn_limit,coor_cyl_agn[:,0][ind_agn],coor_cyl_noagn[:,0][ind_noagn],r'V$_{turb,r}$ (km/s)',r'r (pc)',r'Turbulent radial component',label_dir+'vel_turb_r_2d',snap,scatter=True)


        bin_r_agn_pos,bin_theta_agn_pos,bin_means_agn_pos,bin_r_agn_neg,bin_theta_agn_neg,bin_means_agn_neg,bin_r_noagn_pos,bin_theta_noagn_pos,bin_means_noagn_pos,bin_r_noagn_neg,bin_theta_noagn_neg,bin_means_noagn_neg=twoD_profile_2d(coor_cyl_agn[:,1][ind1_agn],coor_cyl_noagn[:,1][ind1_noagn],coor_cyl_agn[:,0][ind1_agn],coor_cyl_noagn[:,0][ind1_noagn],vel_cyl_agn[:,1][ind1_agn],vel_cyl_noagn[:,1][ind1_noagn],coor_cyl_agn[:,1][ind_agn],coor_cyl_noagn[:,1][ind_noagn],coor_cyl_agn[:,0][ind_agn],coor_cyl_noagn[:,0][ind_noagn],vel_cyl_agn[:,1][ind_agn],vel_cyl_noagn[:,1][ind_noagn],r'$\theta$',r'r (pc)','Velocity profile ',label_dir+'vel_theta_profile_radial_2d',snap,nbins=10)


        #calcualte the mean velocity of each plot for positive and negatives
        ind_theta_agn_pos=(vel_cyl_agn[:,1][ind_agn]>0)
        ind_neg=(vel_cyl_agn[:,1][ind_agn]<0)

        print(np.shape(bin_means_agn_pos))
        v_r_ind_agn_pos=ident_bin(coor_cyl_agn[:,0][ind_agn][ind_theta_agn_pos],bin_r_agn_pos,bin_means_agn_pos)
        v_r_ind_agn_neg=ident_bin(coor_cyl_agn[:,0][ind_agn][ind_neg],bin_r_agn_neg,bin_means_agn_neg)

        v_theta_ind_agn_pos=ident_bin(coor_cyl_agn[:,1][ind_agn][ind_theta_agn_pos],bin_theta_agn_pos,bin_means_agn_pos)
        v_theta_ind_agn_neg=ident_bin(coor_cyl_agn[:,1][ind_agn][ind_neg],bin_theta_agn_neg,bin_means_agn_neg)

        v_means_agn_pos=gen_v_mean(v_r_ind_agn_pos,v_theta_ind_agn_pos,bin_means_agn_pos)
        v_means_agn_neg=gen_v_mean(v_r_ind_agn_neg,v_theta_ind_agn_neg,bin_means_agn_neg)



        #print(v_means_agn_pos)

        vel_turb_theta_agn_limit=vel_turb(vel_cyl_agn[:,1][ind_agn], v_means_agn_pos, v_means_agn_neg)

        #print(vel_turb_r_agn_limit)
        ind_theta_noagn_pos=(vel_cyl_noagn[:,1][ind_noagn]>0)
        ind_neg=(vel_cyl_noagn[:,1][ind_noagn]<0)

        v_r_ind_noagn_pos=ident_bin(coor_cyl_noagn[:,0][ind_noagn][ind_theta_noagn_pos],bin_r_noagn_pos,bin_means_noagn_pos)
        v_r_ind_noagn_neg=ident_bin(coor_cyl_noagn[:,0][ind_noagn][ind_neg],bin_r_noagn_neg,bin_means_noagn_neg)

        v_theta_ind_noagn_pos=ident_bin(coor_cyl_noagn[:,1][ind_noagn][ind_theta_noagn_pos],bin_theta_noagn_pos,bin_means_noagn_pos)        
        v_theta_ind_noagn_neg=ident_bin(coor_cyl_noagn[:,1][ind_noagn][ind_neg],bin_theta_noagn_neg,bin_means_noagn_neg)

        v_means_noagn_pos=gen_v_mean(v_r_ind_noagn_pos,v_theta_ind_noagn_pos,bin_means_noagn_pos)
        v_means_noagn_neg=gen_v_mean(v_r_ind_noagn_neg,v_theta_ind_noagn_neg,bin_means_noagn_neg)

        #print(v_means_agn_pos)

        vel_turb_theta_noagn_limit=vel_turb(vel_cyl_noagn[:,1][ind_noagn], v_means_noagn_pos, v_means_noagn_neg)
        simple_plot(vel_turb_theta_agn_limit,vel_turb_theta_noagn_limit,coor_cyl_agn[:,0][ind_agn],coor_cyl_noagn[:,0][ind_noagn],r'V$_{turb,\theta}$ (km/s)',r'r (pc)',r'Turbulent $\theta$ component',label_dir+'vel_turb_theta_2d',snap,scatter=True)

#z component
        bin_r_agn_pos,bin_theta_agn_pos,bin_means_agn_pos,bin_r_agn_neg,bin_theta_agn_neg,bin_means_agn_neg,bin_r_noagn_pos,bin_theta_noagn_pos,bin_means_noagn_pos,bin_r_noagn_neg,bin_theta_noagn_neg,bin_means_noagn_neg=twoD_profile_2d(coor_cyl_agn[:,1][ind1_agn],coor_cyl_noagn[:,1][ind1_noagn],coor_cyl_agn[:,0][ind1_agn],coor_cyl_noagn[:,0][ind1_noagn],vel_cyl_agn[:,2][ind1_agn],vel_cyl_noagn[:,2][ind1_noagn],coor_cyl_agn[:,1][ind_agn],coor_cyl_noagn[:,1][ind_noagn],coor_cyl_agn[:,0][ind_agn],coor_cyl_noagn[:,0][ind_noagn],vel_cyl_agn[:,2][ind_agn],vel_cyl_noagn[:,2][ind_noagn],r'$\theta$',r'r (pc)','Velocity profile ',label_dir+'vel_z_profile_radial_2d',snap,nbins=10)


        #calcualte the mean velocity of each plot for positive and negatives
        ind_pos=(vel_cyl_agn[:,2][ind_agn]>0)
        ind_neg=(vel_cyl_agn[:,2][ind_agn]<0)

        print(np.shape(bin_means_agn_pos))
        v_r_ind_agn_pos=ident_bin(coor_cyl_agn[:,0][ind_agn][ind_pos],bin_r_agn_pos,bin_means_agn_pos)
        v_r_ind_agn_neg=ident_bin(coor_cyl_agn[:,0][ind_agn][ind_neg],bin_r_agn_neg,bin_means_agn_neg)

        v_theta_ind_agn_pos=ident_bin(coor_cyl_agn[:,1][ind_agn][ind_pos],bin_theta_agn_pos,bin_means_agn_pos)
        v_theta_ind_agn_neg=ident_bin(coor_cyl_agn[:,1][ind_agn][ind_neg],bin_theta_agn_neg,bin_means_agn_neg)

        v_means_agn_pos=gen_v_mean(v_r_ind_agn_pos,v_theta_ind_agn_pos,bin_means_agn_pos)
        v_means_agn_neg=gen_v_mean(v_r_ind_agn_neg,v_theta_ind_agn_neg,bin_means_agn_neg)



        #print(v_means_agn_pos)

        vel_turb_z_agn_limit=vel_turb(vel_cyl_agn[:,2][ind_agn], v_means_agn_pos, v_means_agn_neg)
      #print(vel_turb_r_agn_limit)
        ind_pos=(vel_cyl_noagn[:,2][ind_noagn]>0)
        ind_neg=(vel_cyl_noagn[:,2][ind_noagn]<0)

        v_r_ind_noagn_pos=ident_bin(coor_cyl_noagn[:,0][ind_noagn][ind_pos],bin_r_noagn_pos,bin_means_noagn_pos)
        v_r_ind_noagn_neg=ident_bin(coor_cyl_noagn[:,0][ind_noagn][ind_neg],bin_r_noagn_neg,bin_means_noagn_neg)

        v_theta_ind_noagn_pos=ident_bin(coor_cyl_noagn[:,1][ind_noagn][ind_pos],bin_theta_noagn_pos,bin_means_noagn_pos)
        v_theta_ind_noagn_neg=ident_bin(coor_cyl_noagn[:,1][ind_noagn][ind_neg],bin_theta_noagn_neg,bin_means_noagn_neg)

        v_means_noagn_pos=gen_v_mean(v_r_ind_noagn_pos,v_theta_ind_noagn_pos,bin_means_noagn_pos)
        v_means_noagn_neg=gen_v_mean(v_r_ind_noagn_neg,v_theta_ind_noagn_neg,bin_means_noagn_neg)

        #print(v_means_agn_pos)

        vel_turb_z_noagn_limit=vel_turb(vel_cyl_noagn[:,2][ind_noagn], v_means_noagn_pos, v_means_noagn_neg)

        simple_plot(vel_turb_z_agn_limit,vel_turb_z_noagn_limit,coor_cyl_agn[:,0][ind_agn],coor_cyl_noagn[:,0][ind_noagn],r'V$_{turb,z}$ (km/s)',r'r (pc)',r'Turbulent z component',label_dir+'vel_turb_z_2d',snap,scatter=True)

            #calculate the final turbulence velocity.
        vel_turbulence_agn=calc_turb(vel_turb_r_agn_limit[ind_theta_agn_pos],vel_turb_theta_agn_limit[ind_theta_agn_pos],vel_turb_z_agn_limit[ind_theta_agn_pos])
        vel_turbulence_noagn=calc_turb(vel_turb_r_noagn_limit[ind_theta_noagn_pos],vel_turb_theta_noagn_limit[ind_theta_noagn_pos],vel_turb_z_noagn_limit[ind_theta_noagn_pos])
    
        ylimit=[-10,30]
        sigma_vel,mean_vel=PDF_simple(vel_turbulence_noagn,vel_turbulence_agn,10,r'$v_{turb}$','Turbulent velocity PDF',label_dir+'pdf_vel_turb_2d',ylimit,snap,rad1)
            #mach number calculation
        mach_noagn=mach_number(vel_turbulence_noagn,np.array(temp_noagn[ind_noagn][ind_theta_noagn_pos]))
        mach_agn=mach_number(vel_turbulence_agn, np.array(temp_agn[ind_agn][ind_theta_agn_pos]))

    #plot density and mach number
#        simple_plot(mach_agn,mach_noagn,np.log(den_agn[ind_agn][ind_theta_agn_pos]/np.mean(den_agn[ind_agn][ind_theta_agn_pos])),np.log(den_noagn[ind_noagn][ind_theta_noagn_pos]/np.mean(den_noagn[ind_noagn][ind_theta_noagn_pos])),r'M',r'log($\rho/\rho_{0})$',r'M','den_mach_'+str(snap),scatter=True)



    #make pdf of mach number
        ylimit=[0,70]
        sigma_mach, mach_mean=PDF_simple(mach_noagn,mach_agn,20,r'M',r'Mach number PDF',label_dir+'pdf_mach_number_2d',ylimit,snap,rad1)

        mach_number_noagn.append(mach_mean[0])
        mach_number_agn.append(mach_mean[1])

    snap=0
    simple_plot(mach_number_agn,mach_number_noagn,np.array(snaps)/10,np.array(snaps)/10,r'M',r'Myr',r'M',label_dir+'mach_number_2d',snap,scatter=False)
    #simple_plot(alpha_agn,alpha_noagn,np.array(snaps)/10,np.array(snaps)/10,r'$\alpha$',r'Myr',r'$\alpha$ ','alpha_evol_',scatter=False)
    #simple_plot(sigma_agn,sigma_noagn,np.array(snaps)/10,np.array(snaps)/10,r'$\sigma$',r'Myr',r'$\sigma$ ','sigma_evol_',scatter=False)

