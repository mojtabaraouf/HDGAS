#This is for multiplot
#https://yt-project.org/doc/cookbook/complex_plots.html
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import yt
import yt.units as u
#import caesar
from readgadget import *
import sys
#import pylab as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from yt.units import kpc
from yt.units import pc
from yt.units import dimensions
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import AxesGrid


Dens  = './plots/Movie/Dens/'
Temp  = './plots/Movie/Temp/'

#Different Feedback model
input_AGN_NONP_V5B6 = '/data2/mojtaba/gizmo-public/output_GP_2609_AGN/BH3V5B6/'
input_AGN_NONP_V10B6 = '/data2/mojtaba/gizmo-public/output_Disk_0103_AGN/BH3V10B6/'
input_AGN_NONP_V5B1 = '/data2/mojtaba/gizmo-public/output_Disk_0103_AGN/BH3V5B1/'
input_AGN_NONP_V10B1 = '/data2/mojtaba/gizmo-public/output_Disk_0103_AGN/BH3V10B1/'
input_NoAGN_NONP= '/data2/mojtaba/gizmo-public/output_Disk_0103_NoAGN/BH3V0B0/'

color_map = plt.cm.get_cmap('BuPu')
cmap=color_map
# read input files from command line
ProtonMass = yt.YTQuantity(1.67e-24, "g")

UnitLength_in_cm   = 3.085678e21
UnitMass_in_g      = 1.989e43
class UnitFloat(float):

    def __new__(self, value, unit=None):
       return float.__new__(self, value)

    def __init__(self, value, unit=None):
        self.unit = unit

def Density(field, data):
    return (data['gas', 'density']/ProtonMass)


for snap in range(1,101):

    snapfile = input_AGN_NONP_V5B6 + 'snapshot_%03d.hdf5' % snap
    bbox_lim = 1000
       #kpc
    bbox = [[-bbox_lim,bbox_lim],
              [-bbox_lim,bbox_lim],
              [-bbox_lim,bbox_lim]]

    ds = yt.load(snapfile,bounding_box=bbox)
    # Volume density
    yt.add_field(("gas", "Density"), units="cm**-3",function=Density, sampling_type='cell')
    print('Read ',snapfile)
    ad = ds.all_data()

    ## compile the galaxy data from the caesar file
    x0 = ad['PartType5', 'Coordinates'].value[0][0]
    y0 = ad['PartType5', 'Coordinates'].value[0][1]
    z0 = ad['PartType5', 'Coordinates'].value[0][2]

    # center = [2.3935749603046648, 2.3250796989462676, 2.33450542394853]
    center = [x0,y0,z0]
    print(center)

    fontsize = 20
    scale = 200

# Temperature
# -------------------------------------------------------------------------------------------------
    fig = plt.figure()
    grid = AxesGrid(
        fig,
        (0.18, 0.09, 0.7, 0.90),
        # (0.015, 0.001, 0.89, 0.99),
        nrows_ncols=(2, 1),
        axes_pad=0.003,
        label_mode="L",
        # aspect = False,
        add_all = True,
        # share_all=False,
        # cbar_location="right",
        # cbar_mode="edge",
        # cbar_size="5%",
        # cbar_pad="0%",
    )

    cuts = ["x","z"]
    fields = [
        ("gas", "temperature"),
        ("gas", "temperature"),
    ]
    jj = 0
    for i, (direction, field) in enumerate(zip(cuts, fields)):
        # Load the data and create a single plot
        if jj == 1:
            p = yt.SlicePlot(ds, direction, field, center=center, width=((200 * pc, 200 * pc)))
            p.annotate_timestamp(corner='upper_left', redshift=False, time=True, draw_inset_box=True, time_format='{time:.1f} {units}',text_args={'size':40,'color':'white'})
            p.annotate_sphere(center, radius=(50, "pc"))
        else:
            # p = yt.SlicePlot(ds, direction, field, center=center, width=((200 * pc, 20 * pc)))
            p = yt.SlicePlot(ds, direction, field, center=center, width=((200 * pc, 50 * pc)))

        p.annotate_particles((10, 'pc'),p_size=50,marker='o',ptype='PartType5')

        p.set_cmap(field=("gas", "temperature"), cmap=cmap)
        # p.annotate_scale(corner='upper_right')
        p.set_zlim(("gas", "temperature"),10, 1e7)
        p.set_font_size(fontsize)
        p.hide_colorbar()
        # p.hide_axes()
        # This forces the ProjectionPlot to redraw itself on the AxesGrid axes.
        plot = p.plots[field]
        plot.figure = fig
        plot.axes = grid[i].axes

        p._setup_plots()
        jj +=1
    savename = Temp+"Map_Temp_%03d.png" % snap
    p.save(savename)

