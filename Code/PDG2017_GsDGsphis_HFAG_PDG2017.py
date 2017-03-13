# PDG 2017 version
# coding: utf-8

# In[1]:

## HFAG, Combined Gs, DGs including correlated statistical and
## systematical uncertainties 
#### Rick van Kooten, Olivier Leroy, update after ICHEP 2014 
#### Re-writing code in Python: Dawid Gerstel


# In[ ]:

# useful visualisation and numerical tools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from iminuit import Minuit, describe, Struct, util

# # Inputs
# 
# - LHCb data: have txt file for $J/\Psi KK$
# - ATLAS data: https://arxiv.org/abs/1601.03297
# - CMS data: https://arxiv.org/abs/1507.07527
# - CDF data: missing
# - D0 data (RickVK): https://www-d0.fnal.gov/Run2Physics/WWW/results/final/B/B11C/

# Below, I considerd LHCb Psi2S Phi and LHCb J/Psihh as seperate experiments.
# LHCb Jpsi hh means (PRL114, JpsiPiPi + JpsiKK around the phi mass, 3fb-!1)

# In[ ]:

# PDG2017 list, paper published before March 2017
experiments = ["LHCb_Psi2S_Phi", "LHCb_JPsi_hh", "ATLAS", "CMS", "CDF", "D0"]


# Full latest list, including preliminary results
#experiments = ["LHCb_Psi2S_Phi", "LHCb_JPsi_hh", "LHCb_JPsi_KKHM", "ATLAS", "CMS", "CDF", "D0"]



# params to be just read (not computed)
params = ["Gs", "Gs_estat", "Gs_esyst", "DGs", "DGs_estat", "DGs_esyst", "rho_Gs_DGs_stat", "rho_Gs_DGs_syst"]
# 2-dim dictionary with keys: [EXPERIMENT][PARAMETER]
val = {}
for exp in experiments:
    val[exp] = {}

# === INPUT DATA ==============================================================================
# === LHCb J/Psi hh (PRL114, JpsiPiPi+JpsiKK 3fb-1) data ===
val["LHCb_JPsi_hh"]['phis']              = -0.008519515914473901
val["LHCb_JPsi_hh"]['phis_estat']        = 0.
val["LHCb_JPsi_hh"]['phis_esyst']        = 0.038025510606221016
val["LHCb_JPsi_hh"]['Gs']                = 0.6603
val["LHCb_JPsi_hh"]['Gs_estat']          = 0.0027
val["LHCb_JPsi_hh"]['Gs_esyst']          = 0.0015
val["LHCb_JPsi_hh"]['DGs']               = 0.0805
val["LHCb_JPsi_hh"]['DGs_estat']         = 0.0091
val["LHCb_JPsi_hh"]['DGs_esyst']         = 0.0032
val["LHCb_JPsi_hh"]['rho_Gs_DGs_stat']   = -.45 
val["LHCb_JPsi_hh"]['rho_Gs_DGs_syst']   = +.44 
val["LHCb_JPsi_hh"]['rho_phis_DGs_stat'] = 0 
val["LHCb_JPsi_hh"]['rho_phis_DGs_syst'] = 0  # assume no correlation

# === LHCb J/Psi KK above the phi (High Mass), 3fb-1 data ===
# val["LHCb_JPsi_KKHM"]['phis']              = 0.12
# val["LHCb_JPsi_KKHM"]['phis_estat']        = 0.11
# val["LHCb_JPsi_KKHM"]['phis_esyst']        = 0.03
# val["LHCb_JPsi_KKHM"]['Gs']                = 0.650
# val["LHCb_JPsi_KKHM"]['Gs_estat']          = 0.006
# val["LHCb_JPsi_KKHM"]['Gs_esyst']          = 0.004
# val["LHCb_JPsi_KKHM"]['DGs']               = 0.066
# val["LHCb_JPsi_KKHM"]['DGs_estat']         = 0.018
# val["LHCb_JPsi_KKHM"]['DGs_esyst']         = 0.010
# val["LHCb_JPsi_KKHM"]['rho_Gs_DGs_stat']   = 0.0 # tot
# val["LHCb_JPsi_KKHM"]['rho_Gs_DGs_syst']   = 0
# val["LHCb_JPsi_KKHM"]['rho_Gs_DGs_tot']    = 0.54
# val["LHCb_JPsi_KKHM"]['rho_phis_DGs_stat'] = 0  # 
# val["LHCb_JPsi_KKHM"]['rho_phis_DGs_syst'] = 0  # 
# val["LHCb_JPsi_KKHM"]['rho_phis_DGs_tot']  = 0.02  # hard-coded


# === LHCb Psi2S Phi data ===
val["LHCb_Psi2S_Phi"]['phis']              = 0.23
val["LHCb_Psi2S_Phi"]['phis_estat']        = 0.285
val["LHCb_Psi2S_Phi"]['phis_esyst']        = 0.02
val["LHCb_Psi2S_Phi"]['Gs']                = .668
val["LHCb_Psi2S_Phi"]['Gs_estat']          = .011
val["LHCb_Psi2S_Phi"]['Gs_esyst']          = .007
val["LHCb_Psi2S_Phi"]['DGs']               = .066
val["LHCb_Psi2S_Phi"]['DGs_estat']         = .0425
val["LHCb_Psi2S_Phi"]['DGs_esyst']         = .007  
val["LHCb_Psi2S_Phi"]['rho_Gs_DGs_stat']   = -.4
val["LHCb_Psi2S_Phi"]['rho_Gs_DGs_syst']   = 0  
val["LHCb_Psi2S_Phi"]['rho_phis_DGs_stat'] = 0.19
val["LHCb_Psi2S_Phi"]['rho_phis_DGs_syst'] = 0 # assume no corr...


# === ATLAS params ===
# taking this from altas run1 published paper (concerns phis vs DGs)
val["ATLAS"]['phis']              = -0.090
val["ATLAS"]['phis_estat']        = 0.078
val["ATLAS"]['phis_esyst']        = 0.041
val["ATLAS"]['Gs']                = .675
val["ATLAS"]['Gs_estat']          = .003
val["ATLAS"]['Gs_esyst']          = .003
val["ATLAS"]['DGs']               = .085
val["ATLAS"]['DGs_estat']         = .011
val["ATLAS"]['DGs_esyst']         = .007
val["ATLAS"]['rho_Gs_DGs_stat']   = -.414
val["ATLAS"]['rho_Gs_DGs_syst']   = 0
val["ATLAS"]['rho_phis_DGs_stat'] = 0.097
val["ATLAS"]['rho_phis_DGs_syst'] = 0  # assume no correlation


# === CMS params ===
# some computations needed:
# published Phys.Lett.B 757,97 (2016), 20fb-1 8TeV only
clight = 299.792458
ctauCMS = 447.2
ctauCMSestat = 2.9
ctauCMSesyst = 3.7
val["CMS"]['phis']              = -0.075
val["CMS"]['phis_estat']        = 0.097
val["CMS"]['phis_esyst']        = 0.031
val["CMS"]['Gs']                = clight / ctauCMS
val["CMS"]['Gs_estat']          = clight * ctauCMSestat / ctauCMS ** 2
val["CMS"]['Gs_esyst']          = clight * ctauCMSesyst / ctauCMS ** 2
val["CMS"]['DGs']               = 0.095
val["CMS"]['DGs_estat']         = 0.013 
val["CMS"]['DGs_esyst']         = 0.007
val["CMS"]['rho_Gs_DGs_stat']   = -0.55 
val["CMS"]['rho_Gs_DGs_syst']   = 0  # assume no correlated systematics between Gs and DGs
val["CMS"]['rho_phis_DGs_stat'] = 0.10 
val["CMS"]['rho_phis_DGs_syst'] = 0  # assume no correlated systematics between phis and DGs

# === CDF params ===
# Taken from Phys.Rev.Lett 109,171802 (2012), assuming phis Gaussian
# and centered (phis vs DGs)
val["CDF"]['phis']             = -0.24
val["CDF"]['phis_estat']       = 0.36  # Trick 
val["CDF"]['phis_esyst']       = 0     # Trick
val["CDF"]['Gs']               = 0.65445026178
val["CDF"]['Gs_estat']         = 0.0081377977577
val["CDF"]['Gs_esyst']         = 0.0038547463
val["CDF"]['DGs']              = 0.068
val["CDF"]['DGs_estat']        = 0.026
val["CDF"]['DGs_esyst']        = 0.009
val["CDF"]['rho_Gs_DGs_stat']  = -0.52 
val["CDF"]['rho_Gs_DGs_syst']  = 0 # assume no correlated systematics between Gs and DGs 
val["CDF"]['rho_phis_DGs_tot'] = 0  # hard-coded

# === D0 params ===
# Taken from Phys Rev D 85, 032006 (2011)
val["D0"]['phis']             = -0.55
val["D0"]['phis_estat']       = 0.37  # trick
val["D0"]['phis_esyst']       = 0
val["D0"]['Gs']               = 0.6930
val["D0"]['Gs_estat']         = 0.017529
val["D0"]['Gs_esyst']         = 0.  
val["D0"]['DGs']              = 0.163
val["D0"]['DGs_estat']        = 0.0645
val["D0"]['DGs_esyst']        = 0.0
val["D0"]['rho_Gs_DGs_stat']  = -0.05
val["D0"]['rho_Gs_DGs_syst']  = 0 # assume no correlated systematics between Gs and DGs 
val["D0"]['rho_phis_DGs_tot'] = -0.05  # hard-coded
# ============================================================================================



# Compute and add to `val` etots and rho_tots
# 2 utilities:
def etot(exp, paramName):
    return np.sqrt(val[exp][paramName + "_estat"] ** 2 + val[exp][paramName + "_esyst"] ** 2)

def rho(exp, param1, param2):
    return (val[exp][param1+"_estat"] * val[exp][param2+"_estat"] * val[exp]["rho_"+param1+"_"+param2+"_stat"] +\
            val[exp][param1+"_esyst"] * val[exp][param2+"_esyst"] * val[exp]["rho_"+param1+"_"+param2+"_syst"])/\
            (val[exp][param1+"_etot"] * val[exp][param2+"_etot"])

for exp in experiments:
    for param in ["Gs", "DGs", "phis"]:
        val[exp][param+"_etot"] = etot(exp, param)
    val[exp]["rho_Gs_DGs_tot"] = rho(exp, "Gs", "DGs")
    if exp not in ["CDF", "D0"]:  # skip hard-coded ones
            val[exp]["rho_phis_DGs_tot"] = rho(exp, "phis", "DGs")


# ### p.d.f.'s for the experiments

# In[ ]:

def gaussian2D(x, y, x0, xsig, y0, ysig, rho):
    """Joint p.d.f. of Gs and DGs for JPsi_Phi-like channels
    params:
    =======
    x, y - specify point for the function to be evaluated at
    xsig, ysig - total uncertainties of these
    rho - total rho
    """
    ### ol protect /0
    if ( (xsig*ysig)!=0 ) and ( xsig * ysig * np.sqrt(1 - rho ** 2) !=0 ):
     return 1. / (2 * np.pi * xsig * ysig * np.sqrt(1 - rho ** 2)) *\
            np.exp( (-1. / (2 * (1 - rho **2))) * ((x - x0) ** 2 / xsig ** 2 +\
            (y - y0) ** 2 / ysig ** 2 - 2 * rho * (x - x0) * (y - y0) / (xsig * ysig)) )
    else:
     return 0

# In[ ]:

def jointPDF(x, y, pm, exps):
    """Returns product of 2-dim Gaussians from all experiments considered
    parameters:
    ===========
    x - n-dim array of arguments
    pm - tuple / list of observables of which to compute joint pdf,
             e.g. pm[0] = "Gs" and pm[1] = "DGs" for analysing these two together
    exps - list of experiments to be taken into account; allows to constrain analysis to e.g. one experiment
    """
    res = 1.
    for exp in exps:
        res *= gaussian2D(x, y, val[exp][pm[0]], val[exp][pm[0]+"_etot"],
                    val[exp][pm[1]], val[exp][pm[1]+"_etot"], val[exp]["rho_"+pm[0]+"_"+pm[1]+"_tot"])
    return res        
    

def jointPDFlog(x, y, pm, exps): return -2 * np.log(jointPDF(x, y, pm, exps))


# # Analysis of $\Delta \Gamma_{s}$ and $\phi_{s}^{c\bar{c}s}$

# ### Minimisation with Minuit (migrad, minos)
###########################################################################################################

# In[ ]:


class Maxlikelihood(object):
    """ Class to be passed to Minuit constructor. Method `__call__(self, x, y)` is to be minimised.
        Using class has advantage over simple function, because it allows to set parameters (`par` and `exps`),
        which is not possible otherwise (Minuit accepts functions with no parameters but those to be optimised))

        If `log`==False, product of double gaussians for specified experiments `exps` is returned,
        unless a PDF `pdf` is specified - then `pdf` is returned.
        If `log`==True (default) -2*log of that product of pdf's is returned
    """
    def __init__(self, par, exps, pdf=None, log=True):
        self.par, self.exps, self.pdf, self.log = (par, exps, pdf, log)
        print '=' * 100
        print "Max lihelihood for parameters: ", self.par, "\nFor experiments: ", self.exps
        print '=' * 100
            
    def __call__(self, x, y):
        """ Function wrapper of function to be minimised """
        if self.log:
            if self.pdf == None:
                return jointPDFlog(x, y, pm=self.par, exps=self.exps)
            return -2 * np.log(self.pdf(x, y))
        else:
            if self.pdf == None:
                return jointPDF(x, y, pm=self.par, exps=self.exps)
            return self.pdf(x, y)


class Minimiser(object):
    def __init__(self, fun_obj, fitParams, fname, fmode='w', header="", parnames=('x','y')):
        self.minimise(fun_obj, fitParams)
        self.printall(fname, fmode, header, parnames)

    def minimise(self, fun_obj, fitParams):
        self.m = Minuit(fun_obj, **fitParams)
        self.m.migrad()
        self.m.minos()

    def x(self):
        return self.m.values['x']

    def y(self):
        return self.m.values['y']

    def xerr(self):
        return self.m.errors['x']

    def yerr(self):
        return self.m.errors['y']

    def eminus_x(self):
        return self.m.get_merrors()['x']['lower']

    def eplus_x(self):
        return self.m.get_merrors()['x']['upper']

    def eminus_y(self):
        return self.m.get_merrors()['y']['lower']

    def eplus_y(self):
        return self.m.get_merrors()['y']['upper']

    def rho(self):
        return self.m.matrix(correlation=True)[0][1]

    def printall(self, fname, fmode, header, parnames):
            # Extract all fit params
            x, y = self.m.values['x'], self.m.values['y']
            errs = self.m.get_merrors()
            exminus, explus = errs['x']['lower'], errs['x']['upper']
            eyminus, eyplus = errs['y']['lower'], errs['y']['upper']
            rho = self.m.matrix(correlation=True)[0][1]

            with open(fname, fmode) as f:
                print >>f, '\n'
                print >>f, header
                print >>f, '=' * 30
                print >>f, parnames[0], "=", x, "^{+", explus, "}_{", exminus, "}"
                print >>f, parnames[1], "=", y, "^{+", eyplus, "}_{", eyminus, "}"
                print >>f, "rho(", parnames[0], ", ", parnames[1], ") = ", rho
                
# starting values for the Phis, DGs parameters
# Directory of all results (pictures and txt)
ResDir = ""
outputFile = ResDir + "Results_GsDGs_and_PhisDGs_PDG2017.txt"
fitParams = dict(x=-0.3, y=0.085, error_x=0.0325, error_y=0.0065, limit_x=None, limit_y=None)
parnames = ('phis', 'DGs')
func = Maxlikelihood(par=parnames, exps=experiments)
m = Minimiser(func, fitParams, fname=outputFile, header="Result from the global fit:", parnames=parnames)
###########################################################################################
# ### Contour plots

# In[ ]:

# grid
X = np.linspace(-1.2, 0.4, 1000)
Y = np.linspace(0, 0.3, 1000)
X, Y = np.meshgrid(X, Y)

# lists of colours, labels, experiment channels, coordinates of labels
lhcb_channels = experiments[0:2]
channels = [["ATLAS"], ["D0"], ["CMS"], ["CDF"], experiments, lhcb_channels]
labels = [r"ATLAS 19.2 fb$^{-1}$", r"D0 8 fb$^{-1}$", r"CMS 19.7 fb$^{-1}$", r"CDF 9.6 fb$^{-1}$", "Combined", r"LHCb 3 fb$^{-1}$"]
coords = [(0.42, 0.1), (0.63, 0.8), (0.73, 0.6), (0.98, 0.2), (0.6, 0.45), (0.65, 0.1)]
colors = ['brown', 'b', 'r', 'orange', 'white', 'g']

fig, ax = plt.subplots(1, figsize=(12,8))

# Draw contours
for i in range(len(channels)):
    lnL = jointPDFlog(X, Y, pm=("phis", "DGs"), exps=[exper for exper in channels[i]])
    dlnL = lnL - np.min(lnL)
    ## contour 2.30 <-> 1-sigma confidence interval on 2 delta log likelihood
    ## filled in contours and then edge contours (for pretty display)
    plt.contourf(X, Y, dlnL, levels=[0, 2.30], alpha=0.5, colors=colors[i])
    plt.contour(X, Y, dlnL, levels=[0, 2.30], alpha=1, colors=colors[i], linewidths=2)
    plt.text(coords[i][0], coords[i][1], labels[i], verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, color=colors[i], fontsize=15)
    
# SM theoretical prediction (all errors > 0 for convenience)
phis_SM, phis_SM_err_down, phis_SM_err_up = -0.0376, 0.0007, 0.0008
DGs_SM, DGs_SM_err_down, DGs_SM_err_up = 0.088, 0.020, 0.020

# draw rectangle for SM
def rectangle(x, xerr_down, xerr_up, y, yerr_down, yerr_up, kwargs):
    import matplotlib.patches as patches
    assert(xerr_down >= 0 and xerr_up >= 0 and yerr_down >= 0 and yerr_up >= 0)
    left_bottom = (x - xerr_down, y - yerr_down)
    dx, dy = (xerr_down + xerr_up, yerr_down + yerr_up)
    return patches.Rectangle(left_bottom, width=dx, height=dy, **kwargs)

cosmetics = {'linewidth':1, 'edgecolor':'k', 'facecolor':'none'}
rect = rectangle(phis_SM, phis_SM_err_down, phis_SM_err_up, DGs_SM, DGs_SM_err_down, DGs_SM_err_up, cosmetics)

# Add the rectangle to the axes
ax.add_patch(rect)
plt.text(0.515, 0.5, "SM", verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes, color='k', fontsize=15)


# Add plot description
plt.text(0.95, 0.70, "68% CL contours", verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes, color='k', fontsize=16)
plt.text(0.95, 0.65, r"($\Delta$ log $\mathcal{L}$ = 1.15)", verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes, color='k', fontsize=18)

# Display ranges and axes ticks; axes titles
phismin, phismax, DGsmin, DGsmax = (-0.5, 0.5, 0.05, 0.15)
plt.axis([phismin, phismax, DGsmin, DGsmax])
plt.xticks(np.arange(phismin, phismax, .05))
plt.yticks(np.arange(DGsmin, DGsmax, .005))

# hide every other tick-label
def hide_half_labels(ax):
    for label in ax.get_xticklabels()[1::2]:
        label.set_visible(False)
    for label in ax.get_yticklabels()[1::2]:
        label.set_visible(False)    

hide_half_labels(ax)

# Axes settings and labels    
ax.set_autoscale_on(False)
plt.xlabel(r'$\phi_{s}^{c\bar{c}s}[\mathrm{rad}]$', fontsize=26)
plt.ylabel(r'$\Delta \Gamma_{s}[\mathrm{ps}^{-1}]$', fontsize=26)

# plt.show()
def saveplt(plt, name):
    for suffix in ['png', 'pdf', 'eps', 'jpg']:
        plt.savefig(ResDir + name + '.' + suffix, bbox_inches='tight')  
        
saveplt(plt, name='Phis_vs_DGs')

#############################################################################################
#############################################################################################
#############################################################################################
# # Analysis of $\Delta \Gamma_{s}$ and $\Gamma_{s}$
#############################################################################################


# ### Input data (flavour-specific and CP-related) and p.d.f.'s
# In[ ]:
# (* tauDsDs constraint, only LHCb.  tauDsDs = tauL(1+phis^2 ys/2 )*)
tauDsDs1 = 1.37900
etauDsDs1 = 0.03106
# (* JpsiEta, LHCb ICHEP 2016, CP-even, tauL *)
tauJpsiEta1 = 1.479
etauJpsiEta1 = 0.03573513677
# (*tauJpsif0 constraint,only CDF. tauJpsif0=tauH(1-phis^2 ys/2)*)
# (* average of CDF, LHCb JpsiPiPi 1fb-1 and D0 2016 *)
tauf01 = 1.65765
etauf01 = 0.03188 

# Flavour specific lifetime, including LakeLouis2017, tauFS (DsMuNu, LHCb)
# computed by OS
tauFS1 = 1.516
etauFS1 = 0.014
phis1 = 0.

def CP(x, y, tauCP, etauCP, phis, eta):
    """Generic joint pdf of Gs and DGs for CP-eigenstate channels (Bs2KK ??)"""
    return 1 / (np.sqrt(2 * np.pi) * etauCP) * np.exp( -1. / 2 * ((2. / (2 * x + eta * y) *\
               (1 + eta * phis ** 2 * y / (4 * x)) - tauCP) / etauCP) ** 2) 

def CP_even(x, y):
    return CP(x, y, tauDsDs1, etauDsDs1, phis1, eta=1) * CP(x, y, tauJpsiEta1, etauJpsiEta1, phis1, eta=1)

def CP_odd(x, y):
    return CP(x, y, tauf01, etauf01, phis1, eta=-1)

def flavour_specific(x, y, tauFS=tauFS1, etauFS=etauFS1):
    """Joint pdf of Gs and DGs for flavour-specific lifetime tauFS"""
    return 1 / (np.sqrt(2 * np.pi) * etauFS) * np.exp(-1. / 2 * ((1. / x * (1 + (y / (2 * x)) ** 2) / (1 - (y / (2 * x)) ** 2) - tauFS) / etauFS) ** 2)
    
def hadronic(x, y):
    """ pdf taking into account Bs->J/Psi hh only """
    return jointPDF(x, y, pm=("Gs", "DGs"), exps=experiments)

def hadronic_and_lifetimes(x, y):
    """ pdf taking into account Bs->J/Psi hh and CP even and odd """
    return hadronic(x, y) * CP_even(x, y) * CP_odd(x, y)
    
def totpdf_Gs_DGs(x, y):
    """Total pdf, including JpsiKK, JpsiPipi, CP and flavour specific lifetimes"""
    return hadronic_and_lifetimes(x, y) * flavour_specific(x, y, tauFS1, etauFS1)

# Change of variable to tau
def CP_tau(x, y, tauCP, etauCP, phis, eta):
    """Generic joint pdf of Gs and DGs for CP-eigenstate channels (Bs2KK ??)"""
    if (eta == 1):
        return 1. / (np.sqrt(2 * np.pi) * etauCP) * np.exp( -1./2 * ( (x*eta - tauCP)/etauCP )**2 )
    return 1. / (np.sqrt(2 * np.pi) * etauCP) * np.exp( -1./2 * ( (-y*eta - tauCP)/etauCP )**2 )

def CP_even_tau(x,y):
    return CP_tau(x, y, tauDsDs1, etauDsDs1, phis1, eta=1) * CP_tau(x, y, tauJpsiEta1, etauJpsiEta1, phis1, eta=1)

def CP_odd_tau(x,y):
    return CP_tau(x, y, tauf01, etauf01, phis1, -1)

def flavour_specific_tau(x, y, tauFS=tauFS1, etauFS=etauFS1):
    """Joint pdf of Gs and DGs for flavour-specific lifetime tauFS"""
    return 1 / (np.sqrt(2 * np.pi) * etauFS) * np.exp(-1. / 2 * ( ((x**2 + y**2)/(x+y) - tauFS )/etauFS)** 2)


# ### Minimisation with MINUIT (MIGRAD, MINOS)
def translate_gamma2tau(gs, dgs):
    taul = 2. / (2 * gs + dgs)
    tauh = 2. / (2 * gs - dgs)
    return (taul, tauh)

def print_tau_fit(m, fname):
    Gs, DGs = m.x(), m.y()
    errGs, errDGs = m.xerr(), m.yerr()
    eplusGs, eminusGs = m.eplus_x(), m.eminus_x()
    rho = m.rho()
    tauS = 1. / Gs
    eplus_tauS = eplusGs / Gs ** 2
    eminus_tauS = eminusGs / Gs ** 2
    tauL, tauH = translate_gamma2tau(Gs, DGs)
    etauL = (2. / (2 * Gs + DGs) ** 2) * np.sqrt(4 * errGs ** 2 + errDGs ** 2 \
             + 4 * rho * errGs * errDGs)
    etauH = (2. / (2 * Gs - DGs) ** 2) * np.sqrt(4 * errGs ** 2 + errDGs ** 2 \
             - 4 * rho * errGs * errDGs)
    DGsoverGs = DGs / Gs
    eDGsOvGs = np.sqrt((1. / Gs) ** 2 * errDGs ** 2 + (DGs / Gs ** 2) ** 2 * \
               errGs ** 2 + 2 * rho * errGs * errDGs * (1 / Gs) * \
               (-DGs / Gs ** 2))

    with open(fname, 'a') as f:
        print >>f, 'taus = 1/Gs =', tauS, "^{+", eplus_tauS, "}_{", eminus_tauS, "}"
        print >>f, 'tauL = 1/GL = ', tauL, ' +/- ', etauL
        print >>f, 'tauH = 1/GH = ', tauH, ' +/- ', etauH
        print >>f, 'DGs/Gs = ', DGsoverGs, ' +/- ', eDGsOvGs

# starting values for the Gs, DGs parameters
parnames = ('Gs', 'DGs')
fitParams = dict(x=.664, y=0.085, error_x=0.0022, error_y=0.0065, limit_x=None, limit_y=None)

# (J/PSI hh only)
func = Maxlikelihood(par=parnames, exps=experiments, pdf=hadronic)
m = Minimiser(func, fitParams, fname=outputFile, fmode='a', header="Fit using ONLY Bs->Jpsihh:", parnames=parnames)
print_tau_fit(m, outputFile)

# J/Psi hh and CP even and odd
func = Maxlikelihood(par=parnames, exps=experiments, pdf=hadronic_and_lifetimes)
m = Minimiser(func, fitParams, fname=outputFile, fmode='a',
              header="Fit using Jpsihh and effective lifetimes (CP-even and odd):", parnames=parnames)
print_tau_fit(m, outputFile)

# J/Psi hh and CP even and odd, and flavour specific
func = Maxlikelihood(par=parnames, exps=experiments, pdf=totpdf_Gs_DGs)
m = Minimiser(func, fitParams, fname=outputFile, fmode='a',
              header="DEFAULT fit using Jpsihh, effective lifetimes and tauFS:", parnames=parnames)
print_tau_fit(m, outputFile)

##############################################################################################
# ### Contour plots

# In[ ]:

# grid
Gsmin, Gsmax, DGsmin, DGsmax = (0.62, 0.75, 0, 0.25)
X = np.linspace(Gsmin, Gsmax, 1000)
Y = np.linspace(DGsmin, DGsmax, 1000)
X, Y = np.meshgrid(X, Y)

# lists of colours, labels, experiment channels, coordinates of labels
channels = [experiments]
labels = [r"$\tau(B^{0}_{s} \rightarrow D_{s} D_{s},$"+"\n"+r"$J/\psi \eta)$",
          r"$\tau (B^{0}_{s} \rightarrow J/\psi \pi \pi,$"+"\n"+r"$J/\psi f_0)$",
          r"$\tau(B^{0}_{s} \rightarrow$ flavour specific)", 
          r"$B^{0}_{s}\rightarrow c\bar{c}KK$", "Combined"]

coords = [(0.8, 0.05), (0.7, 0.5), (0.5, 0.93), (0.45, 0.25), (0.4, 0.36)]
colors = ['magenta', 'green', 'blue', 'red', 'Black']

# all p.d.f.'s to be displayed
pdfs = [CP_even, CP_odd, flavour_specific, Maxlikelihood(par=('Gs', 'DGs'), exps=experiments, pdf=None, log=False), totpdf_Gs_DGs]

fig, ax = plt.subplots(1, figsize=(10,9))

# Draw contours
for i in range(len(pdfs)):
    lnL = -2 * np.log(pdfs[i](X, Y))
    dlnL = lnL - np.min(lnL)
    # contour 1 <-> 39% confidence interval on 2 delta log likelihood
    # filled in contours and then edge contours (for pretty display)
    plt.contourf(X, Y, dlnL, levels=[0, 1.], alpha=0.2, colors=colors[i])
    cntr = plt.contour(X, Y, dlnL, levels=[0, 1.], alpha=1, colors=colors[i], linewidths=2)

     # Save contours from all experiments without (with) flavours for the last plot (lifetimes)
    if (i == 3):
        cntr_all_chann = cntr
    if (i == 4):
        cntr_all_chann_and_flavour = cntr
        
    plt.text(coords[i][0], coords[i][1], labels[i], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color=colors[i], fontsize=17)


## SM prediction
dGs_SM, dGs_err_SM = 0.088, 0.020  # ps^-1
left_bottom = (Gsmin, dGs_SM - dGs_err_SM)
width = Gsmax - Gsmin
height = 2 * dGs_err_SM
cosmetics = {'linewidth':1, 'edgecolor':'none', 'facecolor':'khaki',
             'alpha':0.75}
rec = patches.Rectangle(left_bottom, width=width, height=height, **cosmetics)

# Add the rectangle to the axes
ax.add_patch(rec)
plt.text(0.9, 0.32, s="Theory", verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes, color='k', fontsize=15)

# Add plot description
plt.text(0.95, 0.75, "39% CL contours", verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes, color='k', fontsize=16)
plt.text(0.95, 0.70, r"($\Delta$ log $\mathcal{L}$ = 0.5)", verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='k', fontsize=18)

# Display ranges and axes ticks; axes titles
plt.axis([Gsmin, Gsmax, DGsmin, DGsmax])
plt.xticks(np.arange(Gsmin, Gsmax, .01))
plt.yticks(np.arange(DGsmin, DGsmax, .02))

hide_half_labels(ax)

# Axes settings and labels    
ax.set_autoscale_on(False)
plt.xlabel(r'$\Gamma_{s}[\mathrm{ps}^{-1}]$', fontsize=26)
plt.ylabel(r'$\Delta \Gamma_{s}[\mathrm{ps}^{-1}]$', fontsize=26)

saveplt(plt, name='Gs_vs_DGs')


# ### Analysis in terms of lifetimes

# In[ ]:

# lists of colours, labels, experiment channels, coordinates of labels
coords = [(0.5, 0.82), (0.95, 0.5), (0.9, 0.2), (0.45, 0.3), (0.42, 0.42)]

# all p.d.f.'s to be displayed (-2 log computed for all except one already with -2 log)
pdfs = [CP_even_tau, CP_odd_tau, flavour_specific_tau]

fig, ax = plt.subplots(1, figsize=(10,9))

## tau grid for computing and plotting
tauLmin, tauLmax, tauHmin, tauHmax = (1.24, 1.66, 1.44, 1.86)
tau_L, tau_H = np.linspace(tauLmin, tauLmax, 1000), np.linspace(tauHmin, tauHmax, 1000)
tau_L, tau_H = np.meshgrid(tau_L, tau_H)

# Draw contours
for i in range(len(coords)):
    if (i < 3):
        # compute delta log-likelihood
        lnL = -2 * np.log(pdfs[i](tau_L, tau_H))
        dlnL = lnL - lnL.min()

        cs = plt.contourf(tau_L, tau_H, dlnL, levels=[0, 1.], alpha=0.2, colors=colors[i])
        # skip contour edges for rectangular bands
        if (i != 0 and i != 1):
            plt.contour(tau_L, tau_H, dlnL, levels=[0, 1.], alpha=0.8, colors=colors[i], linewidths=1)

    plt.text(coords[i][0], coords[i][1], labels[i], verticalalignment='bottom', 
             horizontalalignment='right', transform=ax.transAxes, color=colors[i], fontsize=17)

def extract_pyplot_contour(cs, which=1):
    #for level in range(len(cs)):
    p = cs.collections[which].get_paths()[0]
    v = p.vertices
    xx = v[:,0]
    yy = v[:,1]
    return (xx, yy)


# Get contour of all channels without (with) flavour
taul_ach, tauh_ach = translate_gamma2tau(*extract_pyplot_contour(cntr_all_chann))
taul_achf, tauh_achf = translate_gamma2tau(*extract_pyplot_contour(cntr_all_chann_and_flavour))

# Fill contour for all experiments without (with) flavours
plt.fill(taul_ach, tauh_ach, color=colors[3], alpha=0.8)
plt.fill(taul_achf, tauh_achf, color=colors[4], alpha=0.8)

# Add plot description
plt.text(0.95, 0.75, "39% CL contours", verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes, color='k', fontsize=16)
plt.text(0.95, 0.70, r"($\Delta$ log $\mathcal{L}$ = 0.5)", verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='k', fontsize=18)

# Display ranges and axes ticks; axes titles
plt.axis([tauLmin, tauLmax, tauHmin, tauHmax])
# # Axes settings and labels    
ax.set_autoscale_on(False)
plt.xticks(np.arange(tauLmin, tauLmax, .02))
plt.yticks(np.arange(tauHmin, tauHmax, .02))
# hide every other tick-label
for label in ax.get_xticklabels()[1::2]:
    label.set_visible(False)
for label in ax.get_yticklabels()[1::2]:
    label.set_visible(False)    
plt.xlabel(r'$1/\Gamma_{L}[\mathrm{ps}]$', fontsize=26)
plt.ylabel(r'$1/\Gamma_{H}[\mathrm{ps}]$', fontsize=26)
saveplt(plt, 'tauL_vs_tauH')