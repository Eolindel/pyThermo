#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determination of the shape of binary diagram for an ideal system, (example gold/silver)

The vector x corresponds to the unknown variables which are 
xG, xS :  the molar fractions in the liquid phase
yG, yS : the molar fractions in the gas phase
The temperature

x[0] = xG
x[1] = xS
x[2] = yG
x[3] = yS
x[4] = T

For the five unknowns, we need five equations :
    xG+xS = 1 for the solid phase
    yG+yS = 1 for the liquid phase
    from the equality of chemical potential in the liquid and solid phase, the expression of the chemical potential in both phases and the integration of the Gibbs Helmholtz relationship
        and if we suppose that the enthalpy of fusion is constant :
        
    ln(y_i/x_i)= \Delta_fus H° / R * (1/T_fus,i-1/T)
    and we want to solve the full system for a given value of xG = x0

The system is expressed in a form such that AX = 0

From the equations, it is then possible to get the total quantity of liquid and solid (n_l(T), n_s(T) with n_l+n_s = n_tot=1) from the «Théorème des moments» (the relation stating that the overall composition of the systm is the barycentre of the solid and liquid phases with their given molar fractions)


The cooling curve is assumed for an hypothesis where the loss of enthalpy is constant with respect to time. A small trick is used : the enthalpy of the system is computed as a function of Temperature from the composition of the system.

H = sum_i n_i(T)H_i(T) 
(with H_i = H_i(T=298.15)+c_p(T-298.15))

Then, the hypothesis means that plotting T =f(time) is equivalent to plotting the inverse function of T with respect to the enthalpy. A linear tranformation is done to convert the enthalpy range to a time between 0 and 1.


All the thermodynamic values are taken from Wikipedia and the engineering toolbox website.

Informations
------------
Author : Martin Vérot  from the ENS de Lyon, France, with the help of some scripts to create the buttons and sliders taken from https://github.com/araoux/python_agregation (written by P Cladé, A Raoux and F Levrier)
Licence : Creative Commons CC-BY-NC-SA 4.0

WARNING this program requires the widgets.py file to work
"""
#Libraries
import numpy as np
from scipy import optimize
import scipy.constants as constants
import math
import matplotlib.pyplot as plt
import matplotlib
import widgets

def Equations(x,T0,Tfus,Hfus):
    """
    System to solve
    """
    R=constants.R
    return [x[0]+x[1]-1,#x1+x2=1
             x[2]+x[3]-1,#y1+y2=1
             (x[2])-x[0]*np.exp(Hfus[0]/R*(1/Tfus[0]-1/x[4])),
             (x[3])-x[1]*np.exp(Hfus[1]/R*(1/Tfus[1]-1/x[4])) ,
             x[4]-T0  ]#x = x0

def molarEnthalpy(Hs,Hl,Cps,Cpl,T):
    """
    Returns the molar enthalpy of the liquid (first) and gas over a given range of temperature (second)
        Hl contains the liquid enthalpy of formation of benzene and toluene 
        Hg contains the vapor enthalpy of formation of benzene and toluene 
        Cpl contains the heat capacity of benzene and toluene in their liquid phase 
        Cpg contains the heat capacity of benzene and toluene in their vapor phase 
        T is the temperature
    """
    Hs = np.reshape(Hs,(1,2))
    #print(Hl.shape)
    Temp = np.reshape(T,(-1,1))
    Hs = Hs*np.ones_like(Temp)
    Hl = Hl*np.ones_like(Temp)
    Cps = Cps*np.ones_like(Temp)
    Cpl = Cpl*np.ones_like(Temp)
    return Hs + (Temp-298.15)* Cps, Hl + (Temp-298.15)*Cpl
     
def FindTliqMel(x0,arr):
    """
    Finding the temperature corresponding to the Liquefaction point and Melting point from the solution of the binary diagram for the given initial composition : xG = x0
        x0 given molar fraction of the overall system
        arr solutions for the equation of the binary diagram
    Return two values : the Temperature of the solidifying point and the Melting temperature
    """
    miny = np.abs(arr[:,2]-x0)
    indixy = np.argmin(miny)
    minx = np.abs(arr[:,0]-x0)
    indixx = np.argmin(minx)
    return arr[indixy,4],arr[indixx,4]

def SysComposition(Temps,arr,TempRange,x0):
    """
    Returns the composition of the system (total = 1 mol) as a function of temperature 
        Temps are the solidifying point and Melting temperatures for the given composition
        arr solutions for the equation of the binary diagram
        Temprange the range of temperatures for which we can solve the problem
        x0 given molar fraction of the overall system
    
    Returns : the total quantity of liquid phase, the benzene molar fraction in the liquid phase and the correspoding value for the vapor phase
    
    """
    ns = np.zeros_like(TempRange)
    #Below the solidus point, there is only one solid phase
    ns[TempRange<=Temps[1]]=1
    #Above the liquidus Point, there is only a liquid phase (useless as the array is made of zeros initially, but given here for an educational purpose)
    ns[TempRange>Temps[0]]=0
    #Finding the ratio n_s/n_l from x0*n_tot=x*n_s+y*n_l
    lov = (arr[:,2]-x0)/(x0-arr[:,0]) 
    mom = lov/(1+lov)
    ns[(TempRange<=Temps[0]) & (TempRange > Temps[1])]=mom[(TempRange<=Temps[0]) & (TempRange > Temps[1])]
    #xG finding the composition of the liquid phase at each temperature
    x = np.zeros_like(TempRange)
    x[TempRange<=Temps[1]]=x0 #below the Melting point, the liquid has the stated composition
    #in the liquid vapor part, we have to read the xG value in the array containing the solution
    x[(TempRange<=Temps[0]) & (TempRange > Temps[1])]=arr[(TempRange<=Temps[0]) & (TempRange > Temps[1]),0]
    #yG finding the vapor composition of the liquid phase at each temperature
    y = np.zeros_like(TempRange)
    y[TempRange>Temps[0]]=x0
    y[(TempRange<=Temps[0]) & (TempRange > Temps[1])]=arr[(TempRange<=Temps[0]) & (TempRange > Temps[1]),2]
    return ns,x,y
     
def plot_data(x):
    Temps = FindTliqMel(x,arrsol)
    compSys = SysComposition(Temps,arrsol,TempRange,x) 
    #Amount of Gold in the solid phase
    nGs = compSys[1]*compSys[0]
    #Amount of Silver in the solid phase
    nSs = (1-compSys[1])*compSys[0]
    #Amount of gold in the liquid phase
    nGl = compSys[2]*(1-compSys[0])
    #Amount of silver in the liquid phase
    nSl = (1-compSys[2])*(1-compSys[0])
    #Computing the molar enthalpy as a function of the system composition
    Hm = molarEnthalpy(Hs,Hl,Cps,Cpl,TempRange)
    #computing the enthalpy as a function of temperature
    H = nGs*Hm[0][:,0]+nGl*Hm[1][:,0]+nSs*Hm[0][:,1]+nSl*Hm[1][:,1]
    #H(T) is computed but as we assume that dH/dt = constant, it means that findig T(t) corresponds to reverting to T(H) so we just flip H and T to have a picture of temperature as a function of enthalpy 
    #and it is equivalent we then change things to have a cooling curve instead of a heating curve
    #H = np.flip(H)
    H = -H
    H = (H-min(H))/(max(H)-min(H))
    
    lines['cooling'].set_data(H,TempRange )
    #Plotting horizontal lines for the liquidus point
    lines['liql'].set_data([0.,1.],Temps[0] )
    lines['liqr'].set_data([0.,1.],Temps[0] )
    #Plotting horizontal lines for the solidus point
    lines['soll'].set_data([0.,1.],Temps[1] )
    lines['solr'].set_data([0.,1.],Temps[1] )
    #Plotting an horizontal line for the value of x0
    lines['x'].set_data([x,x],[min(TempRange),max(TempRange)] )
    ax4.set_title('Cooling curve for x = {:3.2f}'.format(x))
    fig.canvas.draw_idle()
     
if __name__ == "__main__":


    parameters = {
        'x' : widgets.FloatSlider(value=0.5, description='x_G', min=0.0001, max=0.999999),
    }
    #Save the data as a csv file
    saveCsv = True
    fileCsv = "diagbin.csv"
    Tfus = np.array([1337.33,1234.93]) #Gold, Silver
    Hfus = np.array([12.55e3,11.28e3])
    Cps = np.array([25.418,25.350])
    Cpl = np.array([29.22,30.66])
    Hs = np.array([0,0])
    Hl = Hs+Hfus

    #the list that will contain the solutions
    solutions = []
    if saveCsv == True:
        f = open(fileCsv, "w")
        f.write('#xG,xS,yG,yS,T\n')

    TempRange = np.arange(min(Tfus), max(Tfus), 0.01)

    for T0 in TempRange:
        sol = optimize.root(Equations, [0, 1,0,1,383.8],args=(T0,Tfus,Hfus), jac=False, method='hybr')
        solutions.append(sol.x)

        if saveCsv == True:
            f.write('{:05.4f}'.format(sol.x[0]) +','+'{:05.4f}'.format(sol.x[1])+','+'{:05.4f}'.format(sol.x[2])+','+'{:05.4f}'.format(sol.x[3])+','+'{:05.4f}'.format(sol.x[4])+'\n')
    if saveCsv == True:
        f.close( )
    #Numpy array containing the solutions 0 : xG, 1 : xS, 2 : yG, 3 : yS, 4 : T 
    arrsol = np.asarray(solutions)
     

    fig,axes = plt.subplots(1,2,figsize=(10,6))
    ax = fig.add_axes([0.35, 0.3, 0.6, 0.6])
    #Binary Diagram
    ax1 = plt.subplot(1,2,1)
    ax1.set_title('Binary diagram for the gold/silver system')
    ax1.set_xlabel('x_G')
    ax1.set_ylabel('T (K)')
    ax1.set_xlim(0.,1.)
    ax1.set_ylim(min(Tfus)-5.,max(Tfus)+5.)
    ax1.plot(arrsol[:,0],arrsol[:,4], label='solidus')
    ax1.plot(arrsol[:,2],arrsol[:,4], label='liquidus')
    ax1.legend(loc='upper right')
    
    ##Cooling curves of the system
    ax4 = plt.subplot(1,2,2)
    ax4.set_xlabel('time (arbitrary unit)')
    ax4.set_ylabel(' T (K)')
    ax4.set_ylim(min(Tfus)-5.,max(Tfus)+5.)
    
    lines = {}
    lines['cooling'], = ax4.plot([],[], label='T(t)')
    #Plotting horizontal lines for the dew point
    lines['liql'], = ax1.plot([],[],color='#cccccc')  
    lines['liqr'], = ax4.plot([],[],color='#cccccc') 
    #Plotting horizontal lines for the Melting point
    lines['soll'], = ax1.plot([],[],color='#cccccc') 
    lines['solr'], = ax4.plot([],[],color='#cccccc')
    #Plotting an horizontal line for the value of x0
    lines['x'], = ax1.plot([],[],color='#cccccc')
   
    param_widgets = widgets.make_param_widgets(parameters, plot_data, slider_box=[0.35, 0.93, 0.4, 0.05])
    #choose_widget = widgets.make_choose_plot(lines, box=[0.015, 0.25, 0.2, 0.15])
    #reset_button = widgets.make_reset_button(param_widgets)

    plt.show()
    #plt.tight_layout()


#checks
    x=0.5
    Temps = FindTliqMel(x,arrsol)
    compSys = SysComposition(Temps,arrsol,TempRange,x) 
    nGs = compSys[1]*compSys[0]
    nGl = compSys[2]*(1-compSys[0])
    ax2 = plt.subplot(2,2,3)
    ax2.plot(TempRange,compSys[0], label='molar liquid phase amount')
    ax2.plot(TempRange,compSys[1], label='xG')
    ax2.plot(TempRange,compSys[2], label='yG')
    ax2.plot(TempRange,nGs, label='nB(l)')
    ax2.plot(TempRange,nGl, label='nB(g)')
    #Verification of the total quantity of Gold
    ax2.plot(TempRange,nGl+nGs, label='nBtot')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.1,1.1)

    #ax3 = plt.subplot(2,2,3)
    #ax2.plot(TempRange,compSys[0], label='molar liquid phase amount')
    #ax3.plot(compSys[1],TempRange, label='xG')
    #ax3.plot(compSys[2],TempRange, label='yG')
    #ax3.plot(nGs,TempRange, label='nB(l)')
    #ax3.plot(nSs,TempRange, label='nT(l)')
    #ax3.plot(nGl,TempRange, label='nB(g)')
    #ax3.plot(nSl,TempRange, label='nT(g)')
    #Verification of the total quantity of Gold
    #ax3.plot(TempRange,nGl+nGs, label='nBtot')
    #ax3.legend(loc='upper right')
    #ax3.set_xlim(-0.,1.)
 


