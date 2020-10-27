#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determination of the shape of binary diagram for an ideal system, (example benzene/toluene)

The vector x corresponds to the unknown variables which are 
xB, xT :  the molar fractions in the liquid phase
yB, yT : the molar fractions in the gas phase
The temperature

x[0] = xB
x[1] = xT
x[2] = yB
x[3] = yT
x[4] = T

For the five unknowns, we need five equations :
    xB+xT = 1 for the liquid phase
    yB+yT = 1 for the vapor phase
    from the equality of chemical potential in the liquid and vapor phase, the expression of the chemical potential in both phases and the integration of the Gibbs Helmholtz relationship
        and if we suppose that the enthalpy of vaporisation is constant :
        
    ln(y_i/x_i)= \Delta_vap H° / R * (1/T_eb,i-1/T)
    and we want to solve the full system for a given value of xB = x0

The system is expressed in a form such that AX = 0

From the equations, it is then possible to get the total quantity of liquid and vapor (n_l(T), n_g(T) with n_l+n_g = n_tot=1) from the «Théorème des moments» (the relation stating that the overall composition of the systm is the barycentre of the vapor and liquid phases with their given molar fractions)


The cooling curve is assumed for an hypothesis where the loss of enthalpy is constant with respect to time. A small trick is used : the enthalpy of the system is computed as a function of Temperature from the composition of the system.

H = sum_i n_i(T)H_i(T) 
(with H_i = H_i(T=298.15)+c_p(T-298.15))

Then, the hypothesis means that plotting T =f(time) is equivalent to plotting the inverse function of T with respect to the enthalpy. A linear tranformation is done to convert the enthalpy range to a time between 0 and 1.


All thei thermodynamic values are taken from the NIST website.

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

def Equations(x,T0,Teb,Hvap):
    """
    System to solve
    """
    R=constants.R
    return [x[0]+x[1]-1,#x1+x2=1
             x[2]+x[3]-1,#y1+y2=1
             (x[2])-x[0]*np.exp(Hvap[0]/R*(1/Teb[0]-1/x[4])),
             (x[3])-x[1]*np.exp(Hvap[1]/R*(1/Teb[1]-1/x[4])) ,
             x[4]-T0  ]#x = x0

def molarEnthalpy(Hl,Hg,Cpl,Cpg,T):
    """
    Returns the molar enthalpy of the liquid (first) and gas over a given range of temperature (second)
        Hl contains the liquid enthalpy of formation of benzene and toluene 
        Hg contains the vapor enthalpy of formation of benzene and toluene 
        Cpl contains the heat capacity of benzene and toluene in their liquid phase 
        Cpg contains the heat capacity of benzene and toluene in their vapor phase 
        T is the temperature
    """
    Hl = np.reshape(Hl,(1,2))
    #print(Hl.shape)
    Temp = np.reshape(T,(-1,1))
    Hl = Hl*np.ones_like(Temp)
    Hg = Hg*np.ones_like(Temp)
    Cpl = Cpl*np.ones_like(Temp)
    Cpg = Cpg*np.ones_like(Temp)
    return Hl + (Temp-298.15)* Cpl, Hg + (Temp-298.15)*Cpg
     
def FindTdewTeb(x0,arr):
    """
    Finding the temperature corresponding to the Dew point and EBullition point from the solution of the binary diagram for the given initial composition : xB = x0
        x0 given molar fraction of the overall system
        arr solutions for the equation of the binary diagram
    Return two values : the Temperature of the dew point and the ebullition temperature
    """
    miny = np.abs(arr[:,2]-x0)
    indixy = np.argmin(miny)
    minx = np.abs(arr[:,0]-x0)
    indixx = np.argmin(minx)
    return arr[indixy,4],arr[indixx,4]

def SysComposition(Temps,arr,TempRange,x0):
    """
    Returns the composition of the system (total = 1 mol) as a function of temperature 
        Temps are the dew point and ebullition temperatures for the given composition
        arr solutions for the equation of the binary diagram
        Temprange the range of temperatures for which we can solve the problem
        x0 given molar fraction of the overall system
    
    Returns : the total quantity of liquid phase, the benzene molar fraction in the liquid phase and the correspoding value for the vapor phase
    
    """
    nl = np.zeros_like(TempRange)
    #Below the Ebulltion point, there is only one liquid phase
    nl[TempRange<=Temps[1]]=1
    #Above the Dew Point, there is only a vapor phase (useless as the array is made of zeros initially, but given here for an educational purpose)
    nl[TempRange>Temps[0]]=0
    #Finding the ratio n_l/n_v from x0*n_tot=x*n_l+y*n_g
    lov = (arr[:,2]-x0)/(x0-arr[:,0]) 
    mom = lov/(1+lov)
    nl[(TempRange<=Temps[0]) & (TempRange > Temps[1])]=mom[(TempRange<=Temps[0]) & (TempRange > Temps[1])]
    #xB finding the composition of the liquid phase at each temperature
    x = np.zeros_like(TempRange)
    x[TempRange<=Temps[1]]=x0 #below the ebullition point, the liquid has the stated composition
    #in the liquid vapor part, we have to read the xB value in the array containing the solution
    x[(TempRange<=Temps[0]) & (TempRange > Temps[1])]=arr[(TempRange<=Temps[0]) & (TempRange > Temps[1]),0]
    #yB finding the vapor composition of the vapor phase at each temperature
    y = np.zeros_like(TempRange)
    y[TempRange>Temps[0]]=x0
    y[(TempRange<=Temps[0]) & (TempRange > Temps[1])]=arr[(TempRange<=Temps[0]) & (TempRange > Temps[1]),2]
    return nl,x,y
     
def plot_data(x):
    Temps = FindTdewTeb(x,arrsol)
    compSys = SysComposition(Temps,arrsol,TempRange,x) 
    #Amount of Benzene in the liquide phase
    nBl = compSys[1]*compSys[0]
    #Amount of Toluene in the liquide phase
    nTl = (1-compSys[1])*compSys[0]
    #Amount of benzene in the vapor phase
    nBg = compSys[2]*(1-compSys[0])
    #Amount of toluene in the vapor phase
    nTg = (1-compSys[2])*(1-compSys[0])
    #Computing the molar enthalpy as a function of the system composition
    Hm = molarEnthalpy(Hl,Hg,Cpl,Cpg,TempRange)
    #computing the enthalpy as a function of temperature
    H = nBl*Hm[0][:,0]+nBg*Hm[1][:,0]+nTl*Hm[0][:,1]+nTg*Hm[1][:,1]
    #H(T) is computed but as we assume that dH/dt = constant, it means that findig T(t) corresponds to reverting to T(H) so we just flip H and T to have a picture of temperature as a function of enthalpy 
    #and it is equivalent we then change things to have a cooling curve instead of a heating curve
    #H = np.flip(H)
    H = -H
    H = (H-min(H))/(max(H)-min(H))
    
    lines['cooling'].set_data(H,TempRange )
    #Plotting horizontal lines for the dew point
    lines['dewl'].set_data([0.,1.],Temps[0] )
    lines['dewr'].set_data([0.,1.],Temps[0] )
    #Plotting horizontal lines for the ebullition point
    lines['ebl'].set_data([0.,1.],Temps[1] )
    lines['ebr'].set_data([0.,1.],Temps[1] )
    #Plotting an horizontal line for the value of x0
    lines['x'].set_data([x,x],[min(TempRange),max(TempRange)] )
    ax4.set_title('Cooling curve for x = {:3.2f}'.format(x))
    fig.canvas.draw_idle()
     
if __name__ == "__main__":


    parameters = {
        'x' : widgets.FloatSlider(value=0.5, description='x_B', min=0.0001, max=0.999999),
    }
    #Save the data as a csv file
    saveCsv = True
    fileCsv = "diagbin.csv"
    Teb = np.array([353.3,383.8]) #Benzene, Toluene
    Hvap = np.array([30.72e3,33.18e3])
    Cpl = np.array([135.6,157.])
    Cpg = np.array([82.44,103.7,])
    Hl = np.array([49.e3,12.e3])
    Hg = Hl+Hvap

    #the list that will contain the solutions
    solutions = []
    if saveCsv == True:
        f = open(fileCsv, "w")
        f.write('#xB,xT,yB,yT,T\n')

    TempRange = np.arange(min(Teb), max(Teb), 0.01)

    for T0 in TempRange:
        sol = optimize.root(Equations, [0, 1,0,1,383.8],args=(T0,Teb,Hvap), jac=False, method='hybr')
        solutions.append(sol.x)

        if saveCsv == True:
            f.write('{:05.4f}'.format(sol.x[0]) +','+'{:05.4f}'.format(sol.x[1])+','+'{:05.4f}'.format(sol.x[2])+','+'{:05.4f}'.format(sol.x[3])+','+'{:05.4f}'.format(sol.x[4])+'\n')
    if saveCsv == True:
        f.close( )
    #Numpy array containing the solutions 0 : xB, 1 : xT, 2 : yB, 3 : yT, 4 : T 
    arrsol = np.asarray(solutions)
     

    fig,axes = plt.subplots(1,2,figsize=(10,6))
    ax = fig.add_axes([0.35, 0.3, 0.6, 0.6])
    #Binary Diagram
    ax1 = plt.subplot(1,2,1)
    ax1.set_title('Binary diagram for the toluene/benzene system')
    ax1.set_xlabel('x_B')
    ax1.set_ylabel('T (K)')
    ax1.set_xlim(0.,1.)
    ax1.set_ylim(min(Teb)-5.,max(Teb)+5.)
    ax1.plot(arrsol[:,0],arrsol[:,4], label='bubble point curve')
    ax1.plot(arrsol[:,2],arrsol[:,4], label='dew point curve')
    ax1.legend(loc='upper right')
    
    ##Cooling curves of the system
    ax4 = plt.subplot(1,2,2)
    ax4.set_xlabel('time (arbitrary unit)')
    ax4.set_ylabel(' T (K)')
    ax4.set_ylim(min(Teb)-5.,max(Teb)+5.)
    
    lines = {}
    lines['cooling'], = ax4.plot([],[], label='T(t)')
    #Plotting horizontal lines for the dew point
    lines['dewl'], = ax1.plot([],[],color='#cccccc')  
    lines['dewr'], = ax4.plot([],[],color='#cccccc') 
    #Plotting horizontal lines for the ebullition point
    lines['ebl'], = ax1.plot([],[],color='#cccccc') 
    lines['ebr'], = ax4.plot([],[],color='#cccccc')
    #Plotting an horizontal line for the value of x0
    lines['x'], = ax1.plot([],[],color='#cccccc')
   
    param_widgets = widgets.make_param_widgets(parameters, plot_data, slider_box=[0.35, 0.93, 0.4, 0.05])
    #choose_widget = widgets.make_choose_plot(lines, box=[0.015, 0.25, 0.2, 0.15])
    #reset_button = widgets.make_reset_button(param_widgets)

    #plt.tight_layout()
    plt.show()

    #ax2 = plt.subplot(2,2,2)
    #ax2.plot(TempRange,compSys[0], label='molar liquid phase amount')
    #ax2.plot(TempRange,compSys[1], label='xB')
    #ax2.plot(TempRange,compSys[2], label='yB')
    #ax2.plot(TempRange,nBl, label='nB(l)')
    #ax2.plot(TempRange,nBg, label='nB(g)')
    #Verification of the total quantity of Benzene
    #ax2.plot(TempRange,nBg+nBl, label='nBtot')
    #ax2.legend(loc='upper right')
    #ax2.set_ylim(-0.1,1.1)

    #ax3 = plt.subplot(2,2,3)
    #ax2.plot(TempRange,compSys[0], label='molar liquid phase amount')
    #ax3.plot(compSys[1],TempRange, label='xB')
    #ax3.plot(compSys[2],TempRange, label='yB')
    #ax3.plot(nBl,TempRange, label='nB(l)')
    #ax3.plot(nTl,TempRange, label='nT(l)')
    #ax3.plot(nBg,TempRange, label='nB(g)')
    #ax3.plot(nTg,TempRange, label='nT(g)')
    #Verification of the total quantity of Benzene
    #ax3.plot(TempRange,nBg+nBl, label='nBtot')
    #ax3.legend(loc='upper right')
    #ax3.set_xlim(-0.,1.)
 


