#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:20:53 2018

@author: alienor
"""

"""
Fonctions
"""

import matplotlib.pyplot as plt # affichage
import numpy as np # calculs
import os
import scipy.sparse as sp # matrices (pour Poisson)
import scipy.sparse.linalg as lg #matrices (pour Poisson)
from PyQt5 import QtWidgets # affichage (reactualisation)
import matplotlib.colors as colors # faire des jolies couleurs
import time as time # montrer_performances (affiche la durée réelle du programme)

### FONCTIONS

  # disptime(t) : affiche une durée t en secondes, minutes, etc...
def disptime(t):
    s = int(t)
    m = s//60
    s %= 60
    h = m//60
    m %=60
    if (h >0): return(str(h)+" h et "+str(m)+" min")
    elif (m>15): return(str(m)+" min")
    elif (m>0): return(str(m)+" min et "+str(s)+" s")
    elif (t>20): return(str(s)+" s")
    elif (t>10): return(str(round(t, 1))+" s")
    elif (t>5): return(str(round(t, 2))+" s")
    elif (t>1): return(str(round(t, 4))+" s")
    elif (t > 0.00001): return(str(round(t, 6))+" s")
    else : return(str(t)+" s")

  # CFL_advection() : donne le dt maximal adapté à l'étape d'advection
def CFL_advection():
    global u,v, U, dmin, precautionADV
    umax = max( np.amax(np.abs(v)) , np.amax(np.abs(u)) , U)
    dt_cfa= dmin/umax * precautionADV
    return dt_cfa
    
  # CFL_diffusion() : donne le dt maximal adapté à l'étape de diffusion
def CFL_diffusion():
    global D, dmin, precautionDIFF
    dt_cfl = dmin**2/(4*D) * precautionDIFF
    return dt_cfl

  # Advect() : renvoie la valeur interpolée qui correspond à l'advection de la vitesse
def Advect():
    global NX, NY, dx, dy, dt, atot, Resu, Resv, Resc, u, v
    
    u_r = - u[1:-1,1:-1]
    v_r = - v[1:-1,1:-1]
    
    Mu_p = (np.sign(u_r)+1)/2 # Mu_p[i,j]=1 si u_r[i,j]>=0, sinon c'est 0 (=Mu_positive)
    Mu_n = 1-Mu_p    # Inverse de Mu_p (Mu_negative)
    Mv_p = (np.sign(v_r)+1)/2
    Mv_n = 1-Mv_p
    
    
    a_center = (dx-np.abs(u_r)*dt)*(dy-np.abs(v_r)*dt)/atot
    a_updown = (dx-np.abs(u_r)*dt)*np.abs(v_r)*dt/atot
    a_rl = np.abs(u_r)*dt*(dy-np.abs(v_r)*dt)/atot
    a_diag = np.abs(u_r*dt*v_r*dt)/atot
    
    # Calcul des matrices de resultat pour les vitesses u et v
    
    Resu[1:-1,1:-1] = a_center*u[1:-1,1:-1] + a_diag*(Mu_p*Mv_p*u[:-2,2:]+Mu_p*Mv_n*u[2:,2:]+Mu_n*Mv_p*u[:-2,:-2]+Mu_n*Mv_n*u[2:,:-2]) + a_rl*(Mu_p*u[1:-1,2:]+Mu_n*u[1:-1,:-2])+a_updown*(Mv_p*u[:-2,1:-1]+Mv_n*u[2:,1:-1])
    
    Resv[1:-1,1:-1] = a_center*v[1:-1,1:-1] + a_diag*(Mu_p*Mv_p*v[:-2,2:]+Mu_p*Mv_n*v[2:,2:]+Mu_n*Mv_p*v[:-2,:-2]+Mu_n*Mv_n*v[2:,:-2]) + a_rl*(Mu_p*v[1:-1,2:]+Mu_n*v[1:-1,:-2])+a_updown*(Mv_p*v[:-2,1:-1]+Mv_n*v[2:,1:-1])
    
    Resc[1:-1,1:-1] = a_center*c[1:-1,1:-1] + a_diag*(Mu_p*Mv_p*c[:-2,2:]+Mu_p*Mv_n*c[2:,2:]+Mu_n*Mv_p*c[:-2,:-2]+Mu_n*Mv_n*c[2:,:-2]) + a_rl*(Mu_p*c[1:-1,2:]+Mu_n*c[1:-1,:-2])+a_updown*(Mv_p*c[:-2,1:-1]+Mv_n*c[2:,1:-1])

    return 0

def BuildLaPoisson():
    """pour l'étape de projection matrice de Laplacien phi avec CL Neumann pour phi BUT condition de Neumann pour phi ==> non unicite de la solution 
  besoin de fixer la pression en un point pour lever la degenerescence: ici [0][1]==> need to build a correction matrix"""
    global nx,ny,dx_2,dy_2                        # ne pas prendre en compte les points fantome (-2)
    ###### Definition de l'opérateur de Laplace 1D
    ###### AXE X
    datanx = [np.ones(nx), -2*np.ones(nx), np.ones(nx)]      # Termes diagonaux
    # Conditions aux limites
    datanx[2][1]=2 # Gauche
    datanx[0][-2]=0# Droite
    ###### AXE Y
    datany = [np.ones(ny), -2*np.ones(ny), np.ones(ny)]     # Termes diagonaux
    # Conditions aux limites
    datany[2][1]=2  #Haut
    datany[0][-2]=2  # Bas
    ###### POSITIONS
    offsets = np.array([-1,0,1])                    
    DXX = sp.dia_matrix((datanx,offsets), shape = (nx,nx)) * dx_2
    DYY = sp.dia_matrix((datany,offsets), shape = (ny,ny)) * dy_2
    ###### Opérateur de Laplace 2D
    LAP = sp.kron(sp.eye(ny,ny), DXX) + sp.kron(DYY, sp.eye(nx,nx)) # eye donne l'identity, kron le produit de kronecker (produit tensoriel)
    ###### BUILD CORRECTION MATRIX
    ## Upper Diagonal terms
    #datanynx = [np.zeros(ny*nx)]
    #offset = np.array([1])
    ### Fix coef: 2+(-1) = 1 ==> Dirichlet en un point (redonne Laplacien)
    ### ATTENTION  COEF MULTIPLICATIF : dx_2 si M(j,i) j-ny i-nx
    #datanynx[0][1] = -1 * dx_2
    #LAP0 = sp.dia_matrix((datanynx,offset), shape=(ny*nx,ny*nx))
    return LAP #+ LAP0

def ILUdecomposition(LAP):
    """return the Incomplete LU decomposition of a sparse matrix LAP"""
    return  lg.splu(LAP.tocsc(),)


def ResoLap(splu,RHS):
    """solve the system SPLU * x = RHS
    Args:--RHS: 2D array((NY,NX))
         --splu: (Incomplete) LU decomposed matrix shape (NY*NX, NY*NX)
    Return: x = array[NY,NX]
    Rem1: taille matrice fonction des CL"""
    # array 2D -> array 1D
    f2 = RHS.ravel()
    # Solving the linear system
    x = splu.solve(f2)
    return x.reshape(RHS.shape)

def Laplacien(f):
    """Calcule le laplacien scalaire du champ scalaire f avec points fantomes"""
    global NX,NY,dx_2,dy_2
    lapv = np.zeros((NY,NX))
  # ecrit le 1811210940 par G
    lapv[1:-1,1:-1] = ( f[2: ,1:-1] - 2*f[1:-1,1:-1] + f[ :-2,1:-1] ) * dy_2 + ( f[1:-1,2: ] - 2*f[1:-1,1:-1] + f[1:-1, :-2] ) * dx_2   # derivee selon y
    return lapv
    
def divergence(u,v):
    """Divergence à l'ordre 2 avec points fantomes
  On utilise la formule de la dérivée du(x)/dx = (u(x+dx)-u(x-dx))/(2dx)
  """                          
    global NX,NY,dx,dy
    divv = np.zeros((NY,NX))     
    divv[1:-1,1:-1] = (u[1:-1,2:]-u[1:-1,:-2])/(2*dx) + (v[:-2:,1:-1]-v[2:,1:-1])/(2*dy)
    return divv
    
def grad(f):
    """Gradient de f à l'ordre 2
  Même méthode que pour la divergence ici, pour l'ordre 2"""
    global NX,NY,dx,dy
    gradfx = np.zeros((NY,NX))
    gradfy = np.zeros((NY,NX))
    gradfy[1:-1,:] = (f[:-2,:]-f[2:,:])/(2*dy)
    gradfx[:,1:-1] = (f[:,2:]-f[:,:-2])/(2*dx)
    return [gradfx,gradfy]

def rot(u,v):
    global dx, dy
    RRR = np.zeros([ny, nx])
    RRR = (v[1:-1,2:]-v[1:-1,:-2])/(2*dx) - (u[:-2,1:-1]-u[2:,1:-1])/(2*dy) 
    return RRR

        
def PhiGhostPoints(phi):
    """
    copie les points fantomes
    toujours Neumann

    global ==> pas de return 

    """
    ### left               
    phi[1:-1,0] = phi[1:-1,2]
    ### right             
    phi[1:-1,-1] = np.zeros(ny)
    ### bottom   
    phi[-1,1:-1] = phi[-3,1:-1]
    ### top               
    phi[0,1:-1] = phi[2,1:-1]
    
def condition_limites(u,g,d,h,b):
    """Définit les conditions aux limites sur les composantes de la vitesse.
    g, d, h, b sont des chaines de caractères ou un tableau de valeurs ou une constante qui peuvent valoir:
    -'grad', le gradient est nul
    -'nul' la composante est nulle (paroi)
    -les"""
    if g=='grad':
        u[1:-1,0] = u[1:-1,2]
    elif g=='nul':
        u[1:-1,0] = np.zeros(ny)
    elif type(g)==np.ndarray:
        u[1:-1,0] = g
    else:
        u[1:-1,0] = g*np.ones(ny)
        
    if d=='grad':
        u[1:-1,-1] = u[1:-1,-3]
    elif d=='nul':
        u[1:-1,-1]= np.zeros(ny)
    elif type(d)==np.ndarray:
        u[1:-1,-1] = d
    else:
        u[1:-1,-1] = d*np.ones(ny)
        
    if h=='grad':
        u[0,1:-1] = u[2,1:-1]
    elif h=='nul':
        u[0,1:-1] = np.zeros(nx)
    elif type(h)==np.ndarray:
        u[0,1:-1] = h
    else:
        u[0,1:-1] = h*np.ones(nx)
        
    if b=='grad':
        u[-1,1:-1] = u[-3,1:-1]
    elif b=='nul':
        u[-1,1:-1] = np.zeros(nx)
    elif type(b)==np.ndarray:
        u[-1,1:-1] = b
    else:
        u[-1,1:-1] = d*np.ones(nx)
        
        
   