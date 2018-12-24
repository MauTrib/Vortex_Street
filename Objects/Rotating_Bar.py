import numpy as np
from Objects.Vortex_Object import *
import matplotlib.pyplot as plt

class R_Bar(Rotating_Object):
    
    def __init__(self,larg=1,long=1,x_c=0,y_c=0,theta=0,resolution=100,period=50):
        self.x_c = x_c
        self.y_c = y_c
        self.center = (x_c,y_c)
        
        self.long = long
        self.larg = larg
        self.theta = theta
        
        self.period = period
        
        self.resolution = resolution
        self.set_contour(resolution)
        
    def get_mask(self,ox,oy):
        """Renvoie un masque avec 1 disant que le point est dans la barre, 0 sinon
        Entrées: - xx : meshgrid selon x
                 - yy : meshgrid selon y
        Sortie : - masque : masque correspondant
        """
        #Translation
        xx = ox - self.x_c
        yy = oy - self.y_c
        #Rotation  dans le repère 
        xx_p,yy_p = self.rotation(xx,yy,-self.theta)
        
        def f(x,A):
            return np.heaviside(x+A,1) - np.heaviside(x-A,1)
        masque = np.heaviside(f(xx_p,self.larg/2) + f(yy_p,self.long/2)-1.5,0).astype(bool)
        
        return masque
        
    
    def __repr__(self):
        return "Rotating Bar object\nCenter={}, Angle={}, dimensions={}".format(self.center,self.theta,(self.long,self.larg))

    def set_contour(self,resolution=100,proper_referential = False,debug=False):
        tot_long = 2*(self.long + self.larg)
        r_long = self.long/tot_long #Ratio de longueur, donnera le nombre de point
        r_larg = self.larg/tot_long
        y_coord = np.linspace(-self.long/2,self.long/2,int(r_long*resolution),endpoint=True)
        x_coord = np.linspace(-self.larg/2,self.larg/2,int(r_larg*resolution))[1:-1]
        y_fill = np.ones(len(x_coord))
        x_fill = np.ones(len(y_coord))
        
        c_x = np.concatenate((x_coord,(self.larg/2)*x_fill,x_coord[::-1],(-self.larg/2)*x_fill))
        c_y = np.concatenate(((self.long/2)*y_fill,y_coord[::-1],(-self.long/2)*y_fill,y_coord))
        #plt.scatter(c_x,c_y,label="Avant translation")
        if proper_referential:
            return c_x,c_y
        c_x,c_y = self.rotation(c_x,c_y,self.theta)
        #plt.scatter(c_x,c_y,label="Après rotation")
        c_x += self.x_c
        c_y += self.y_c
        #plt.scatter(c_x,c_y,label="Après translation")
        self.contour = (c_x,c_y)
        #plt.axis("equal")
        #plt.legend()
        
        return None
    
    def get_indices(self,xx,yy,debug=False):
        mesh_x = xx[0,:]
        mesh_y = yy[:,0]
        
        c_x,c_y = self.contour
        N = len(c_x)
        #Extension des dimensions
        c_x = np.expand_dims(c_x,0) #Transforme les shape (N,) en (1,N)
        c_y = np.expand_dims(c_y,0) #càd en vecteurs lignes qu'on peut transposer
        mesh_x = np.expand_dims(mesh_x,0)
        mesh_y = np.expand_dims(mesh_y,0)
        
        mesh_x = np.repeat(mesh_x,N,0)
        mesh_y = np.repeat(mesh_y,N,0)
        
        res_x = c_x.T - mesh_x #Calcule la distance (en positif et négatif)
        res_y = c_y.T - mesh_y#au point correspondant en gardant le tout vectorisé
        
        # Pénalisation des points négatifs par le maximum en valeur absolue de res
        res_x = np.abs(res_x) + np.heaviside(-res_x,0)*(np.max(np.abs(res_x))+1)
        res_y = np.abs(res_y) + np.heaviside(res_y,0)*(np.max(np.abs(res_y))+1)
        if debug:
            print("Selon x: ",res_x)
            print("Selon y: ",res_y)
        
        #Calcul des arguments minimaux
        x_indices = np.argmin(res_x,1)
        y_indices = np.argmin(res_y,1)
        if debug:
            plt.scatter(xx[y_indices,x_indices],yy[y_indices,x_indices],label="Approximation")
            plt.scatter(c_x,c_y,label="Contour")
            plt.legend()
        return x_indices,y_indices
        
        
        
def apply_pressure(objet,p,xx,yy,dt):
    objet.set_contour()
    i_x,i_y = objet.get_indices(xx,yy)
    c_x,c_y = objet.contour
    
    def ponder_list(l_x,l_y,xx,yy,p,i_x,i_y):
        l_p = []
        dx = xx[0,1]-xx[0,0]
        dy = yy[1,0]-yy[0,0]
        for k in range(len(l_x)):
            i,j = i_y[k],i_x[k]
            x,y,x_m,y_m = l_x[i],l_y[i],xx[i,j],yy[i,j]
            deltax = x-x_m
            deltay = y-y_m
            
            point_p = (1/(dx*dy))*(p[i,j]*((dy-deltay)*(dx-deltax)) + p[i,j+1]*(deltax*(dy-deltay)) + p[i+1,j+1]*(deltax*deltay) + p[i+1,j]*(deltay*(dx-deltax)))
            l_p.append(point_p)
        return l_p
    
    l_p = np.array(ponder_list(c_x,c_y,xx,yy,p,i_x,i_y))
    
    classe = type(objet).__name__
    if classe == 'R_Bar':
        b = objet.long
        c = objet.larg
        J = 1.8e3 * ((b*c)/12)*(b**2 + c**2) #Masse volumique fibre carbone
    else:
        J=1
        print("WARNING : object type {} has no defined Inertial moment in function 'apply_pressure'.\nInertial moment put to 1.".format(classe))
    
    if not(hasattr(objet,"theta_p")):
        objet.theta_p = 0
    
    c_x,c_y = objet.set_contour(proper_referential=True)
    x_frac,y_frac = np.abs(c_x)/np.max(c_x),np.abs(c_y)/np.max(c_y)
    
    ds = 2*(objet.long+objet.larg)/len(c_x)
    
    moments = np.sign(c_x)*np.sign(c_y)*np.sign((y_frac - x_frac)) #Sign
    moments = moments * (np.heaviside(y_frac-x_frac,0)*(np.abs(c_x)*l_p*ds) + np.heaviside(x_frac-y_frac,0)*np.abs(c_y)*l_p*ds) #Value
    
    print(np.sum(moments))
    
    theta_pp = (1/J)*np.sum(moments)
    objet.theta_p += theta_pp*dt
    objet.theta += objet.theta_p*dt
    return None
            
            


bar = R_Bar(1,2,x_c = 1,y_c=2,theta = np.pi/4)