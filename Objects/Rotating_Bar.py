import numpy as np

class R_Bar:
    
    def __init__(self,larg=1,long=1,x_c=0,y_c=0,theta=0):
        self.x_c = x_c
        self.y_c = y_c
        self.center = (x_c,y_c)
        
        self.long = long
        self.larg = larg
        self.theta = theta
    
    def get_mask(self,xx,yy):
        """Renvoie un masque avec 1 disant que le point est dans la barre, 0 sinon
        Entrées: - xx : meshgrid selon x
                 - yy : meshgrid selon y
        Sortie : - masque : masque correspondant
        """
        #Translation
        xx -= self.x_c
        yy -= self.y_c
        #Rotation  dans le repère 
        xx_p = xx*np.cos(self.theta) + yy*np.sin(self.theta)
        yy_p = yy*np.cos(self.theta) - xx*np.sin(self.theta)
        
        def f(x,A):
            return np.heaviside(x+A,1) - np.heaviside(x-A,1)
        masque = np.heaviside(f(xx_p,self.larg/2) + f(yy_p,self.long/2)-1.5,0,dtype=int)
        return masque
        
    
    def __repr__(self):
        return "Rotating Bar object\nCenter={}, Angle={}, dimensions={}".format(self.center,self.theta,(self.long,self.larg))






bar = R_Bar(1,1,np.pi/2)