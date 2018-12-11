import numpy as np

class Cylinder:
    
    def __init__(self,R=1,x_c=0,y_c=0):
        self.x_c = x_c
        self.y_c = y_c
        self.center=(x_c,y_c)
        self.R = R
    
    def get_mask(self,xx,yy):
        return ((xx-self.x_c)**2+(yy-self.y_c)**2 < self.R**2)
    
    def __repr__(self):
        return "Cylinder object : Center={}, Radius={}".format(self.center,self.R)
