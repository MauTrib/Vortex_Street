# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:55:04 2018

@author: MauTrib
"""
import inspect
import numpy as np

class Vortex_Object:
    
    demanded = [] #
    
    def __init__(self):
        pass
    
    def function(self,**kwargs):
        pass
    
    def update(self,**kwargs):
        """
        Sera appelé à chaque temps pour update l'objet
        """
        if (kwargs)=={}:
            self.function()
        else:
            self.function(self,**kwargs)
        return None
    
    def set_function(self,func):
        sig = inspect.signature(func)
        self.demanded = []
        for key in sig.parameters:
            if not(key in ["objet","args","kwargs"]):
                self.demanded.append	(key)
        self.function = func
        return None
    
    def get_mask(self,xx,yy):
        pass

class Rotating_Object(Vortex_Object):
    
    theta = 0
    period = 50
    
    def rotation(self,x,y,theta):
        x_r = x*np.cos(theta) - y*np.sin(theta)
        y_r = x*np.sin(theta)+ y*np.cos(theta)
        return x_r,y_r


def constant_rotation(objet,dt):
    objet.theta += 2*np.pi*dt/objet.period