# main program
   
# Import des autres fichiers


from Variables import *
from Objects.Vortex_Object import constant_rotation
from Objects.Cylinder import Cylinder
from Objects.Rotating_Bar import R_Bar


##FONCTIONS A GARDER ABSOLUMENT DANS CE FICHIER MAIN!
def get_vars(demanding_object):
    """
    A partir d'un objet (héritant de la classe Vortex_Object), renvoie la liste des variables nécessaires à l'exécution de la fonction intégrée à l'objet (None si l'objjet n'en a pas)
    """
    d_var = {}
    for key in demanding_object.demanded:
        try:
            d_var[key]=eval(key)
        except:
            print("Variable {} not found".format(key))
    return (d_var)

def Update_objects(l_objects):
    for objet in l_objects:
        d_var = get_vars(objet)
        objet.update(**d_var)
    return None

print("Début des calculs...")
t_deb = time.time()
while (dontstop):
    niter+=1
    
    
    dt = min(dt, CFL_advection(u,v,U,dmin,precautionADV), dt_exp) 
    
    t+=dt                                        #Avancement du temps total
  
    #COLORANT AJOUTE A LA PREMIERE COLONNE
    for i in pointsAjoutColorant:
        col[i,1] = 1
    
    #ETAPE D'ADVECTION SEMI-LAGRANGIENNE
    if (montrer_perf): t1 = time.time()
    Resu,Resv,Rescol = Advect(u,v,col,NX,NY,dx,dy,dt)
    if (montrer_perf) : t2 = time.time() ; tadv += t2-t1
    
    #ETAPE DE DIFFUSION
    if (RK2):
        ustar = Resu + D*Laplacien( u + dt/2 * D*Laplacien(u,dx_2,dy_2) ,dx_2,dy_2) * dt
        vstar = Resv + D*Laplacien( v + dt/2 * D*Laplacien(v,dx_2,dy_2) ,dx_2,dy_2) * dt
    else:
        ustar = Resu + D*Laplacien(u,dx_2,dy_2)*dt
        vstar = Resv + D*Laplacien(v,dx_2,dy_2)*dt
    
    if (montrer_perf) : t3 = time.time() ; tdiff += t3-t2
    
    #CONDITIONS AUX LIMITES SUR LES VITESSES ETOILES
     
   
    """ 
     #Avec les nouvelles conditions aux limites:
     #Possibilités 'grad', 'nul', un nombre ou un array
     #g,d,h,v (gauche, droite, haut, bas)
     g_u, g_v, d_u, d_v, h_u, h_v, b_u, b_v = ....
     conditions_limites(ustar,g_u,d_u,h_u,b_u)
     conditions_limites(vstar,g_v,d_v,h_v,b_v)
     #Obstacle:
     A faire: idée appliquer une liste de mask
    """  
    ConditionLimites(ustar,vstar,U)                        #Sur les bords du domaine
    
    ustar = Apply_objects(ustar,ox,oy,l_objects)                                           #Sur l'obstacle, penalisation
    vstar = Apply_objects(vstar,ox,oy,l_objects)

    #ETAPE DE PROJECTION
    divstar = divergence(ustar,vstar,dx,dy)                    #Calcul de la divergence de u*
    phi[1:-1,1:-1] = ResoLap(LUPoisson,divstar[1:-1,1:-1])        #Résolution du système
    PhiGhostPoints(phi)                                #Mise à jour des points fantomes de phi
    p = phi[1:-1,1:-1]/dt
    u = ustar-grad(phi,dx,dy)[0]                            #Calcul de u et v
    v = vstar-grad(phi,dx,dy)[1]
    
    if (montrer_perf) : t4 = time.time() ; tphi += t4-t3
    
    #CONDITIONS AUX LIMITES
      
    """
    #Avec les nouvelles conditions aux limites:
    conditions_limites(u,g_u,d_u,h_u,b_u)
    conditions_limites(v,g_v,d_v,h_v,b_v)
     
    """
    ConditionLimites(u,v,U)                            #Sur les bords du domaine
    u = Apply_objects(u,ox,oy,l_objects)                                            #Sur l'obstacle, penalisation
    v = Apply_objects(v,ox,oy,l_objects)                                             #Sur l'obstacle, penalisation
    col = Rescol
    col = Apply_objects(col,ox,oy,l_objects)
    #AFFICHAGE DE LA FIGURE
    t_u0 = time.time()
    Update_objects(l_objects)
    tupdate += time.time()-t_u0
    if ( (save_mode == 'iterations') and (nitermax-niter-1)%modulo == 0 ) or ( save and ( (save_mode == 'time') and ( (t-t_ref) > modulo ) ) ):
        t_ref = t
        compt+=1
        #print(np.sum(c))
        
        #affichage
         
        if (affichage == 'col'):
            mat = np.copy(col[1:-1,1:-1])
            title = 'colorant'
        elif (affichage == 'u'):
            mat = np.copy(u[1:-1,1:-1])
            title = r'$u_x$'
        elif (affichage == 'v'):
            mat = np.copy(v[1:-1,1:-1])
            title = r'$u_y$'
        elif (affichage == 'abs'):
            mat = np.sqrt( u[1:-1,1:-1]**2 + v[1:-1,1:-1]**2 )
        elif (affichage == 'rot'):
            mat = rot(u,v,dx,dy)
            title = r'$\omega$ = curl(v)'
        elif (affichage == 'p'):
            mat = np.copy(phi[1:-1,1:-1])/dt
            title = 'Pressure p'
        
        
        if (autoscale == 'full'): # si jamais il faut autoscale, le faire
            AFF.autoscale()
        elif (autoscale == 'adaptative'):
            if (compt == 1):
                vmin = np.min(mat)
                vmax = np.max(mat)
            else:
                vmin = min(vmin, np.min(mat))
                vmax = max(vmin, np.max(mat))
            if (affichage == 'rot'):
                vmax = np.max(np.abs([vmin, vmax]))
                vmin = -vmax
            AFF.set_clim(vmin = vmin, vmax = vmax)
        
        if (display_form): #Si on veut afficher les objets
            mat = Apply_objects(mat,xx,yy,l_objects,None) #Mets des None si le point est dans un objet
            
        AFF.set_data(mat)
        
        
        plt.title(title+'\nRe = '+str(Re)+', L =  '+str(L)+', U = '+str(U)+'\n t = '+str(round(t,2))+' , n = '+str(niter+1)+'/'+str(nitermax)+'\n(frame '+str(compt)+'/'+nFramesReal+')')
        plt.draw()
        if (save):
            if compt<10:
                    nb = '000'+str(compt)
            elif compt<100:
                    nb = '00'+str(compt)
            elif compt<1000:
                    nb = '0'+str(compt)
            elif compt<10000:
                    nb = ''+str(compt)
            name = 'BVK_'+nb+'.png'
            plt.gcf().set_size_inches(figsize)
            plt.savefig( os.path.join(save_directory,name) ,dpi=max(100,realdpi))
            plt.gcf().set_size_inches(figsize)
        plt.pause(0.001)
        if 'qt' in plt.get_backend().lower():
            QtWidgets.QApplication.processEvents()
        if (montrer_perf) : t5 = time.time() ; taff += t5-t4
    if mode == 'iterations' :
        if niter >= nitermax :
            dontstop = False
    elif mode == 'time' :
        if t >= tfinal :
            dontstop = False
    elif mode == 'time_bounded' :
        if (niter >= nitermax or t >= tfinal): 
            dontstop = False
            
t_fin = time.time()
print("Fin des calculs.")

print('Final time : '+str(t))
print('Number of timesteps : '+str(niter))
print('FPS : ' + str(int(nFramesReal)//t))


if (montrer_perf):
    print('\n** PERFORMANCES **\n\nTemps total de calcul : '+disptime(t_fin-t_deb))
    print('\nDuree de l\'étape d\'advection  : '+disptime(tadv)+
          '\nDuree de l\'étape de diffusion : '+disptime(tdiff)+
          '\nDuree de l\'étape de pression  : '+disptime(tphi)+
          '\nDuree de l\'étape d\'affichage  : '+disptime(taff) +
          '\nDuree de l\'étape d\'update des objets  : ' + disptime(tupdate))
    
print("Fin du programme.")
