# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:21:45 2018

@author: MauTrib
"""

#### PARAMETRES CHOISIS PAR L'UTILISATEURICE (à modifier pour changer le résultat) 
#----------------------------------------------------------------------------------------------------------------------------------------------------
 ### PARAMETRES PHYSIQUES 
   # Taille de l'objet (taille caractéristique) (a priori,  = 1)
L = 1.  
   # Vitesse caractéristique (a priori, = 1)
U = 1.
   # Nombre de Reynolds (contrôle tout : en log, entre -8 et 10 )
Re = 1000;
   # Sur quel temps on simule
tfinal = 50.
 
 ### PARAMETRES NUMERIQUES
  ## Paramètres fondamentaux 
'''
mode : choisis le mode d'intégration. Trois possibilités : 
     * 'iterations'   -> Le nombre d'étapes est fixé. Le temps final n'est pas connu.
     * 'time'         -> tfinal est fixé. le nombre d'étapes n'est pas connu (ATTENTION)
     * 'time_bounded' -> comme 'time' mais avec une limite pour le nombre d'étapes
'''
mode = 'iterations'
   # Combien de timesteps au maximum on s'autorise
nitermax = 2000
   # Nombre de pixels du domaine :
NX = 256 ; NY = 128
   # Largeur du canal. On doit avoir LY > 2*L
LY = 10*L
   # Longueur du canal. Les pixels sont carrés ssi LX = LY*nx/ny
LX = LY * (NX-2)/(NY-2)
   # Emplacement de l'obstacle
x_c = (1/4)*LX
y_c = LY/2

  ## Paramètres d'optimisation
   # Marge sur le temps d'advection pour éviter la divergence : dt = precautionADV*dt_calculé
precautionADV = 0.95				
   # Marge sur le temps de diffusion pour éviter la divergence : dt = precautionDIFF*dt_calculé							
precautionDIFF = 0.9 												
   # RK2 : utilise une méthode RK2 pour le Laplacien (sécurité à bas Re)
RK2 = True
   # veriffactor : si 1, ne fait rien ; si > 1 : fait la même simulation avec des plus petits pas de temps
veriffactor = 1

 ### PARAMETRES D'AFFICHAGE 
   # combien on affiche de frames (pertinent si save == False)
nFrames = 20
   # le nombre de dpi. 100 est la taille normale
taille_figure = 100
   # on affiche la boule d'une couleur différente ?
display_form = True
   # la couleur (en format RGB avec des valeurs entre 0 et 1 OU entre 0 et 255)
color_form = [0.2, 0.1, 0.1]
   # est ce qu'on sauvegarde
save = False
'''
save_mode : choisis le mode d'enregistrement. Deux possibilités : 
     * 'iterations'   -> On enregistre toutes les Delta_n etapes
     * 'time'         -> On enregistre tous les Delta_t temps
'''
save_mode = 'iterations'
   # l'endroit où on sauvegarde
save_directory = 'Figures'
   # le detail low, mid, top, max
qualite_video = 'max'
'''
affichage : ce qu'on affiche. Plusieurs possibilités :
    * 'col'  : affichage du colorant 
    * 'u'    : affichage de la vitesse horizontale 
    * 'v'    : affichage de la vitesse verticale 
    * 'abs'  : affichage de la norme de la vitesse
    * 'rot'  : affichage du rotationnel
    * 'p'    : affichae de la pression
'''
affichage = 'u'
# flux de colorant : 'all' ou un nombre
nombreDeStreams = 9			
# le scaling
'''
affichage : l'actualisation de l'échelle de couleurs. Plusieurs possibilités :
    * 'no' : utilise les valeurs vmin et vmax spécifiée
    * 'full' : utilise une échelle adaptée à chaque affichage
    * 'adaptative' : utilise l'échelle adaptée la plus large
'''
autoscale = 'full' # full
vmin, vmax = -0.25, 1.25			#quand autoscale = 'no', determine les bornes de la colorbar
  ## Performances
   # montrer_perf : affiche les performances (le temps de chaque étape)
montrer_perf = True

#----------------------------------------------------------------------------------------------------------------------------------------------------

#### VARIABLES DU PROGRAMME (à ne pas modifier a priori)
 ### VARIABLES PHYSIQUES 
   # Coefficient de diffusion (1/Re)
D = 1./Re
   # Origine des temps
t = 0.

 
 ### VARIABLES NUMERIQUES
  ## Paramètres fondamentaux 
   # Le compteur de timesteps
niter = 0
   # Nombre de points du domaine réel
nx = NX-2  ; ny = NY-2
   # différentielles spatiales
dx = LX/(nx-1) ; dy = LY/(ny-1)
   # aire élémentaire
atot = dx*dy
   # le plus petit entre dx et dy (pour le calcul de dt)
dmin = min(dx,dy)
   # utile pour le laplacien
dx_2 = 1./dx**2 ; dy_2 = 1./dy**2  
   # différentielle temporelle (à choisir le plus grand possible, sera rapetissé de toutes façon à la fin)
dt = 1
   # application de veriffactor
precautionADV /= veriffactor
precautionDIFF /= veriffactor ; 
nitermax *= veriffactor
   # Le dt maximal en prenant en compte l'étape de diffusion
dt_exp = CFL_diffusion()
   # Combien de timesteps au maximum on s'autorise
if (mode == 'time_bounded'):
    nitermax = int( 2.* tfinal / min( dt, dt_exp, dmin/U * precautionADV ) )

 ### PARAMETRES D'AFFICHAGE 
  ## Affichage de la figure
fig = plt.figure(dpi=taille_figure)
  ## Frequence d'affichage
   # nombre de frames
if (save):
    if (save_mode == 'iterations'):
        if (qualite_video == 'low'):
            nFrames = int(50*nitermax/1000)
        elif (qualite_video == 'mid'):
            nFrames = int(100*nitermax/1000)
        elif (qualite_video == 'top'):
            nFrames = int(150*nitermax/1000)
        elif (qualite_video == 'max'):
            nFrames = int(200*nitermax/1000)
        else :
            print('qualite_video n\'est pas defini correctement')
    if (save_mode == 'time'):
        if (qualite_video == 'low'):
            nFrames = int(tfinal*1)
        elif (qualite_video == 'mid'):
            nFrames = int(tfinal*2)
        elif (qualite_video == 'top'):
            nFrames = int(tfinal*5)
        elif (qualite_video == 'max'):
            nFrames = int(tfinal*10)
        else :
            print('qualite_video n\'est pas defini correctement')
    files = os.listdir(save_directory)
    for name in files:
        os.remove(os.path.join(save_directory,name)) # supprime tout ce qu'il y avait avant
   # la figure est actualisée tous les 'modulo' timesteps
modulo = nitermax//nFrames 
if (mode == 'time'):
    modulo = 500
if (save and save_mode == 'time'):
    modulo = tfinal/nFrames
t_ref = 0 # le temps du dernier affichage
   # Le nombre final de frames
nFramesReal = str(nitermax//modulo + np.sign(nitermax%modulo))
if (mode == 'time'):
    nFramesReal = '???'
  ## jolitude

# Position des flux de colorants, régulièrement espacés entre le haut et le bas du canal
if (nombreDeStreams == 'all'):
    pointsAjoutColorant = [k for k in range(1, NY-1)]
else:
    pointsAjoutColorant  = [NY//2+NY//nombreDeStreams*i for i in range(int(-nombreDeStreams/2),int(nombreDeStreams/2)+1)]  	
   # la colormap
if (affichage == 'col'):
    #☻ jet , 'gist_heat', 'binary_r'
    cdict2 = {'red':   ((0.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0)),
    
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 0.0, 0.2),
                       (1.0, 0.0, 0.0))
            }
    blue_red2 = colors.LinearSegmentedColormap('BlueRed2', cdict2)
    cmap = 'afmhot'
    if (nombreDeStreams == 'all'):
        cmap = 'afmhot_r'
elif (affichage == 'u'):
    cmap = 'inferno'
elif (affichage == 'v'):
    cmap = 'inferno'
elif (affichage == 'abs'):
    cmap = 'plasma'
elif (affichage == 'rot'):
    col = 0.05
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, col, col),
                       (0.65, 0.9, 0.9),
                       (1.0, 1.0, 0.0)),
    
             'green': ((0.0, 0.0, 0.0),
                       (0.5, col, col),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 0.0, 1.0),
                       (0.35, 0.9, 0.9),
                       (0.5, col, col),
                       (1.0, 0.0, 0.0))
            }
    
    blue_red = colors.LinearSegmentedColormap('BlueRed', cdict1)
    cmap = blue_red
    # cmap = 'seismic'
elif (affichage == 'p'):
    cmap = 'inferno'
else:
    print("Parametre d'affichage non reconnu !")
  ## Performances
   # les temps des différentes étapes
tadv, tdiff, tphi, taff = 0, 0, 0, 0


###CONDITIONS INITIALES (avec les points fantômes)

#VITESSES
u = np.zeros((NY,NX)) 					# On part d'un fluide au repos
# CI du zbeul 
u[1:-1,1:-1] +=  np.random.rand(ny,nx)*U*0.001 						# impuretés pour briser la symétrie et observer les tourbillons

v = np.zeros((NY,NX))

ConditionLimites(u,v)
#COLORANT
col = np.zeros((NY,NX))

Resu = np.zeros((NY,NX))
Resv = np.zeros((NY,NX))
Rescol = np.zeros((NY,NX))


#PRESSION PHI
phi = np.zeros((NY,NX)) 				# phi = P*dt

 
###CONSTRUCTION DES MATRICES ET DECOMPOSITION LU POUR L'ETAPE DE PROJECTION
LAPoisson = BuildLaPoisson() 
LUPoisson = ILUdecomposition(LAPoisson)

###MAILLAGE POUR L'AFFICHAGE SANS LES POINTS FANTOMES
x,x_step = np.linspace(0,LX,nx,retstep=True) 
y,y_step = np.linspace(0,LY,ny,retstep=True)
[xx,yy] = np.meshgrid(x,y)

##CREATION DE LA MESHGRID AVEC POINTS FANTOMES
x_gp = np.arange(-x_step,LX+1.5*x_step,x_step) #+ 1.5*x_step pour être sûr de prendre la dernière valeur
y_gp = np.arange(-y_step,LY+1.5*y_step,y_step)

ox,oy = np.meshgrid(x_gp,y_gp)

AFF = plt.imshow(np.zeros([ny,nx]), cmap, extent=[-x_c,LX-x_c,-y_c,LY-y_c])					#fenêtre graphique actualisée
if (autoscale == 'no'):
    AFF.set_clim(vmin=vmin, vmax = vmax)

if (display_form):
    z = np.zeros([NY, NX, 4])
    z[:,:,0] = color_form[0]
    z[:,:,1] = color_form[1]
    z[:,:,2] = color_form[2]
    z[(ox-x_c)**2+(oy-y_c)**2 < L**2,-1] = 1
    
    plt.imshow(z[1:-1,1:-1,:], extent=[-x_c,LX-x_c,-y_c,LY-y_c])

plt.colorbar(AFF,fraction=0.0232, pad=0.04)

plt.show()

compt = 0    # Permet de savoir combien de frame ont été affichées



if mode == 'iterations' :
    print(str(nitermax)+' iterations will be done')
elif mode == 'time' :
    print('The program will iterate until time '+str(tfinal)+' is reached')
elif mode == 'time_bounded' :
    print('The program will iterate until time '+str(tfinal)+' is reached or '+str(nitermax)+' iterations are done')
else :
    print('mode n\'est pas defini correctement')

if (save == True and (save_mode == 'time')):
    print('An image will be shown every '+str(modulo)+' time')
else:
    print('An image will be shown every '+str(modulo)+' timestep')

dontstop = True
