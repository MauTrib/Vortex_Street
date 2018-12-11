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
