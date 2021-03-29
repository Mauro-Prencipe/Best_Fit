# best fit di una serie di dati con varie funzioni
# Dati di input nel file best_fit.txt  

from IPython import get_ipython
get_ipython().magic('cls')
get_ipython().magic('reset -sf')

# import delle librerie di python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

class driver_class():
    
    def __init__(self):
        self.deg=None
        self.x=None
        self.y=None
        self.x_orig=None
        self.y_orig=None
        self.flag=False
     
    def set_degree(self,deg):
        
        """
        Fissa il grado del polinomio e della funzione myfunc_2
        
        Parameters
        ----------
            deg : grado del polinomio e della funzione a*x^deg
        """
        self.deg=deg
        
    def set_xlimit(self,xmin,xmax):
        """
        Definisce l'intervallo della variabile indipendente x
        
        Parameters
        ----------
            xmin : valore minimo dell'intervallo
            xmax : valore massimo dell'intervallo
        """
        self.xmin=xmin
        self.xmax=xmax
        select=(self.x_orig >= xmin) & (self.x_orig <= xmax)
        self.x=self.x_orig[select]
        self.y=self.y_orig[select]
        print("X data range set to [%5.2f, %5.2f]" % (xmin, xmax))
        
    def xlimit_reset(self):
        """
        Riporta l'intervallo di definizione della variabile indipendente
        x al valore originale.
        """
        self.x=np.copy(self.x_orig)
        self.y=np.copy(self.y_orig)
        
drv=driver_class()

print("Programma 'best_fit'\n\nUtilizzo:")
print("I dati (x, y) vengono letti dal file 'best_fit.txt'")
print("\nLa funzione 'fit' esegue il fit e accetta il parametro (opzionale) deg che")
print("specifica il grado del polinomio e del modello myfunc_2 (default: 2)")
print("\nIl metodo drv.set_xlimit consente di definire l'intervallo della variabile")
print("indipendente x su cui eseguire il fit")
print("\nIl metodo drv.xlimit_reset riporta l'intervallo di definizione della variabile")
print("al valore originale\n")
print("La funzione load_data carica i dati di input")
        
# definizione della funzione y(x)=a*x^b
# myfunc accetta come input la serie di valori della
# variabile dipendente x, e i parametro "a"  e "b" da ottimizzare
def myfunc(x,a,b):
    return a*x**b

# definizione della funzione y(x)=a*x^deg
# myfunc accetta come input la serie di valori della
# variabile dipendente x, e il parametro "a" da ottimizzare
def myfunc_2(x,a):
    return a*x**drv.deg

def load_data(file='best_fit.txt'):
    """
    Carica i dati di input
    
    Parameters
    ----------
        file : nome del file di dati (default 'best_fit.txt')  
    """
    data=np.loadtxt(file)
    x=data[:,0]
    y=data[:,1]
    drv.x=x
    drv.y=y
    drv.x_orig=np.copy(drv.x)
    drv.y_orig=np.copy(drv.y)
    drv.flag=True

    print("Input file: %s\n" % file)
    dati=(drv.x,drv.y)
    df=pd.DataFrame(dati, index=['X','Y'])
    df=df.T
    df2=df.round(3)
    print(df2.to_string(index=False))
    
def fit(deg=2,file='best_fit.txt',reload=False, ret=False):
    """
    Esegue il fit. 
    
    Parameters
    ----------
        deg :    grado del polinomio e della funzione myfunc_2 (default 2)
        file :   nome del file i dati (default best_fit.txt)
        reload : se True, forza il ricaricamento del file di dati (default False)
        ret :    se True restituisce in uscita i valori dei parametri ottimizzati
                 (default False)
    """
    drv.set_degree(2)
    if deg != 2:
       drv.set_degree(deg)
       
    if (not drv.flag) or reload:
        try:
           load_data(file)
        except:
           print("File di input non presente")
           return
    
    opt, pcov = curve_fit(myfunc, drv.x, drv.y, p0=[1,1])
    err=np.sqrt(np.diag(pcov))

    a=opt[0]
    b=opt[1]

    print("\nFit del modello 'myfunc'")
    print("Parametro 'a' ottimizzato: %5.3f (%3.2f)" % (a, err[0]))
    print("Parametro 'b' ottimizzato: %5.3f (%3.2f)\n" % (b, err[1]))

    opt_2,err= curve_fit(myfunc_2, drv.x, drv.y)
    err=np.sqrt(err)
    a_2=opt_2[0]
    err_2=np.sqrt(err[0,0])
    print("Fit del modello 'myfunc_2'")
    print("Parametro 'a' ottimizzato: %5.3f (%3.2f)\n" % (a_2, err_2))
    
    opt_p, pcov_p=np.polyfit(drv.x,drv.y,drv.deg,cov=True)
    err_p=np.sqrt(np.diag(pcov_p))

    print("Fit polinomiale: parametri ottimizzati:")
    idx=0
    for ip in opt_p:
        id1=idx+1
        print("Parametro %3i: %7.3f  (%6.2f)" % (id1, ip, err_p[idx]))
        idx=idx+1

    x_list=np.linspace(min(drv.x_orig),max(drv.x_orig),30)
    y_list=myfunc(x_list,a,b)
    y_2_list=myfunc_2(x_list,a_2)
    y_p_list=np.polyval(opt_p,x_list)

    yc=myfunc(drv.x,a,b)
    delta=drv.y-yc

    yc_2=myfunc_2(drv.x,a_2)
    delta_2=drv.y-yc_2
    
    yc_p=np.polyval(opt_p,drv.x)
    delta_p=drv.y-yc_p

    print("-------------------------------------")

    print("\nModello 'myfunc'")
    serie=(drv.x,drv.y,yc,delta)
    df=pd.DataFrame(serie, index=['x','y','y calc','delta'])
    df=df.T
    df2=df.round(3)
    print(df2.to_string(index=False))
    sdt=np.sqrt(np.sum((drv.y-yc)**2)/(drv.x.size-1))
    print("\nDeviazione standard: %5.1f" % sdt)
    print("------")
    
    print("\nModello 'myfunc_2'")
    serie=(drv.x,drv.y,yc_2,delta_2)
    df=pd.DataFrame(serie, index=['x','y','y calc','delta'])
    df=df.T
    df2=df.round(3)
    print(df2.to_string(index=False))
    sdt=np.sqrt(np.sum((drv.y-yc_2)**2)/(drv.x.size-1))
    print("\nDeviazione standard: %5.1f" % sdt)
    print("------")

    ds=str(drv.deg)
    print("\nModello polinomiale di grado %s" % ds)
    serie_p=(drv.x,drv.y,yc_p,delta_p)
    df_p=pd.DataFrame(serie_p, index=['x','y','y calc','delta'])
    df_p=df_p.T
    df2_p=df_p.round(3)
    print(df2_p.to_string(index=False))
    sdt_p=np.sqrt(np.sum((drv.y-yc_p)**2)/(drv.x.size-1))
    print("\nDeviazione standard: %5.1f" % sdt_p)

# --- Sezione di plot ---
    lab_p="Poly: grado "+ds
    plt.figure()
    plt.plot(drv.x_orig,drv.y_orig,"k*")              # dati "reali"
    plt.plot(x_list,y_list,"b-",label="myfunc")       # dati calcolati da myfunc
    plt.plot(x_list,y_2_list,"r--",label="myfunc_2")  # dati calcolati myfunc_2
    plt.plot(x_list,y_p_list,"b--", label=lab_p)      # dati modello polinomiale
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=False)
    plt.show()
    
    if ret:
       return a, b, opt_2[0], opt_p 