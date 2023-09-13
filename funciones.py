import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,title,xlabel,ylabel,legend):
    plt.figure()
    plt.stem(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.legend(legend)
    plt.show()

# Generar una señal diente de sierra
def generar_diente_de_sierra(pendiente, periodo, tiempo_total):
    tiempo = np.linspace(0, tiempo_total, int(100))
    señal = pendiente * (tiempo % periodo)
    plot(tiempo,señal,'Señal Diente de Sierra','tiempo','amplitud','diente de sierra')


def tren_de_pulsos(A,w,tiempo_total,muestras):
    # Generar la señal
    tiempo = np.linspace(0, tiempo_total,muestras)  # Crear un vector de tiempo
    señal = A * np.sign(np.sin(w* tiempo))  # Calcular la señal
    plot(tiempo,señal,'Señal x(t)','Tiempo (s)','Amplitud','tren de pulsos')
   
def serie_tren_de_pulsos(a0,bn,w,muestras,cant_armonicos):
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonicos+=((bn / n) * np.sin(n * w * t))
        serie.append((a0 / 2) + armonicos)
    plot(muestras,serie,'serie del tren de pulsos',"muestras",'serie','jkfwoj')


def main():
    tiempo_total = 2.0
    A = 1.0       # Amplitud
    T = (2*np.pi)    # Frecuencia en Hz
    tiempo_total = 6*np.pi # Duración de la señal en segundos
    w= (2*np.pi)/T
    muestras =  int(100)
    pendiente_diente_de_sierra = 1.0
    periodo_diente_de_sierra = 1.0
    generar_diente_de_sierra(pendiente_diente_de_sierra, periodo_diente_de_sierra, tiempo_total)
    tren_de_pulsos(A,w,tiempo_total,muestras)
    muestras = np.linspace(0,2*np.pi,100)
    serie_tren_de_pulsos(0,(A*2)/(np.pi*w),w,muestras,10)
   
if __name__ =='__main__':
    main()