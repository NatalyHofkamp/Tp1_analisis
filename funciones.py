
import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, title, xlabel, ylabel, legend):
    plt.figure()
    plt.stem(x, y, label=legend, linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend([legend])
    plt.show()

def diente_de_sierra(A, f, muestras):
    señal = A * (muestras * f - np.floor((1/2) + f * muestras))
    plot(muestras, señal, 'Señal Diente de Sierra', 'Tiempo', 'Amplitud', 'Diente de Sierra')

def serie_diente_de_sierra(A, a0, T, muestras, cant_armonicos, umbral=1e-6):
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonico = (4*A/T**2)*(((T/((2*np.pi/T)*n))*(1-np.cos((2*np.pi/T)*n*t/2)))+\
                       (np.sin((2*np.pi/T)*n*(t/2))/((2*np.pi/T)*n)**2)) * np.sin((2*np.pi/T)* n * t)
            if abs(armonico) < umbral:
                armonico = 0  
            armonicos += armonico
            #depende por cuánto divido el a0,se grafican mejor o peor la parte de abajo
        serie.append((a0/2) + armonicos)
    plot(muestras, serie, 'Serie Diente de Sierra', 'Muestras', 'Serie', 'serie diente de sierra')

def tren_de_pulsos(A,w, tiempo_total, muestras):
    tiempo = np.linspace(0, tiempo_total, muestras)
    señal = np.sign(np.sin(w * tiempo))
    plot(tiempo, señal, 'Señal x(t)', 'Tiempo (s)', 'Amplitud', 'Tren de Pulsos')

def serie_tren_de_pulsos(A,T, muestras, cant_armonicos):
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonicos += ((4*A/(T*n*(2*np.pi/T)))*(1-np.cos(n*(2*np.pi/T)*T/2)) * np.sin(n *(2*np.pi/T) * t))
        serie.append(armonicos)
    plot(muestras, serie, 'Serie del Tren de Pulsos', 'Muestras', 'Serie', 'serie tren de pulsos')


def main():
    tiempo_total = 2.0
    A = 1.0      
    T = (2*np.pi)   
    tiempo_total = 6*np.pi 
    w= (2*np.pi)/T 
    muestras = np.linspace(0,tiempo_total,100)
    muestras_diente = np.linspace(0,20,100)
    diente_de_sierra(A,1/2,muestras_diente)
    serie_diente_de_sierra(A,A/2,2,muestras_diente,30)
    tren_de_pulsos(A,w,tiempo_total,100)
    serie_tren_de_pulsos(A,T,muestras,50)
   
if __name__ =='__main__':
    main()