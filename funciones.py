
import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, title, xlabel, ylabel, legend):
    plt.stem(x, y, label=legend, linefmt='b-', markerfmt='bo', basefmt=' ')
    # plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.legend([legend])
    plt.show()

def tren_de_pulsos(A,T,muestras):
    w = (2*np.pi)/T 
    signal = np.sign(np.sin(w * muestras))
    # plot(muestras, signal, 'Señal x(t)', 'Tiempo (s)', 'Amplitud', 'Tren de Pulsos')
    return signal


def serie_tren_de_pulsos(A,T, muestras, cant_armonicos):
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonicos += ((4*A/(T*n*(2*np.pi/T)))*(1-np.cos(n*(2*np.pi/T)*T/2)) * np.sin(n *(2*np.pi/T) * t))
        serie.append(armonicos)
    # plot(muestras, serie, 'Serie del Tren de Pulsos', 'Muestras', 'Serie', 'serie tren de pulsos')
    return serie


def diente_de_sierra(A, T, muestras):
    f = 1/T
    signal = A * (muestras * f - np.floor((1/2) + f * muestras))
    # plot(muestras, signal, 'señal Diente de Sierra', 'Tiempo', 'Amplitud', 'Diente de Sierra')
    return signal


def serie_diente_de_sierra(A,T, muestras, cant_armonicos):
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos+1):
            w = (2*np.pi/T)
            a = (4*A/(T**2))
            alpha = w*n*T/2
            b = (T/(2*w*n))
            armonicos += (a*((b*(-np.cos(alpha)))+(np.sin(alpha)/((w*n)**2))) * np.sin(w*n*t))
        serie.append(armonicos)
    # plot(muestras, serie, 'Serie Diente de Sierra', 'Muestras', 'Serie', 'serie diente de sierra')
    return serie




def create_signal_serie(A, T, periodo, cant_muestras, signal, serie):
    plt.figure(figsize=(14, 8))  # Tamaño de figura ajustado para acomodar el cartel con 4 valores
    muestras = np.linspace(0, periodo, cant_muestras)
    signal_ = signal(A, T, muestras)
    plt.plot(muestras, signal_, label='Señal')
    for cant_armonicos, linestyle in [(10, 'solid'), (30, '-.'), (50, '--')]:
        serie_ = serie(A, T, muestras, cant_armonicos)
        plt.plot(muestras, serie_, label=f'{cant_armonicos} armónicos', linestyle=linestyle)
    plt.title('Señal y Serie')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.legend(fontsize=14)



def main():
    A = 1.0      
    T = (2*np.pi)   
    create_signal_serie(A,2,6,100,diente_de_sierra,serie_diente_de_sierra)
    create_signal_serie(A,T,4*np.pi ,100,tren_de_pulsos,serie_tren_de_pulsos)
 
if __name__ =='__main__':
    main()