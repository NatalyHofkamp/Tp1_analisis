
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
    w= (2*np.pi/T)
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonicos += ((4*A/(T*n*w))*(1-np.cos(n*w*T/2)) * np.sin(n *w * t))
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

def graphs(muestras,signal_,series):
    plt.figure(figsize=(14, 8))
    plt.plot(muestras, signal_, label='Señal')
    for serie,cant_armonicos,linestyle in series:
         plt.plot(muestras, serie, label=f'{cant_armonicos} armónicos', linestyle=linestyle)
    plt.title('Señal y Serie',fontsize = 20)
    plt.xlabel('Tiempo',fontsize = 18)
    plt.ylabel('Amplitud',fontsize = 18)
    plt.legend(fontsize=14)

def fenomeno_gibbs (muestras,signal_,series,T):
    print(muestras)
    puntos_de_discontinuidad = [x for x in muestras if x%(T/2)== 0]
    print(puntos_de_discontinuidad)
    for serie, cant_armonicos, linestyle in series:
        error = np.abs(serie - signal_)  # Calcula la diferencia absoluta entre la serie y la señal original
        # Encuentra los índices de los puntos de discontinuidad en las muestras
        indices_discontinuidad = [np.abs(muestras - punto).argmin() for punto in puntos_de_discontinuidad]
        amplitudes_gibbs = [error[idx] for idx in indices_discontinuidad]
        
        # Imprime información sobre el fenómeno de Gibbs en los puntos de discontinuidad
        # for i, punto in enumerate(puntos_de_discontinuidad):
        #     print(f'Fenómeno de Gibbs en punto de discontinuidad {i+1}:')
        #     print(f'Cantidad de armónicos: {cant_armonicos}')
        #     print(f'Amplitud de Gibbs: {amplitudes_gibbs[i]}')
        #     print()

def create_signal_serie(A, T, periodo, cant_muestras, signal, serie):
    muestras = np.linspace(0, periodo, cant_muestras)
    signal_ = signal(A, T, muestras)
    series = []
    for cant_armonicos, linestyle in [(10, 'solid'), (30, '-.'), (50, '--')]:
        serie_ = serie(A, T, muestras, cant_armonicos)
        series.append((serie_,cant_armonicos,linestyle))
    graphs(muestras,signal_,series)
    fenomeno_gibbs(muestras,signal_,series,T)


def main():
    A = 1.0      
    T = (2*np.pi)   
    create_signal_serie(A,2,6,100,diente_de_sierra,serie_diente_de_sierra)
    create_signal_serie(A,T,4*np.pi ,100,tren_de_pulsos,serie_tren_de_pulsos)
 
if __name__ =='__main__':
    main()