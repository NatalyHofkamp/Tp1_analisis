
import numpy as np
import matplotlib.pyplot as plt


def calculate_ECM(signal, approx):
    """
    Calcula el Error Cuadrático Medio (ECM) entre la señal original y la aproximación.
    """
    return np.mean((signal - approx)**2)


def calculate_ECM_excluding_discontinuities(signal, approx, discontinuity_indices):
    """
    Calcula el Error Cuadrático Medio (ECM) excluyendo los puntos de discontinuidad.
    """
    discontinuity_indices = np.array(discontinuity_indices)
    valid_indices = np.delete(np.arange(len(signal)), discontinuity_indices)
    return np.mean((signal[valid_indices] - approx[valid_indices])**2)


def approximate_signal(A, T, muestras, signal, serie, target_ECM):
    """
    Aproxima una señal utilizando Series de Fourier hasta que el ECM sea menor o igual al valor de target_ECM.
    Retorna la cantidad de armónicos necesarios para alcanzar el target_ECM y grafica la aproximación conseguida.

    Parameters:
    - A: Amplitud de la señal.
    - T: Periodo de la señal.
    - muestras: Puntos de tiempo donde se evaluará la señal.
    - signal: Función que genera la señal original.
    - serie: Función que calcula la serie de Fourier.
    - target_ECM: Valor de ECM deseado como criterio de paro.

    Returns:
    - cant_armonicos: Cantidad de armónicos utilizados para alcanzar el target_ECM.
    """
    current_ECM = np.inf  # Inicializar el ECM con infinito.
    cant_armonicos = 0
    discontinuity_indices = np.where(np.diff(signal) != 0)[0]  # Índices de discontinuidades

    while current_ECM > target_ECM:
        cant_armonicos += 1
        approx_signal = np.array(serie(A, T, muestras, cant_armonicos))
        current_ECM = calculate_ECM_excluding_discontinuities(signal, approx_signal, discontinuity_indices)
        print(f'Iteración {cant_armonicos}: ECM = {current_ECM}')

    # Graficar la señal original y la aproximación conseguida.
    plt.figure(figsize=(12, 6))
    plt.plot(muestras, signal, label='Señal Original', linestyle='-', color='b')
    plt.plot(muestras, approx_signal, label=f'Aproximación con {cant_armonicos} armónicos', linestyle='--', color='r')
    plt.title('Aproximación de Señal con Serie de Fourier')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()

    return cant_armonicos



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


def tren_de_pulsos(A, w, tiempo_total, muestras):
    tiempo = np.linspace(0, tiempo_total, muestras)
    señal = np.sign(np.sin(w * tiempo))
    plot(tiempo, señal, 'Señal x(t)', 'Tiempo (s)', 'Amplitud', 'Tren de Pulsos')
    return tiempo, señal

def tren_de_pulsos(A,T,muestras):
    w = (2*np.pi)/T 
    tiempo = np.linspace(0, T, muestras)
    signal = np.sign(np.sin(w * muestras))
    # plot(muestras, signal, 'Señal x(t)', 'Tiempo (s)', 'Amplitud', 'Tren de Pulsos')
    return tiempo, signal



def serie_tren_de_pulsos(A,T, muestras, cant_armonicos):
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonicos += ((4*A/(T*n*(2*np.pi/T)))*(1-np.cos(n*(2*np.pi/T)*T/2)) * np.sin(n *(2*np.pi/T) * t))
        serie.append(armonicos)
    plot(muestras, serie, 'Serie del Tren de Pulsos', 'Muestras', 'Serie', 'serie tren de pulsos')



def calcular_ecm(señal_original, señal_aproximada):
    return np.mean((señal_original - señal_aproximada) ** 2)

 
def calcular_ecm_con_armónicos(A, T, muestras, cant_armonicos):
    w= (2*np.pi)/T 
    tiempo, señal_original = tren_de_pulsos(A, w, T, muestras)
    ecm_resultados = []

    for n in cant_armonicos:
        serie = []
        for t in tiempo:
            armonicos = 0
            for k in range(1, n + 1):
                armonicos += (4 / (k * np.pi)) * np.sin(2 * np.pi * k * t / T)
            serie.append(armonicos)
        ecm = calcular_ecm(señal_original, serie)
        ecm_resultados.append(ecm)
        print(f"ECM con {n} armónicos: {ecm}")

    return ecm_resultados

# Parámetros
A = 1
T = 2
muestras = 1000
cant_armonicos = [10, 30, 50]

# Calcular ECM
ecm_resultados = calcular_ecm_con_armónicos(A, T, muestras, cant_armonicos)

# Resultados de ECM
for i, n in enumerate(cant_armonicos):
    print(f"ECM con {n} armónicos: {ecm_resultados[i]}")

def calcular_ecm(señal_original, señal_aproximada):
    return np.mean((señal_original - señal_aproximada) ** 2)


def calcular_ecm_con_armónicos(A, T, muestras, cant_armonicos):
    tiempo, señal_original = tren_de_pulsos(A, T, muestras)
    ecm_resultados = []

    for n in cant_armonicos:
        serie = []
        for t in tiempo:
            armonicos = 0
            for k in range(1, n + 1):
                armonicos += (4 / (k * np.pi)) * np.sin(2 * np.pi * k * t / T)
            serie.append(armonicos)
        ecm = calcular_ecm(señal_original, serie)
        ecm_resultados.append(ecm)
        print(f"ECM con {n} armónicos: {ecm}")

    return ecm_resultados

# Parámetros
A = 1
T = 2
muestras = 1000
cant_armonicos = [10, 30, 50]

# Calcular ECM
ecm_resultados = calcular_ecm_con_armónicos(A, T, muestras, cant_armonicos)

# Resultados de ECM
for i, n in enumerate(cant_armonicos):
    print(f"ECM con {n} armónicos: {ecm_resultados[i]}")



# def main():
#     tiempo_total = 2.0
#     A = 1.0      
#     T = (2*np.pi)   
#     tiempo_total = 6*np.pi 
#     w= (2*np.pi)/T 
#     muestras = np.linspace(0,tiempo_total,100)
#     muestras_diente = np.linspace(0,20,100)
#     diente_de_sierra(A,1/2,muestras_diente)
#     serie_diente_de_sierra(A,A/2,2,muestras_diente,30)
#     tren_de_pulsos(A,w,tiempo_total,100)
#     serie_tren_de_pulsos(A,T,muestras,50)
#     serie_fourier_cuadrada(T, muestras, num_armonicos)
   
# if __name__ =='__main__':
#     main()

def main():
    A = 1.0      
    T = (2*np.pi)   
    tren_pulsos, muestras_tren,series_tren= create_signal_serie(A,T,4*np.pi ,100,tren_de_pulsos,serie_tren_de_pulsos)
    diente_sierra, muestras_diente,series_diente = create_signal_serie(A,2,6,100,diente_de_sierra,serie_diente_de_sierra)
    print('tren de pulsos - error esperado : 0.5')
    cant_armónicos_tren_0_5 = approximate_signal(A,T,muestras_tren,tren_pulsos,serie_tren_de_pulsos,0.5)
    print('diente de sierra - error esperado : 0.5')
    cant_armónicos_diente_0_5 = approximate_signal(A,T,muestras_diente,diente_sierra,serie_diente_de_sierra,0.5)
    print('tren de pulsos - error esperado : 0.08')
    cant_armónicos_tren_0_08 = approximate_signal(A,T,muestras_tren,tren_pulsos,serie_tren_de_pulsos,0.08)
    print('diente de sierra - error esperado : 0.08')
    cant_armónicos_diente_0_08 = approximate_signal(A,T,muestras_diente,diente_sierra,serie_diente_de_sierra,0.08)
    print('tren de pulsos - error esperado : 0.03')
    cant_armónicos_tren_0_03 = approximate_signal(A,T,muestras_tren,tren_pulsos,serie_tren_de_pulsos,0.03)
    print('diente de sierra - error esperado : 0.03')
    cant_armónicos_diente_0_03 = approximate_signal(A,T,muestras_diente,diente_sierra,serie_diente_de_sierra,0.03)
    
    # Crear gráfico del ECM en función de la cantidad de armónicos
    cantidades_armónicos = [cant_armónicos_tren_0_5, cant_armónicos_diente_0_5,
                            cant_armónicos_tren_0_08, cant_armónicos_diente_0_08, cant_armónicos_tren_0_03, cant_armónicos_diente_0_03]

    errores_objetivo = [0.5, 0.5, 0.08, 0.08, 0.03, 0.03]

    plt.figure(figsize=(10, 6))
    plt.plot(cantidades_armónicos, errores_objetivo, marker='o', linestyle='-', color='b')
    plt.title('ECM Objetivo vs. Cantidad de Armónicos')
    plt.xlabel('Cantidad de Armónicos')
    plt.ylabel('ECM Objetivo')
    plt.grid(True)
    plt.show()

if __name__ =='__main__':
    main()

