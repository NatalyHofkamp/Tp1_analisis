import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def calculate_ECM(signal, approx, auto_threshold=False):
    """
    Calcula el Error Cuadrático Medio (ECM) entre la señal original y la aproximación.

    Parámetros:
    signal (array): La señal original.
    approx (array): La señal aproximada.
    auto_threshold (bool): Indica si se debe calcular automáticamente el umbral de descuento.

    Retorna:
    float: El valor del ECM calculado.
    """
    N = 0
    e = 0
    if auto_threshold:
        threshold = np.max(np.abs(np.diff(signal))) * 0.1
    else:
        threshold = 0.1  # Puedes ajustar este valor manualmente si no deseas el umbral automático
    
    for i in range(len(signal) - 1):
        if np.abs(signal[i] - signal[i + 1]) < threshold:
            N += 1
            e += (signal[i] - approx[i])**2
    
    if N > 0:
        e *= 1/N
    else:
        e = 0.0
    
    return e



def calculate_ECM_excluding_discontinuities(signal, approx, discontinuity_indices):
    """
    Calcula el Error Cuadrático Medio (ECM) excluyendo los puntos de discontinuidad.

    Parámetros:
    signal (array): La señal original.
    approx (array): La señal aproximada.
    discontinuity_indices (array): Índices de los puntos de discontinuidad.

    Retorna:
    float: El valor del ECM calculado excluyendo los puntos de discontinuidad.
    """
    discontinuity_indices = np.array(discontinuity_indices)
    valid_indices = np.delete(np.arange(len(signal)), discontinuity_indices)
    return np.mean((signal[valid_indices] - approx[valid_indices])**2)


def approximate_signal(A, T, muestras, signal, serie, target_ECM, auto_threshold=False):
    """
    Aproxima una señal utilizando Series de Fourier hasta que el ECM sea menor o igual al valor de target_ECM.
    Retorna la cantidad de armónicos necesarios para alcanzar el target_ECM.

    Parámetros:
    A (float): Amplitud máxima de la serie de Fourier.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.
    signal (array): La señal original.
    serie (function): Función que calcula la aproximación de la señal.
    target_ECM (float): Valor de referencia para el ECM.
    auto_threshold (bool): Indica si se debe calcular automáticamente el umbral de descuento.

    Retorna:
    int: La cantidad de armónicos necesarios para alcanzar el target_ECM.
    """
    current_ECM = np.inf 
    cant_armonicos = 0
    discontinuity_indices = np.where(np.diff(signal) != 0)[0]  
    total_ECM_values = []  
    valid_ECM_values = []  
    while current_ECM > target_ECM and cant_armonicos < 500:
        cant_armonicos += 1
        approx_signal = np.array(serie(A, T, muestras, cant_armonicos))
        current_ECM = calculate_ECM(signal, approx_signal, auto_threshold)
        valid_ECM = calculate_ECM_excluding_discontinuities(signal, approx_signal, discontinuity_indices)
        total_ECM_values.append(current_ECM)
        valid_ECM_values.append(valid_ECM)

    print(f'Armónicos: {cant_armonicos}, ECM Total: {current_ECM}, ECM Excluyendo Discontinuidades: {valid_ECM}')

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, cant_armonicos + 1), total_ECM_values, label='ECM Total')
    plt.plot(range(1, cant_armonicos + 1), valid_ECM_values, label='ECM Excluyendo Discontinuidades')
    plt.title('Error Cuadrático Medio (ECM) vs. Cantidad de Armónicos')
    plt.xlabel('Cantidad de Armónicos')
    plt.ylabel('ECM')
    plt.yscale('log')  
    plt.legend()
    plt.show()

    return cant_armonicos




def plot(x, y, title, xlabel, ylabel, legend):
    """
    Genera un gráfico de tipo stem (gráfico de tallo y hojas).

    Parámetros:
    x (array): Valores en el eje x.
    y (array): Valores en el eje y.
    title (str): Título del gráfico.
    xlabel (str): Etiqueta del eje x.
    ylabel (str): Etiqueta del eje y.
    legend (str): Etiqueta para la leyenda.

    """
    plt.figure(figsize=(8, 4))
    plt.stem(x, y, label=legend, linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def tren_de_pulsos(A, T, muestras):
    """
    Genera una señal de tren de pulsos.

    Parámetros:
    A (float): Amplitud de la señal.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.

    Retorna:
    array: La señal generada de tren de pulsos.
    """
    w = (2 * np.pi) / T
    signal = np.sign(np.sin(w * muestras))
    return signal

def serie_tren_de_pulsos(A, T, muestras, cant_armonicos):
    """
    Calcula la serie de Fourier para una señal de tren de pulsos.

    Parámetros:
    A (float): Amplitud de la señal.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.
    cant_armonicos (int): Cantidad de armónicos a calcular en la serie.

    Retorna:
    array: La serie de Fourier calculada para la señal de tren de pulsos.
    """
    serie = []
    w = (2 * np.pi / T)
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            armonicos += ((4 * A / (T * n * w)) * (1 - np.cos(n * w * T / 2)) * np.sin(n * w * t))
        serie.append(armonicos)
    return serie

def diente_de_sierra(A, T, muestras):
    """
    Genera una señal de diente de sierra.

    Parámetros:
    A (float): Amplitud de la señal.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.

    Retorna:
    array: La señal generada de diente de sierra.
    """
    f = 1 / T
    signal = A * (muestras * f - np.floor((1 / 2) + f * muestras))
    return signal

def serie_diente_de_sierra(A, T, muestras, cant_armonicos):
    """
    Calcula la serie de Fourier para una señal de diente de sierra.

    Parámetros:
    A (float): Amplitud de la señal.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.
    cant_armonicos (int): Cantidad de armónicos a calcular en la serie.

    Retorna:
    array: La serie de Fourier calculada para la señal de diente de sierra.
    """
    serie = []
    for t in muestras:
        armonicos = 0
        for n in range(1, cant_armonicos + 1):
            w = (2 * np.pi / T)
            a = (4 * A / (T ** 2))
            alpha = w * n * T / 2
            b = (T / (2 * w * n))
            armonicos += (a * ((b * (-np.cos(alpha))) + (np.sin(alpha) / ((w * n) ** 2))) * np.sin(w * n * t))
        serie.append(armonicos)
    return serie

def triangular(A, T, muestras):
    """
    Genera una señal triangular.

    Parámetros:
    A (float): Amplitud de la señal.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.

    Retorna:
    array: La señal generada de forma triangular.
    """
    periodo = T / 2
    y = np.abs((muestras % T) - periodo)
    return (4 * A * y) / T - A

def serie_triangular(A, T, muestras, cant_armonicos):
    """
    Calcula la serie de Fourier para una señal triangular.

    Parámetros:
    A (float): Amplitud de la señal.
    T (float): Período de la señal.
    muestras (array): Muestras de tiempo.
    cant_armonicos (int): Cantidad de armónicos a calcular en la serie.

    Retorna:
    array: La serie de Fourier calculada para la señal triangular.
    """
    serie = np.zeros_like(muestras)
    w0 = 2 * np.pi / T
    a_0 = 0
    serie += a_0 / 2

    for n in range(1, cant_armonicos * 2, 2):
        a_n = (8 * A) / (n ** 2 * np.pi ** 2)
        serie += a_n * np.cos(n * w0 * muestras)
    return serie


def fenomeno_gibbs(signal_, series, T):
    """
    Calcula el fenómeno de Gibbs en una señal en puntos de discontinuidad.

    Parámetros:
    signal_ (array): La señal original.
    series (list): Lista de tuplas con series de Fourier y parámetros.
    T (float): Período de la señal.

    Imprime:
    Información sobre el fenómeno de Gibbs en puntos de discontinuidad.
    """
    signal_ = np.array(signal_)
    maxima_valor = np.max(signal_)
    rising_edges = np.where(np.diff(signal_ == maxima_valor))[0]
    for serie, cant_armonicos, linestyle in series:
        error = np.abs(serie - signal_)
        amplitudes_gibbs = [error[idx] for idx in rising_edges]
        for i, punto in enumerate(rising_edges):
            print(f'Fenómeno de Gibbs en punto de discontinuidad {i+1}:')
            print(f'Cantidad de armónicos: {cant_armonicos}')
            print(f'Amplitud de Gibbs: {amplitudes_gibbs[i]}')

def graphs(muestras, signal_, series, title):
    """
    Genera gráficos de una señal y sus aproximaciones.

    Parámetros:
    muestras (array): Muestras de tiempo.
    signal_ (array): La señal original.
    series (list): Lista de tuplas con series de Fourier y parámetros.
    title (str): Título del gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(muestras, signal_, label='Señal')
    for serie, cant_armonicos, linestyle in series:
        plt.plot(muestras, serie, label=f'{cant_armonicos} armónicos', linestyle=linestyle)
    plt.title(title, fontsize=20)
    plt.xlabel('Tiempo (s)', fontsize=18)
    plt.ylabel('Señal x(t)', fontsize=18)
    plt.legend(fontsize=14)
    plt.show()


def create_signal_serie(A, T, periodo, cant_muestras, signal, serie):
    """
    Crea una señal, calcula su serie de Fourier y muestra gráficos.

    Parámetros:
    A (float): Amplitud máxima de la serie de Fourier.
    T (float): Período de la señal.
    periodo (float): Período de muestreo.
    cant_muestras (int): Cantidad de muestras.
    signal (function): Función que genera la señal original.
    serie (function): Función que calcula la aproximación de la señal.

    Retorna:
    tuple: Una tupla con la señal, las muestras y las series de Fourier.
    """
    muestras = np.linspace(0, periodo, cant_muestras)
    signal_ = signal(A, T, muestras)
    series = []
    for cant_armonicos, linestyle in [(10, 'solid'), (30, '-.'), (50, '--')]:
        serie_ = serie(A, T, muestras, cant_armonicos)
        series.append((serie_, cant_armonicos, linestyle))
    fenomeno_gibbs(signal_, series, T)
    return (signal_, muestras, series)


def main():
    A = 1.0      
    T = (2*np.pi)   
    tren_pulsos, muestras_tren,series_tren= create_signal_serie(A,T,4*np.pi ,100,tren_de_pulsos,serie_tren_de_pulsos)
    diente_sierra, muestras_diente,series_diente = create_signal_serie(A,2,6,100,diente_de_sierra,serie_diente_de_sierra)
    señal_triangular, muestras_triangular, series_triangular = create_signal_serie(A, T, 4*np.pi, 100, triangular, serie_triangular)
    plot(muestras_triangular,señal_triangular, 'selk','sjf','trianualr','wef')
    print('tren de pulsos - error esperado : 0.5')
    approximate_signal(A,T,muestras_tren,tren_pulsos,serie_tren_de_pulsos,0.5)
    print('diente de sierra - error esperado : 0.5')
    approximate_signal(A,T,muestras_diente,diente_sierra,serie_diente_de_sierra,0.5)
    print('tren de pulsos - error esperado : 0.1')
    approximate_signal(A,T,muestras_tren,tren_pulsos,serie_tren_de_pulsos,0.08)
    print('diente de sierra - error esperado : 0.1')
    approximate_signal(A,T,muestras_diente,diente_sierra,serie_diente_de_sierra,0.1)
    print('tren de pulsos - error esperado : 0.01')
    approximate_signal(A,T,muestras_tren,tren_pulsos,serie_tren_de_pulsos,0.01)
    print('diente de sierra - error esperado : 0.01')
    approximate_signal(A,T,muestras_diente,diente_sierra,serie_diente_de_sierra,0.01)

if __name__ =='__main__':
    main()