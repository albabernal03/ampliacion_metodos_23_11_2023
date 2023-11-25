import matplotlib.pyplot as plt
import numpy as np


def rugen_kutta(f, x, y, h, n):
    '''
    Función que implementa el método de Taylor para resolver una EDO
    '''
    u = []
    v = []
    for i in range(n):
        y = y + h * f(x + (h/2), y + (h/2)*f(x, y)) 
        x = x + h
        u.append(x)
        v.append(y)
    return u, v


def f(x, y):
    '''
    Aquí se define la EDO
    '''
    return -2*x*y + ((2*x)/np.exp(x**2))

def f_exacta(x):
    '''
    Función exacta 1
    '''
    return x**2/np.exp(x**2)


def error(v, v_aprox):
    '''
    Devuelve el error absoluto
    '''
    return abs(v - v_aprox)


# DATOS
# -----
# Rango de x
x_inicial = 0
x_final = 1.5
# Datos iniciales (me da n 3 datos iniciales donde x siempre es igual, pero y varía)
x = 0
y0 = 0

# Número de subintervalos que nos permite calcular el valor de h (paso)
n = 10
h = (x_final - x_inicial)/n
# Aplicamos el método de Predictor correcctor 3 veces (una para cada dato inicial) y obtenemos las soluciones numéricas
u, v = rugen_kutta(f, x, y0, h, n)


# Obtenemos las soluciones exactas
x_real = np.linspace(x_inicial, x_final, n)
y_real = f_exacta(x_real)

# Imprimimos la última y del bucle con 7 decimales 
print('w_100: {:.7f}'.format(v[-1]))

#Solucion real para y(1/2)
print('y(',x_final,'): {:.7f}'.format(f_exacta(x_final)) )

#Error absoluto
print('Error: {:.7f}'.format(error(f_exacta(x_final), v[-1])))


# Graficar la solución
# --------------------
# Dibujamos las soluciones numéricas
plt.plot(u, v)

# Dibujamos las soluciones exactas
plt.plot(x_real, y_real)

plt.grid(True)
plt.show()
