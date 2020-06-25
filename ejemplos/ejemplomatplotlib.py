import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(2,9,10)
print(x)

y= x**2
print(y)

#Grafico 1
plt.plot(x,y, 'red')
plt.title("Este es mi grafico")
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.show()

#Grafico 2
plt.subplot(1,2,1)
plt.plot(x,y,'g')
plt.subplot(1,2,2)
plt.plot(y,x,'blue')
plt.show()

#Grafico 3
plt.subplot(2,1,1)
plt.plot(x,y,'g')
plt.subplot(2,1,2)
plt.plot(y,x,'blue')
plt.show()

#Grafico 4
figura = plt.figure()
grafico = figura.add_axes([0.1,0.1,0.9,0.9])
grafico.plot(x,y,'g')
grafico = figura.add_axes([0.15,0.5,0.4,0.4])
grafico.plot(y,x,'b')
plt.show()

#Multigraficos
figura, graficos = plt.subplots(nrows=2, ncols=2)
graficos[0][0].plot(x,y,'#0D95ED')
graficos[0][1].plot(x,y,color='r', linewidth=10, alpha=0.4, linestyle=':')
graficos[1][0].plot(x,y,'y',linestyle='-.')
graficos[1][1].plot(y,x,'g', linestyle='--')

graficos[0][0].set_title('Azul')
graficos[0][1].set_title('Rojo')
graficos[1][0].set_title('Amarillo')
graficos[1][1].set_title('Verde')
plt.show()

#Tamano
figura = plt.figure(figsize=(8,4))
grafico = figura.add_axes([0.1,0.1,0.9,0.9])
grafico.plot(y,x,'r')
plt.show()

#DosEnUno
figura = plt.figure(figsize=(8,4))
graficos = figura.add_axes([0.1,0.1,0.9,0.9])
graficos.plot(x,x**2, label='Al cuadrado')
graficos.plot(x,x**3, label='Al cubo')
graficos.legend(loc=0)
plt.show()

#Marcadores
figura = plt.figure()
graficos = figura.add_axes([0.1,0.1,0.9,0.9])
graficos.plot(x,y,'#0D95ED',marker='o', markersize=10)
plt.show()