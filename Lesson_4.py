#!/usr/bin/env python
# coding: utf-8

# ### Exercise 1

# Загрузите модуль pyplot библиотеки matplotlib с псевдонимом plt, а также библиотеку numpy с псевдонимом np.

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Примените магическую функцию %matplotlib inline для отображения графиков в Jupyter Notebook и настройки конфигурации ноутбука со значением 'svg' для более четкого отображения графиков.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Создайте список под названием x с числами 1, 2, 3, 4, 5, 6, 7 и список y с числами 3.5, 3.8, 4.2, 4.5, 5, 5.5, 7.

# In[3]:


x=[1, 2, 3, 4, 5, 6, 7]
y=[3.5, 3.8 ,4.2, 4.5, 5, 5.5, 7]


# С помощью функции plot постройте график, соединяющий линиями точки с горизонтальными координатами из списка x и вертикальными - из списка y.

# In[4]:


plt.plot(x,y)
plt.show()


# Затем в следующей ячейке постройте диаграмму рассеяния (другие названия - диаграмма разброса, scatter plot).

# In[5]:


plt.scatter(x,y)
plt.show()


# ### Exercise 2

# С помощью функции linspace из библиотеки Numpy создайте массив t из 51 числа от 0 до 10 включительно.

# In[10]:


t = np.linspace(0, 10, 51)
t


# Создайте массив Numpy под названием f, содержащий косинусы элементов массива t.

# In[12]:


f = np.cos(t)
f


# Постройте линейную диаграмму, используя массив t для координат по горизонтали,а массив f - для координат по вертикали. Линия графика должна быть зеленого цвета.

# In[14]:


plt.plot(t,f,
        color = "green")
plt.show()


# Выведите название диаграммы - 'График f(t)'. Также добавьте названия для горизонтальной оси - 'Значения t' и для вертикальной - 'Значения f'.

# In[21]:


plt.plot(t,f,
        color = "green")
plt.title(
"Chart function f(t)")
plt.xlabel("t")
plt.ylabel("f")
plt.show()


# Ограничьте график по оси x значениями 0.5 и 9.5, а по оси y - значениями -2.5 и 2.5.

# In[22]:


plt.plot(t,f,
        color = "green")
plt.title(
"Chart function f(t)")
plt.xlabel("t")
plt.ylabel("f")
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show()


# ### Exercise 3*

# С помощью функции linspace библиотеки Numpy создайте массив x из 51 числа от -3 до 3 включительно.

# In[65]:


x = np.linspace(-3, 3, 51)
x


# Создайте массивы y1, y2, y3, y4 по следующим формулам:
# y1 = x**2
# y2 = 2 * x + 0.5
# y3 = -3 * x - 1.5
# y4 = sin(x)
# 

# In[66]:


import math
from math import sin


# In[69]:


y1 = x**2 
y2 = 2 * x + 0.5 
y3 = -3 * x - 1.5
vector=np.vectorize(np.float)
v=vector(x)
v
y4 = np.sin(v)
print(y1, y2, y3, y4)


# Используя функцию subplots модуля matplotlib.pyplot, создайте объект matplotlib.figure.Figure с названием fig и массив объектов Axes под названием ax,причем так, чтобы у вас было 4 отдельных графика в сетке, состоящей из двух строк и двух столбцов. В каждом графике массив x используется для координат по горизонтали.В левом верхнем графике для координат по вертикали используйте y1,в правом верхнем - y2, в левом нижнем - y3, в правом нижнем - y4.Дайте название графикам: 'График y1', 'График y2' и т.д.

# In[73]:


fig, ax=plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()

ax1.plot(x,y1)
ax1.set_title("Chart 1")

ax2.plot(x,y2)
ax2.set_title("Chart 2")

ax3.plot(x,y3)
ax3.set_title("Chart 3")

ax4.plot(x,y4)
ax4.set_title("Chart 4")


# Для графика в левом верхнем углу установите границы по оси x от -5 до 5.
# Установите размеры фигуры 8 дюймов по горизонтали и 6 дюймов по вертикали.
# Вертикальные и горизонтальные зазоры между графиками должны составлять 0.3.
# 

# In[95]:


fig, ax=plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()

fig.set_size_inches(8,6)
fig.subplots_adjust(wspace=0.3, hspace=0.3)

ax1.plot(x,y1)
ax1.set_title("Chart 1")
ax1.set_xlim(-5, 5)

ax2.plot(x,y2)
ax2.set_title("Chart 2")

ax3.plot(x,y3)
ax3.set_title("Chart 3")

ax4.plot(x,y4)
ax4.set_title("Chart 4")


# In[ ]:



