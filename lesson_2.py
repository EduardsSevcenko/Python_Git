#!/usr/bin/env python
# coding: utf-8

# ### Lesson-1

# ### Ex.1.1

# Импортируйте библиотеку Numpy и дайте ей псевдоним np.
# Создайте массив Numpy под названием a размером 5x2, то есть состоящий из 5 строк и 2 столбцов. Первый столбец должен содержать числа 1, 2, 3, 3, 1, а второй - числа 6, 8, 11, 10, 7. Будем считать, что каждый столбец - это признак, а строка - наблюдение. Затем найдите среднее значение по каждому признаку, используя метод mean массива Numpy. Результат запишите в массив mean_a, в нем должно быть 2 элемента.

# In[6]:


import numpy as np


# In[21]:


a=np.array([[1, 6],
           [3, 11],
           [3, 10],
           [1, 7]])

mean_a=np.mean(a, axis=0)
print(mean_a)


# ### Ex.1.2

# Вычислите массив a_centered, отняв от значений массива “а” средние значения соответствующих признаков, содержащиеся в массиве mean_a. Вычисление должно производиться в одно действие. Получившийся массив должен иметь размер 5x2.

# In[27]:


a_centered=a-mean_a
print(a_centered)


# ### Ex.1.3

# Найдите скалярное произведение столбцов массива a_centered. В результате должна получиться величина a_centered_sp. Затем поделите a_centered_sp на N-1, где N - число наблюдений.

# In[31]:


a_centered_sp=a_centered*a_centered
print(a_centered_sp)
a_centered_sb=a_centered_sp-a
print(a_centered_sb)


# ### Ex.2.1

# Импортируйте библиотеку Pandas и дайте ей псевдоним pd. Создайте датафрейм authors со столбцами author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев', 'Чехов', 'Островский'].
# Затем создайте датафрейм book cо столбцами author_id, book_title и price, в которых соответственно содержатся данные:  
# [1, 1, 1, 2, 2, 3, 3],
# ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
# [450, 300, 350, 500, 450, 370, 290].
# 

# In[109]:


import pandas as pd
auth={
    "author_id":[1, 2, 3],
    "author_name":['Тургенев', 'Чехов', 'Островский']
}

bk={
    "author_id":[1, 1, 1, 2, 2, 3, 3],
    "book_title":['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    "price":[450, 300, 350, 500, 450, 370, 290]
}

authors=pd.DataFrame(auth)
book=pd.DataFrame(bk)
print(authors)
print(book)


# ### Ex.2.2

# Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id.

# In[116]:


authors_price=pd.merge(authors, book, on='author_id', how='left')
authors_price


# ### Ex.2.3

# Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами.

# In[117]:


top5 = authors_price.nlargest(5,'price')
top5


# In[131]:


#authors_price_sum = authors_price.groupby('author_name').agg({'price':sum})
#authors_price_sum


# In[129]:


sum=authors_price.groupby('author_name').sum()
sum


# In[121]:


count=authors_price.groupby('author_name').count()
count


# In[135]:


mean=sum/count
mean


# ### Ex.2.4

# Создайте датафрейм authors_stat на основе информации из authors_price. В датафрейме authors_stat должны быть четыре столбца:
# author_name, min_price, max_price и mean_price,
# в которых должны содержаться соответственно имя автора, минимальная, максимальная и средняя цена на книги этого автора.
# 

# In[133]:


authors_stat=authors_price.groupby('author_name').agg({'price':[max, min]})
authors_stat


# In[136]:


df2=authors_stat.assign(mean = mean['price'])


# In[137]:


df2


# In[ ]:
