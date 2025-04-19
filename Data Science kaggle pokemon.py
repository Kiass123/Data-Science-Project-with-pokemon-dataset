#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # visualization tool

# Load your CSV file (update the file path as needed)
file_path = r"D:\Kaggle\Data Science\pokemon\pokemon.csv"  # Use raw string to avoid escape characters

# Read the dataset
data = pd.read_csv(file_path)


# In[2]:


# Show the first 5 rows of the dataset
print(data.head())


# In[3]:


# Example plot: Visualize the distribution of the 'Attack' column
sns.histplot(data['Attack'], kde=True)
plt.title('Distribution of Attack')  # Customize title
plt.show()


# In[4]:


data.info()


# In[5]:


# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr_matrix = numeric_data.corr()

# Display the correlation matrix
print(corr_matrix)


# In[6]:


# Optional: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(18, 18))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[7]:


data.head(10)


# In[8]:


data.columns


# In[9]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[10]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot


# In[11]:


# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,10))
plt.show()


# In[12]:


# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()


# In[13]:


#create dictionary and look its keys and values
dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())


# In[14]:


# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['spain']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)


# In[15]:


# In order to run all code you need to take comment this line
#del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted


# In[16]:


file_path = r"D:\Kaggle\Data Science\pokemon\pokemon.csv" #input csv file


# In[17]:


data = pd.read_csv(file_path) #read the dataset


# In[18]:


print(data)


# In[19]:


series = data['Defense']        # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[20]:


# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)


# In[21]:


# 1 - Filtering Pandas data frame
x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200
data[x]


# In[22]:


# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]


# In[23]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Defense']>200) & (data['Attack']>100)]


# In[24]:


# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1
print(i,' is equal to 5')


# In[25]:


# stay in loop if condition ( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)


# In[26]:


# example of what we learn above
def tuple_ex():
    """ return defined t tuple"""
    t = (1,2,3)
    return t
a,b,c = tuple_ex()
print(a,b,c)


# In[27]:


# guess prints what
x = 2
def f():
     x = 3
     return x
print(x) # x = 2 global scope
print(f()) # x = 3 local scope


# In[28]:


# what if there is no local scope
x = 2
def f():
    y = 2*x  # there is no local scope x
    return y
print (f()) # it uses global scope x  
# First local scope searched, then global scope searched, 
# if two of them cannot be found lastly built in scope searched.


# In[29]:


# How can we learn what is built in scope
import builtins
dir(builtins)


# In[30]:


#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())   


# In[31]:


# default arguments
def f(a, b = 1, c = 2):
    y = a + b + c
    return y
print(f(2))
# what if we want to change default arguments
print(f(5,4,3))


# In[32]:


# flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)
f(country = 'spain', capital = 'madrid', population = 123456)


# In[33]:


# lambda function
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))


# In[34]:


number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))


# In[35]:


# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration


# In[36]:


# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)


# In[37]:


z_list = list(z)
print(z_list)


# In[38]:


un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuple
print(un_list1)
print(un_list2)
print(type(un_list2))


# In[39]:


# Example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)


# In[40]:


# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 
        if i == 10
          else i-5 
            if i < 7 
               else i+5 
        for i in num1]
print(num2)


# In[41]:


# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.Speed)/len(data.Speed)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later


# In[42]:


# CLEANING DATA


# In[43]:


# Load your CSV file (update the file path as needed)
file_path = r"D:\Kaggle\Data Science\pokemon\pokemon.csv"  # Use raw string to avoid escape characters

# Read the dataset
data = pd.read_csv(file_path)
data.head()


# In[44]:


# tail shows last 5 rows
data.tail()


# In[45]:


# columns gives column names of features
data.columns


# In[46]:


# shape gives number of rows and columns in a tuble
data.shape


# In[47]:


# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info


# In[48]:


# For example lets look frequency of pokemom types
print(data['Type 1'].value_counts(dropna =False)) 
# if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon


# In[49]:


1,2,3,4,200


# In[50]:


# For example max HP is 255 or min defense is 5
data.describe() #ignore null entries


# In[51]:


# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Green line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack',by = 'Legendary')


# In[52]:


# Firstly I create new data from pokemons data to explain melt nore easily.
data_new = data.head()    # I only take 5 rows into new data
data_new


# In[53]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted


# In[54]:


# Pivoting Data - reverse of melting
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')


# In[55]:


#Concatenating Data - concatenate two dataframe
# Firstly lets create 2 data frame
data1 = data.head()
data2 = data.head()
conc_data_row = pd.concat([data1,data2], axis =0, ignore_index =True) #axis =0 : adds datafreames in row
conc_data_row


# In[56]:


data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column
conc_data_col


# In[57]:


data.dtypes


# In[58]:


# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')


# In[59]:


data.dtypes


# In[60]:


# missing data and testing with assert---
# Lets look at does pokemon data have nan value
# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.
data.info()


# In[61]:


# Lets chech Type 2
data["Type 2"].value_counts(dropna =False)
# As you can see, there are 386 NAN value


# In[62]:


# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?


# In[63]:


#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true


# In[64]:


# In order to run all code, we need to make this line comment
# assert 1==2 # return error because it is false


# In[68]:


data['Type 2'].fillna('None', inplace=True)

# returns nothing because we drop nan values


# In[69]:


data["Type 2"].fillna('empty',inplace = True)


# In[70]:


assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values


# In[71]:


# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int


# In[72]:


BUILDING DATA FRAMES FROM SCRATCH


# In[73]:


# data frames from dictionary
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# In[74]:


# Add new columns
df["capital"] = ["madrid","paris"]
df


# In[75]:


# Broadcasting
df["income"] = 0 #Broadcasting entire column
df


# In[76]:


#VISUAL EXPLORATORY DATA ANALYSIS


# In[77]:


# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# it is confusing


# In[78]:


# subplots
data1.plot(subplots = True)
plt.show()


# In[79]:


# scatter plot  
data1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()


# In[80]:


# hist plot
data1.plot(kind="hist", y="Defense", bins=50, range=(0, 250), density=True)


# In[81]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),density = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),density = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt


# In[82]:


#STATISTICAL EXPLORATORY DATA ANALYSIS
#count: number of entries
#mean: average of entries
#std: standart deviation
#min: minimum entry
#25%: first quantile
#50%: median or second quantile
#75%: third quantile
#max: maximum entry


# In[83]:


data.describe()


# In[84]:


#INDEXING PANDAS TIME SERIES¶
datetime = object
parse_dates(boolean): Transform date to ISO 8601 (yyyy-mm-dd hh:mm:ss ) format


# In[ ]:


time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))


# In[86]:


# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 


# In[87]:


# Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])


# In[89]:


# We will use data2 that we create at previous part
# Select only numeric columns and then resample and take mean
data2.select_dtypes(include='number').resample("A").mean()


# In[92]:


# Lets resample with month
data2.select_dtypes(include='number').resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months


# In[94]:


# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
# Resample and interpolate only numeric columns
data2.resample("M").first().select_dtypes(include='number').interpolate("linear")


# In[98]:


# Or we can interpolate with mean()
data2.select_dtypes(include='number').resample("M").mean().interpolate("linear")


# In[ ]:


#MANIPULATING DATA FRAMES WITH PANDAS
INDEXING DATA FRAMES
Indexing using square brackets
Using column attribute and row label
Using loc accessor
#Selecting only some columns


# In[99]:


#read data
data = pd.read_csv(r"D:\Kaggle\Data Science\pokemon\pokemon.csv")
data = data.set_index("#")
data.head()


# In[100]:


# indexing using square brackets
data["HP"][1]


# In[101]:


# using column attribute and row label
data.HP[1]


# In[102]:


# using loc accessor
data.loc[1,["HP"]]


# In[103]:


# Selecting only some columns
data[["HP","Attack"]]


# In[ ]:


#SLICING DATA FRAME¶
Difference between selecting columns
Series and data frames
Slicing and indexing series
Reverse slicing
#From something to end


# In[104]:


# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames


# In[105]:


# Slicing and indexing series
data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive


# In[106]:


# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 


# In[107]:


# From something to end
data.loc[1:10,"Speed":] 


# In[ ]:


#FILTERING DATA FRAMES
#Creating boolean series Combining filters Filtering column based others


# In[108]:


# Creating boolean series
boolean = data.HP > 200
data[boolean]


# In[109]:


# Combining filters
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]


# In[110]:


# Filtering column based others
data.HP[data.Speed<15]


# In[ ]:


#TRANSFORMING DATA
Plain python functions
Lambda function: to apply arbitrary python function to every element
#Defining column using other columns


# In[111]:


# Plain python functions
def div(n):
    return n/2
data.HP.apply(div)


# In[112]:


# Or we can use lambda function
data.HP.apply(lambda n : n/2)


# In[113]:


# Defining column using other columns
data["total_power"] = data.Attack + data.Defense
data.head()


# In[ ]:


#INDEX OBJECTS AND LABELED DATA
#index: sequence of label


# In[114]:


# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()


# In[115]:


# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1)
data3.head()


# In[ ]:


# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
# data= data.set_index("#")
# also you can use 
# data.index = data["#"]


# In[116]:


#HIERARCHICAL INDEXING¶
#Setting indexing
# lets read data frame one more time to start from beginning
data = pd.read_csv(r"D:\Kaggle\Data Science\pokemon\pokemon.csv")
data.head()
# As you can see there is index. However we want to set one or more column to be index


# In[117]:


# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Type 1","Type 2"]) 
data1.head(100)
# data1.loc["Fire","Flying"] # howw to use indexes


# In[ ]:


#PIVOTING DATA FRAMES
#pivoting: reshape tool


# In[118]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df


# In[119]:


# pivoting
df.pivot(index="treatment",columns = "gender",values="response")


# In[ ]:


#STACKING and UNSTACKING DATAFRAME¶
#deal with multi label indexes
#level: position of unstacked index
#swaplevel: change inner and outer level index position


# In[120]:


df1 = df.set_index(["treatment","gender"])
df1
# lets unstack it


# In[121]:


# level determines indexes
df1.unstack(level=0)


# In[122]:


df1.unstack(level=1)


# In[123]:


# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2


# In[ ]:


#MELTING DATA FRAMES
#Reverse of pivoting


# In[124]:


df


# In[125]:


# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"])


# In[126]:


#CATEGORICALS AND GROUPBY
# We will use df
df


# In[128]:


# according to treatment take means of other features
df.groupby("treatment").mean(numeric_only=True)
   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min


# In[129]:


# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() 


# In[130]:


df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
#df["gender"] = df["gender"].astype("category")
#df["treatment"] = df["treatment"].astype("category")
#df.info()


# In[ ]:




