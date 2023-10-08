import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read data

diamonds = pd.read_csv('diamonds.csv')


# so we have 10 columns ; 8 numeric and 2 categorical. [carat, depth, table, price, x, y, z] are numeric and [cut, color, clarity] are categorical.

#General Notes:
#on clarity feature: I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)
# the table, x y and z imply the dimensions of the diamond.
# the depth is the height of the diamond, which is the z divided by the average of x and y.
# the table is the width of the diamond's table expressed as a percentage of its average diameter.
# the density of the diamond is the mass divided by the volume.
# the mass of the diamond is the density times the volume.
# the volume of the diamond is the product of x, y and z.


# So what do we want to know about this data?

# We'll check the distributions of the numeric features and see if there are any outliers.
# we can use the dimensions to calculate the volume of the diamond and use that as a feature.
# we can also use the dimensions to calculate the density of the diamond and use that as a feature.

#explore
print(diamonds.head())
print(diamonds.info())
print(diamonds.describe())

#our data has 53940 rows and 11 columns. so 53940 diamonds with 11 features.
#our most expensive diamond is 18,823 dollars and our cheapest diamond is 326 dollars.
#our largest diamond is 5.01 carats and our smallest diamond is 0.2 carats.

#check for nulls
print(diamonds.isnull().sum())

#checking distributions of numeric features
#lets make the price labels more readable by adding more markers on the plot using seaborn

sns.histplot(diamonds['price'], bins=50)
plt.xticks(np.arange(0, 20000, step=1000))  # steps of 1000 dollars
plt.yticks(np.arange(0, 10000, step=1000)) 
plt.xlabel('Price in Dollars')
plt.ylabel('Nr of Diamonds') 
plt.show()

#histogram of carat
sns.histplot(diamonds['carat'], bins=50)
plt.xticks(np.arange(0, 6, step=0.5))  # steps of 0.5 carat
plt.yticks(np.arange(0, 10000, step=1000))
plt.xlabel('Carat')
plt.ylabel('Nr of Diamonds')
plt.show()

