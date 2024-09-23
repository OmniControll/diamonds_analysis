import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

diamonds = pd.read_csv('C:\\Users\\Kays\\Documents\\CODING\\Backup_py\\diamonds_analysis\\diamonds.csv')



# Display DataFrame info
diamonds.info()


# some of our columns contain byte strings, we'll need to convert them to strings.
# we'll also need to convert the categorical features to strings.
diamonds['cut'] = diamonds['cut'].astype(str)
diamonds['color'] = diamonds['color'].astype(str)
diamonds['clarity'] = diamonds['clarity'].astype(str)

# fix byte strings
diamonds['cut'] = diamonds['cut'].str.replace("b'", '')
diamonds['cut'] = diamonds['cut'].str.replace("'", '')
diamonds['color'] = diamonds['color'].str.replace("b'", '')
diamonds['color'] = diamonds['color'].str.replace("'", '')
diamonds['clarity'] = diamonds['clarity'].str.replace("b'", '')
diamonds['clarity'] = diamonds['clarity'].str.replace("'", '')


# Business Question: 
# How do the physical attributes and quality grades of a diamond relate to its market price, 
# can we accurately predict the price of a diamond based on these features?"

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

# we're going to need to clean the data a bit before we can use it.
# we'll rename some columns to make them more readable.
#rename columns
diamonds.rename(columns={"'x'": 'length', "'y'": 'width', "'z'": 'height'}, inplace=True)
print(diamonds.head())

print(diamonds['clarity'].unique())
print(diamonds['color'].unique())

# rename the clarity values to make them more readable
diamonds['clarity'] = diamonds['clarity'].astype(str)
diamonds['clarity'].replace({'I1': 'Included', 'SI2': 'Slightly Included', 'SI1': 'Slightly Included',
                                'VS2': 'Very Slightly Included', 'VS1': 'Very Slightly Included',
                                'VVS2': 'Very Very Slightly Included-2', 'VVS1': 'Very Very Slightly Included-1',
                                'IF': 'Internally Flawless'}, inplace=True)

#rename the color values to make them more readable, and group them into colorless and near colorless
# the colorless is distinguished by the letters D, E and F, and the near colorless is distinguished by the letters G, H, I and J.
# it ranges from D (best) to J (worst), we'll try to make that apparent in the plot.
diamonds['color'] = diamonds['color'].astype(str)
diamonds['color'].replace({'D': 'D-Colorless', 'E': 'E-Colorless', 'F': 'F-Colorless',
                            'G': 'G-Near Colorless', 'H': 'H-Near Colorless', 'I': 'I-Near Colorless', 'J': 'J-Near Colorless'}, inplace=True)

print(diamonds['clarity'].unique())
print(diamonds['color'].unique())

diamonds.head()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15)) 

# Flattening the array of axes for easy indexing
axes = axes.flatten()

numeric_feats = ['carat', 'depth', 'table', 'price', 'length', 'width']
category_feats = ['cut', 'color', 'clarity']

all_feats = numeric_feats + category_feats  # Merging both lists
#define cut order
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
#define color order
color_order = ['D-Colorless', 'E-Colorless', 'F-Colorless', 'G-Near Colorless', 'H-Near Colorless', 'I-Near Colorless', 'J-Near Colorless']
#define clarity order
clarity_order = ['Included', 'Slightly Included', 'Very Slightly Included', 'Very Very Slightly Included-2', 'Very Very Slightly Included-1', 'Internally Flawless']

# Looping through all features and plotting
for i, feature in enumerate(all_feats):
    ax = axes[i]
    
    if feature in numeric_feats:
        sns.histplot(diamonds[feature], bins=50, ax=ax, color='skyblue', kde=True)
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        mean_val = diamonds[feature].mean()
        ax.axvline(mean_val, color='r', linestyle='--')
        ax.text(mean_val + 0.05, ax.get_ylim()[1] * 0.8, 'Mean: {:.2f}'.format(mean_val), bbox=dict(facecolor='red', alpha=0.5))
    else:
        #sorting the 'cut' feature by quality
        if feature == 'cut':
            sns.countplot(x=feature, data=diamonds, ax=ax, palette='viridis', order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, which='major')
        #sorting the 'color' feature by quality
        elif feature == 'color':
            sns.countplot(x=feature, data=diamonds, ax=ax, palette='viridis', order=['D-Colorless', 'E-Colorless', 'F-Colorless', 'G-Near Colorless', 'H-Near Colorless', 'I-Near Colorless', 'J-Near Colorless'])
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, which='major')
        #sorting the 'clarity' feature by quality
        elif feature == 'clarity':
            sns.countplot(x=feature, data=diamonds, ax=ax, palette='viridis', order=['Included', 'Slightly Included', 'Very Slightly Included', 'Very Very Slightly Included-2', 'Very Very Slightly Included-1', 'Internally Flawless'])
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, which='major')
        else:
            sns.countplot(x=feature, data=diamonds, ax=ax, palette='viridis')
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7, which='major')
        
    ax.set_xlabel(feature)
    ax.set_ylabel('Nr of Diamonds')
    ax.set_title(f'Distribution of Diamond {feature}')

plt.tight_layout()
plt.show()

#remove empty plots
axes[-1].remove()
axes[-2].remove()
axes[-3].remove()

#add title
fig.suptitle('Distribution of Diamond Features', fontsize=16)

plt.tight_layout()
plt.show()

#checking distributions of numeric features
# the depth is the height of the diamond, which is the z divided by the average of x and y.
#table is the width of the diamond's table expressed as a percentage of its average diameter.\

#lets make the price labels more readable by adding more markers on the plot using seaborn
#there are some outliers in the carat feature. we can remove them by removing the rows with carat > 3
#diamonds = diamonds[diamonds['carat'] < 3]
#But considering we're dealing with luxury products, it's not surprising to have some outliers representing rare cases. so we'll keep them.
#in linear regression, outliers can affect the model's performance. so we'll apply random forest regressor instead later on.


#checking unique combinations 
print(diamonds.groupby(['cut', 'color', 'clarity']).size())

# 276 unique combinations, this implies that we have 276 different types of diamonds in our data.

print(diamonds.groupby(['color', 'clarity']).size().sort_values(ascending=False)) 
# the most common diamond is a G-Near Colorless with a clarity of Slightly Included.