#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors 

from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[10]:


df=pd.read_csv("C:/Users/user/Desktop/IndianFoodDatasetCSV.csv")


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


df.isna().sum()


# In[15]:


df[df['Ingredients'].isna()]


# ## Removing null values & unwanted columns

# In[16]:


df = df[df['Ingredients'].notna()]


# In[17]:


df.shape


# In[18]:


df.isna().sum()


# In[19]:


df.head()


# In[20]:


df = df.drop(['Srno','RecipeName','Ingredients','Instructions','TranslatedInstructions','URL'],axis=1)


# In[21]:


df.head()


# In[22]:


df['TranslatedRecipeName'].nunique()


# In[23]:


df.TranslatedRecipeName.value_counts()


# In[24]:


df.Cuisine.value_counts()


# In[25]:


df[df.Cuisine == 'Burmese']


# In[26]:


df2 = df[df['Cuisine'].isin(['Indian','North Indian Recipes','South Indian Recipes','Bengali Recipes',
                            'Maharashtrian Recipes','Kerala Recipes','Tamil Nadu',
                            'Karnataka','Fusion','Rajasthani','Andhra','Gujarati Recipes',
                            'Goan Recipes','Punjabi','Chettinad','Kashmiri','Mangalorean',
                            'Indo Chinese','Parsi Recipes','Awadhi','Oriya Recipes','Sindhi',
                            'Konkan','Mughlai','Bihari','Assamese','Hyderabadi','North East India Recipes',
                            'Himachal','Sri Lankan','Udupi','Coorg','Uttar Pradesh','North Karnataka',
                            'Coastal Karnataka','Malabar','Lucknowi','South Karnataka','Malvani',
                            'Nagaland','Uttarakhand-North Kumaon','Kongunadu','Haryana','Jharkhand'
                            ])]


# In[27]:


df2.Cuisine.value_counts()


# In[28]:


df2.shape


# In[29]:


df2.head()


# In[30]:


df2.Course.value_counts()


# In[31]:


df2.Diet.value_counts()


# ## EDA

# ### TotalTimeInMins

# In[32]:


df2['TotalTimeInMins'].describe()


# In[33]:


# Specify the percentiles
percentiles = [95, 97.5, 99, 99.9]

# Calculate the percentiles using pandas
percentile_results = df2['TotalTimeInMins'].quantile(q=[p / 100 for p in percentiles])

# Display the results
for p, result in zip(percentiles, percentile_results):
    print(f'The {p}th percentile of the data is: {result}')


# In[34]:


# Create a boxplot
sns.boxplot(x=df2['TotalTimeInMins'])

# Show the boxplot
plt.title('Boxplot of Total Times')
plt.show()


# In[35]:


# Filter the DataFrame for preptimes less than 100
filtered_df = df2[df2['TotalTimeInMins'] <= 100]

# Create a boxplot
sns.boxplot(x=filtered_df['TotalTimeInMins'])

# Show the boxplot
plt.title('Boxplot of Total Times < 100')
plt.show()


# In[36]:


# Create a histogram
sns.histplot(df2['TotalTimeInMins'], bins=50, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Total Times')
plt.show()


# In[37]:


# Create a histogram (99.9th percentile)
filtered_df = df2[df2['TotalTimeInMins'] <= 730]

sns.histplot(filtered_df['TotalTimeInMins'], bins=10, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Total Times')
plt.show()


# In[38]:


# Create a histogram (95th percentile)
filtered_df = df2[df2['TotalTimeInMins'] <= 145]

sns.histplot(filtered_df['TotalTimeInMins'], bins=10, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Total Times')
plt.show()


# ### PrepTimeInMins

# In[39]:


df2['PrepTimeInMins'].describe()


# In[40]:


# Specify the percentiles
percentiles = [95, 97.5, 99, 99.9]

# Calculate the percentiles using pandas
percentile_results = df2['PrepTimeInMins'].quantile(q=[p / 100 for p in percentiles])

# Display the results
for p, result in zip(percentiles, percentile_results):
    print(f'The {p}th percentile of the data is: {result}')


# In[41]:


# Create a boxplot
sns.boxplot(x=df2['PrepTimeInMins'])

# Show the boxplot
plt.title('Boxplot of Preparation Times')
plt.show()


# In[42]:


# Filter the DataFrame for preptimes less than 100
filtered_df = df2[df2['PrepTimeInMins'] <= 100]

# Create a boxplot
sns.boxplot(x=filtered_df['PrepTimeInMins'])

# Show the boxplot
plt.title('Boxplot of Preparation Times < 100')
plt.show()


# In[43]:


# Create a histogram
sns.histplot(df2['PrepTimeInMins'], bins=50, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Preparation Times')
plt.show()


# In[44]:


# Create a histogram (99.9th percentile)
filtered_df = df2[df2['PrepTimeInMins'] <= 720]

sns.histplot(filtered_df['PrepTimeInMins'], bins=10, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Preparation Times')
plt.show()


# In[45]:


# Create a histogram (95th percentile)
filtered_df = df2[df2['PrepTimeInMins'] <= 70]

sns.histplot(filtered_df['PrepTimeInMins'], bins=10, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Preparation Times')
plt.show()


# ### CookTimeInMins

# In[46]:


df2['CookTimeInMins'].describe()


# In[47]:


# Specify the percentiles
percentiles = [95, 97.5, 99, 99.9]

# Calculate the percentiles using pandas
percentile_results = df2['CookTimeInMins'].quantile(q=[p / 100 for p in percentiles])

# Display the results
for p, result in zip(percentiles, percentile_results):
    print(f'The {p}th percentile of the data is: {result}')


# In[48]:


# Create a boxplot
sns.boxplot(x=df2['CookTimeInMins'])

# Show the boxplot
plt.title('Boxplot of Cooking Times')
plt.show()


# In[49]:


# Filter the DataFrame for preptimes less than 60
filtered_df = df2[df2['CookTimeInMins'] <= 60]

# Create a boxplot
sns.boxplot(x=filtered_df['CookTimeInMins'])

# Show the boxplot
plt.title('Boxplot of Cooking Times < 100')
plt.show()


# In[50]:


# Create a histogram
sns.histplot(df2['CookTimeInMins'], bins=50, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Cooking Times')
plt.show()


# In[51]:


# Create a histogram (99.9th percentile)
filtered_df = df2[df2['CookTimeInMins'] <= 260]

sns.histplot(filtered_df['CookTimeInMins'], bins=10, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Cooking Times')
plt.show()


# In[52]:


# Create a histogram (95th percentile)
filtered_df = df2[df2['CookTimeInMins'] <= 60]

sns.histplot(filtered_df['CookTimeInMins'], bins=10, kde=True)  # Adjust 'bins' as needed

# Show the histogram
plt.title('Histogram of Cooking Times')
plt.show()


# ### Relationship

# In[53]:


sns.pairplot(df2[['CookTimeInMins', 'PrepTimeInMins', 'TotalTimeInMins','Servings']])
plt.show()


# In[54]:


# Calculate the correlation matrix
correlation_matrix = df2[['CookTimeInMins', 'PrepTimeInMins', 'TotalTimeInMins','Servings']].corr()

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Display the heatmap
plt.title('Heatmap of Time Variables')
plt.show()


# In[55]:


result = df2.groupby('Diet')[['TotalTimeInMins', 'PrepTimeInMins', 'CookTimeInMins','Servings']].mean()

print(result)


# In[56]:


result = df2.groupby('Course')[['TotalTimeInMins', 'PrepTimeInMins', 'CookTimeInMins','Servings']].mean()

print(result)


# In[57]:


result = df2.groupby('Cuisine')[['TotalTimeInMins', 'PrepTimeInMins', 'CookTimeInMins','Servings']].mean()

print(result)


# In[58]:


df2.head()


# In[59]:


# For regular expressions
import re
# For handling string
import string
# For performing mathematical operations
import math
import nltk


# ## Cosine Similarity & Euclidean Distance and Linear Kernel

# In[60]:


import nltk
vocabulary = nltk.FreqDist()
# This was done once I had already preprocessed the ingredients
for ingredients in df2['TranslatedIngredients']:
    ingredients = ingredients.split()
    vocabulary.update(ingredients)
for word, frequency in vocabulary.most_common(200):
    print(f'{word};{frequency}')


# In[61]:


for index,text in enumerate(df2['TranslatedIngredients'][35:40]):
  print('Ingredient %d:\n'%(index+1),text)


# In[62]:


#lowercase
df2['cleaned_ingredient']=df2['TranslatedIngredients'].apply(lambda x: x.lower())

print(df2['cleaned_ingredient'])


# In[63]:


#Remove digits and words containing digits
df2['cleaned_ingredient']=df2['cleaned_ingredient'].apply(lambda x: re.sub('\w*\d\w*','', x))

print(df2['cleaned_ingredient'])


# In[64]:


# remove - / 
df2['cleaned_ingredient'] = df2['cleaned_ingredient'].apply(lambda x: re.sub(r'[-/+]', ' ', x))

print(df2['cleaned_ingredient'])


# In[65]:


# remove punctuation
df2['cleaned']=df2['cleaned_ingredient'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

print(df2['cleaned'])


# In[66]:


def remove_units(text):
    units_to_remove = ["pinch","inch","tbsp","cup", "cups", "tablespoon","tablespoons","teaspoons", "teaspoon","ounces", "ounce","gram","grams", "kg", "मिटर"]  # Add other units as needed
    for unit in units_to_remove:
        text = text.replace(unit, "")
    return text

df2['cleaned2'] = df2['cleaned'].apply(remove_units)

# Display the DataFrame
print(df2['cleaned2'])


# In[67]:


# Removing extra spaces
df2['cleaned2']=df2['cleaned2'].apply(lambda x: re.sub(' +',' ',x))

print(df2['cleaned2'])


# In[68]:


for index,text in enumerate(df2['cleaned2'][35:40]):
  print('Ingredient %d:\n'%(index+1),text)


# In[69]:


# tokenization
df2['cleaned_tokens'] = df2['cleaned2'].apply(lambda x: x.split())
print (df2['cleaned_tokens'])


# In[70]:


#Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
df2['cleaned_lemmas'] = df2['cleaned_tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

print (df2['cleaned_lemmas'])


# In[71]:


# remove stopwords
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df2['cleaned_no_stopwords'] = df2['cleaned_lemmas'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

print (df2['cleaned_no_stopwords'])


# In[72]:


df2 = df2.dropna(subset=['cleaned_no_stopwords'])


# In[73]:


# words frequency analysis
from collections import Counter

word_frequencies = Counter([word for words in df2['cleaned_no_stopwords'] for word in words])

print (word_frequencies)


# In[74]:


# Assuming df2 is your DataFrame
grouped_by_diet = df2.groupby('Diet')

# Example: Get word frequencies for each diet group
word_frequencies_by_diet = grouped_by_diet['cleaned_no_stopwords'].apply(lambda x: Counter([word for word_list in x for word in word_list]))
print(word_frequencies_by_diet)


# In[75]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(word_frequencies)

print (tfidf_matrix)


# In[76]:


import time

# Record start time
start = time.time()

#Explore Similarity Metrics:
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

print (similarity_matrix)

print("Time taken: %s seconds" % (time.time() - start))


# In[77]:


# Record start time
start = time.time()

euclidean_distance_matrix = euclidean_distances(tfidf_matrix)

print (euclidean_distance_matrix)

print("Time taken: %s seconds" % (time.time() - start))


# In[78]:


# Record start time
start = time.time()

linear_kernel_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

print (linear_kernel_matrix)

print("Time taken: %s seconds" % (time.time() - start))


# In[79]:


if 'cleaned_no_stopwords' not in df2.columns or df2['cleaned_no_stopwords'].isnull().all():
    print("Error: 'cleaned_no_stopwords' column not found or is empty.")
else:

    ingredient = "karela"
    ingredient_indices = df2[df2['cleaned_no_stopwords'].astype(str).str.contains(ingredient, case=False)].index

    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
    else:
        # Display a few records to confirm the data
        print(df2.loc[ingredient_indices, 'cleaned_no_stopwords'])


# In[80]:


print(df2.columns)


# In[81]:


df2.head()


# In[82]:


def recommend_food(ingredient, diet,course, similarity_matrix, df2):
    # Check for NaN values in 'cleaned_no_stopwords' column and 'Servings' and 'PrepTime' columns
    nan_indices = df2[df2['cleaned_no_stopwords'].isna() | df2['Diet'].isna() | df2['Course'].isna()].index
    df2 = df2.drop(index=nan_indices)

    # Find the index of the ingredient in the DataFrame
    ingredient_indices = df2[df2['cleaned_no_stopwords'].apply(lambda x: ingredient in x)].index

    # Check for NaN values for the specified ingredient
    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
        return None

    # Get the similarity scores for the corresponding row
    idx = ingredient_indices[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Calculate weighted similarity scores based on the presence of the ingredient
    weighted_scores = [
        (i, score + 0.5 if ingredient in df2['cleaned_no_stopwords'].iloc[i] else score + 0.3 if diet in df2['Diet'].iloc[i] else score+ 0.1 if course in df2['Course'].iloc[i] else score)
        for i, score in similarity_scores
    ]

    # Sort the food items by weighted similarity score
    sorted_items = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

    # Extract the indices of recommended food items (excluding the input ingredient)
    recommended_indices = [i[0] for i in sorted_items if i[0] != idx]

    # Get the entire rows for recommended items from the original DataFrame
    recommended_data = df2.iloc[recommended_indices]
        
    return recommended_data


# In[83]:


top_recommendations = recommend_food("egg", "Eggetarian","Breakfast", similarity_matrix, df2)
print(top_recommendations)


# In[84]:


def recommend_food(ingredient, diet,course, similarity_matrix, df2, top_n=5):
    # Check for NaN values in 'cleaned_no_stopwords' column and 'Servings' and 'PrepTime' columns
    nan_indices = df2[df2['cleaned_no_stopwords'].isna() | df2['Diet'].isna() | df2['Course'].isna()].index
    df2 = df2.drop(index=nan_indices)

    # Find the index of the ingredient in the DataFrame
    ingredient_indices = df2[df2['cleaned_no_stopwords'].apply(lambda x: ingredient in x)].index

    # Check for NaN values for the specified ingredient
    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
        return None

    # Get the similarity scores for the corresponding row
    idx = ingredient_indices[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Calculate weighted similarity scores based on the presence of the ingredient
    weighted_scores = [
        (i, score + 0.5 if ingredient in df2['cleaned_no_stopwords'].iloc[i] else score + 0.3 if diet in df2['Diet'].iloc[i] else score+ 0.1 if course in df2['Course'].iloc[i] else score)
        for i, score in similarity_scores
    ]

    # Sort the food items by weighted similarity score
    sorted_items = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

    # Extract the indices of top recommended food items (excluding the input ingredient)
    top_recommended_indices = [i[0] for i in sorted_items if i[0] != idx][:top_n]
    
    # Get the entire rows for recommended items from the original DataFrame
    recommended_data = df2.iloc[top_recommended_indices]
    
    return recommended_data


# In[85]:


top_recommendations = recommend_food("egg", "Eggetarian","Breakfast", similarity_matrix, df2, top_n=5)
print(top_recommendations)


# In[86]:


import numpy as np

def recommend_food_euclidean_distance(ingredient, diet, course, euclidean_distance_matrix, df2, top_n=10):
    # Check for NaN values in 'cleaned_no_stopwords' column and 'Diet' and 'Course' columns
    nan_indices = df2[df2['cleaned_no_stopwords'].isna() | df2['Diet'].isna() | df2['Course'].isna()].index
    df2 = df2.drop(index=nan_indices)

    # Find the index of the ingredient in the DataFrame
    ingredient_indices = df2[df2['cleaned_no_stopwords'].apply(lambda x: ingredient in x)].index

    # Check for NaN values for the specified ingredient
    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
        return None

    # Get the similarity scores (inverse of Euclidean distance) for the corresponding row
    idx = ingredient_indices[0]
    # similarity_scores = list(enumerate(1 / (1 + np.linalg.norm(similarity_matrix - similarity_matrix[idx], axis=1))))
    euclidean_scores = list(enumerate(euclidean_distance_matrix[idx]))
    
    euclidean_scores = sorted(euclidean_scores, key=lambda x: x[1], reverse=True)
    
    # Calculate weighted similarity scores based on the presence of the ingredient, diet, and course
    weighted_scores = [
        (i, score + 0.5 if ingredient in df2['cleaned_no_stopwords'].iloc[i] else score + 0.3 if diet in df2['Diet'].iloc[i] else
             score + 0.1 if course in df2['Course'].iloc[i] else score)
        for i, score in euclidean_scores
    ]

    # Sort the food items by weighted similarity score
    sorted_items = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    
    # Extract the indices of top recommended food items (excluding the input ingredient)
    top_recommended_indices = [i[0] for i in sorted_items if i[0] != idx][:top_n]
    
    # Get the entire rows for recommended items from the original DataFrame
    recommended_data_euclidean = df2.iloc[top_recommended_indices]
    
    return recommended_data_euclidean


# In[87]:


top_recommendations = recommend_food_euclidean_distance("egg", "Eggetarian","Breakfast", euclidean_distance_matrix, df2, top_n=5)
print(top_recommendations)


# In[88]:


import numpy as np

def recommend_food_linear_kernel(ingredient, diet, course, linear_kernel_matrix, df2, top_n=10):
    # Check for NaN values in 'cleaned_no_stopwords' column and 'Diet' and 'Course' columns
    nan_indices = df2[df2['cleaned_no_stopwords'].isna() | df2['Diet'].isna() | df2['Course'].isna()].index
    df2 = df2.drop(index=nan_indices)

    # Find the index of the ingredient in the DataFrame
    ingredient_indices = df2[df2['cleaned_no_stopwords'].apply(lambda x: ingredient in x)].index

    # Check for NaN values for the specified ingredient
    if len(ingredient_indices) == 0:
        print(f"No records found for ingredient '{ingredient}'.")
        return None

    # Get the similarity scores (inverse of Euclidean distance) for the corresponding row
    idx = ingredient_indices[0]
    linear_kernel_scores = list(enumerate(linear_kernel_matrix[idx]))
    
    linear_kernel_scores = sorted(linear_kernel_scores, key=lambda x: x[1], reverse=True)    

    # Calculate weighted similarity scores based on the presence of the ingredient, diet, and course
    weighted_scores = [
        (i, score + 0.5 if ingredient in df2['cleaned_no_stopwords'].iloc[i] else score  + 0.3 if diet in df2['Diet'].iloc[i] else
             score + 0.1 if course in df2['Course'].iloc[i] else score)
        for i, score in linear_kernel_scores
    ]

    # Sort the food items by weighted similarity score
    sorted_items = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    
    # Extract the indices of top recommended food items (excluding the input ingredient)
    top_recommended_indices = [i[0] for i in sorted_items if i[0] != idx][:top_n]
    
    # Get the entire rows for recommended items from the original DataFrame
    recommended_data_linear_kernel = df2.iloc[top_recommended_indices]
    
    return recommended_data_linear_kernel


# In[89]:


top_recommendations = recommend_food_linear_kernel("egg", "Eggetarian","Breakfast", linear_kernel_matrix, df2, top_n=5)
print(top_recommendations)


# In[92]:


import streamlit as st

tfidf_matrix = tfidf_matrix 

# Print the TF-IDF matrix for checking
print("TF-IDF Matrix:")
print(tfidf_matrix)


# User selects similarity method
similarity_method = st.selectbox("Select Similarity Method", ["Cosine Similarity", "Euclidean Distance", "Linear Kernel"])

# Compute similarity matrix based on user selection
if similarity_method == "Cosine Similarity":
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
elif similarity_method == "Euclidean Distance":
    similarity_matrix = euclidean_distances(tfidf_matrix)
elif similarity_method == "Linear Kernel":
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Display the similarity matrix or use it in your application
st.write("Similarity Matrix:")
st.write(similarity_matrix)


