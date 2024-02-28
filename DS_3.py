# %% [markdown]
# Load Data from csv file

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\\uk_universities.csv')

# Histogram of world ranks
plt.hist(df['World_rank'], bins=20)
plt.xlabel('World Rank')
plt.ylabel('Frequency')
plt.title('Distribution of University World Ranks')
plt.savefig('world_rank_hist.png')

# %% [markdown]
# Scatter Plot for World Rank VS Student Satisfaction

# %%
plt.scatter(df['World_rank'], df['Student_satisfaction'])
plt.xlabel('World Rank') 
plt.ylabel('Student Satisfaction %')
plt.title('World Rank vs Student Satisfaction')
plt.savefig('world_rank_vs_satisfaction.png') 

# %%

plt.figure(figsize=(12, 8))  # Set the figure size

# Scatter plot with adjusted point size and transparency
plt.scatter(df['World_rank'], df['Student_satisfaction'], s=100, alpha=1)

plt.xlabel('World Rank') 
plt.ylabel('Student Satisfaction %')
plt.title('World Rank vs Student Satisfaction')


plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Sample data
np.random.seed(0)
df = pd.DataFrame({
    'World_rank': np.random.randint(1, 100, 100),
    'Student_satisfaction': np.random.randint(50, 100, 100)
})

plt.figure(figsize=(12, 8))  # Set the figure size

# Scatter plot with adjusted point size and transparency
plt.scatter(df['World_rank'], df['Student_satisfaction'])

plt.xlabel('World Rank') 
plt.ylabel('Student Satisfaction %')
plt.title('World Rank vs Student Satisfaction')

# Set y-axis ticks to show only a subset of values
plt.yticks(np.arange(50, 101, step=10))  # Change the step value as needed


plt.show()


# %% [markdown]
# Correlation and Descriptive Statistics for the data

# %%
# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap='RdBu_r',
            annot=True)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

print(df.describe())
print(df.corr())

# %% [markdown]
# A function to do the same thing

# %%

def visualize_correlation(df):
    """
    Calculate the correlation matrix and display a heatmap along with descriptive statistics for a DataFrame.

    Args:
    - df (DataFrame): The input DataFrame.

    Returns:
    - None
    """
    # Calculate the correlation matrix
    corr = df.corr()

    # Create the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, 
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap='RdBu_r',
                annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.show()

    # Display descriptive statistics
    print("Descriptive Statistics:")
    print(df.describe())
    print("\nCorrelation Matrix:")
    print(df.corr())

# Usage
visualize_correlation(df)


# %% [markdown]
# Box Plot

# %%
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
ax.boxplot(df['World_rank'])

ax.set_xlabel('World Rank')
ax.set_title('Box Plot of World Ranks')

plt.tight_layout()
plt.savefig('world_rank_box.png')

# %%




