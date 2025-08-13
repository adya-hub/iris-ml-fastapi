# Use non-GUI backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')

# Import libraries
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Print first few rows and target labels
print(df.head())
print("Target labels:", data.target_names)

# Create pairplot
sns.pairplot(df, hue='target')

# Save the plot as an image file (instead of showing it in a window)
plt.savefig("iris_pairplot.png")

# Optional: print confirmation
print("Plot saved as iris_pairplot.png")
