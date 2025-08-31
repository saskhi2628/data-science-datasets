# Task 1 - Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Replace with your downloaded dataset path
data = pd.read_csv("data.csv")  
print(data.head())
plt.figure(figsize=(8,5))
sns.histplot(data['Age'], bins=10, kde=True, color='blue')
plt.title("Age Distribution in Population")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=data, palette='pastel')
plt.title("Gender Distribution in Population")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
}
