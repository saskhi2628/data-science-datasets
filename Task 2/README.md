import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Download dataset from the GitHub link or Kaggle Titanic dataset
data = pd.read_csv("titanic.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
data.drop(columns=['Cabin'], inplace=True)
print(data.dtypes)
sns.countplot(x='Survived', data=data, palette='Set2')
plt.title("Survival Distribution")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=data, palette='pastel')
plt.title("Survival by Gender")
plt.show()
sns.histplot(data['Age'], bins=30, kde=True, color='blue')
plt.title("Age Distribution of Passengers")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=data, palette='muted')
plt.title("Survival by Passenger Class")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
}
