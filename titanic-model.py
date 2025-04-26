import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the Titanic dataset
df = pd.read_csv('D:/titanic-survival-prediction/tested.csv')

# Select features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]   
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})  
X['Age'].fillna(X['Age'].median())  

y = df['Survived']  

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as titanic-model.pkl!")
