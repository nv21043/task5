# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data preprocessing
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df = df.dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a simple MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Flask web interface
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        input_data = [[pclass, sex, age, sibsp, parch, fare, embarked]]
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


