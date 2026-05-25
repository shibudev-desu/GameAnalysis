from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

try: 
  gameData = pd.read_csv("games.csv")
  print("Data arrived.")
except:
  print("Error reading data!")

enc = LabelEncoder()

gameData['Publisher_enc'] = enc.fit_transform(gameData['Publisher'])
gameData['Genre_enc'] = enc.fit_transform(gameData['Genre'])
gameData['Platform_enc'] = enc.fit_transform(gameData['Platform'])

X = gameData[['Genre_enc', 'Platform_enc', 'Publisher_enc', 'Year']]
y = (gameData['Global_Sales'] >= 1).astype(int) # Games that have more than one million sales

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # Train x Test ratio
    random_state=42 
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') 
model.fit(X_train, y_train) # Training model

# Faz previsões
y_pred = model.predict(X_test)

# Avalia
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))