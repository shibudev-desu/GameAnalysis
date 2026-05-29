from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
# Script using regression rather than classification
# Model needs to try and guess the number of ´global_sales´ rather than if 1M sales where reached
# ============================================================

try: 
  gameData = pd.read_csv("games.csv")
  print("Data arrived.")
except:
  print("Error reading data!")

# X is the features
# Y is the target
X = gameData[['Genre', 'Platform', 'Publisher', 'Year']].copy()
y = np.log1p(gameData['Global_Sales']) # Reducing the range of the answer, better results derived from lesser range
X = X.dropna() # Cleaning data (Nan)
y = y[X.index] # Equalizing X and Y

# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # Train x Test ratio
    random_state=42 
)

def encode_features(X):
    X = pd.DataFrame(X, columns=['Genre', 'Platform', 'Publisher', 'Year'])
    return X

pipeline = Pipeline([
    ('encoder', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )),
    ('to_df', FunctionTransformer(
        lambda x: pd.DataFrame(x, columns=['Genre', 'Platform', 'Publisher', 'Year'])
    )),
    ('model', LGBMRegressor(n_estimators=100, random_state=42, verbose=-1))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_real = np.expm1(y_pred)
y_test_real = np.expm1(y_test)

# ============================================================
# Returning range to original aspect
print("Test Split Results")
print("MAE:", mean_absolute_error(y_test_real, y_pred_real))
print("RMSE:", np.sqrt(mean_squared_error(y_test_real, y_pred_real)).round(3))
print("R²:", r2_score(y_test, y_pred))

# Cross-validation
print("\nCross-Validation Results")
scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print("R² por fold:", scores.round(3))
print("R² médio:", scores.mean().round(3))
print("Desvio padrão:", scores.std().round(3))