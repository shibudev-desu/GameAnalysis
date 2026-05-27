# 🎮 GameAnalysis
Exploratory data analysis in 16,000+ games, uncovering trends in sales, genres and publishers

## 📊 Key Findings
- Action is the best-selling genre globally
- 2007–2010 was the peak period for game releases
- Post-2010 saw a sharp decline in new game launches
- North America accounts for ~49% of global sales
- PS2 is the highest-grossing platform of all time in this dataset
- Nintendo leads the global sales market with a significant gap over competitors
- Japan overwhelmingly dominates RPG sales compared to other regions

## 🛠️ Tech Stack
- Python 3.x
- Pandas — data manipulation
- Matplotlib — visualizations

## 📁 Dataset
[Video Game Sales - Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales)  
16,598 games with sales data across NA, EU, JP and other regions.

## 🚀 How to Run
```bash
pip install pandas matplotlib
python analysis.py
```

## 📈 Charts
![Sales Analysis](charts.png)

## 🤖 Machine Learning Model

Built a binary classifier to predict whether a game will sell over 1 million copies globally.

* **Approach:** Random Forest Classifier with 100 estimators 
+ **Features:** Genre, Platform, Publisher, Year 
- **Target:** Global Sales ≥ 1M (binary: 0 or 1) 

| | Default | Balanced Weights | SMOTE |
|---|---|---|---|
| Accuracy | 86% | 83% | 84% |
| Recall (1M+ games) | 31% | 42% | 41% |
| F1 (1M+ games) | 35% | 37% | 38% |

- **Key note:** The dataset used is heavily imbalanced — only ~12% of games sold over 1M copies. 
Using `class_weight='balanced'` reduced overall accuracy but improved the model's 
ability to identify sucessful games, revealing a precision-recall trade-off.

- **Note on SMOTE:** Applying SMOTE before the train/test split 
inflated results to 89% (data leakage). Correct application — 
SMOTE only on training data — yielded an honest 84%, with improved 
recall on hit games compared to the baseline.