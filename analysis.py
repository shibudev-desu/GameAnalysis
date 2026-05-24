# ============================================================
# Game Sales Analysis
# Dataset: Video Game Sales (Kaggle)
# Download: https://www.kaggle.com/datasets/gregorut/videogamesales
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

try: 
  gameData = pd.read_csv("games.csv")
  print("Data arrived.")
except:
  print("Error reading data!")

print(gameData.shape)

# Cleaning Empty Data, making sure Year and Publisher both exist. Aswell as making sure the years are correctly informed.
gameData = gameData.dropna(subset=['Year', 'Publisher'])
gameData['Year'] = gameData['Year'].astype('Int32')
gameData = gameData[gameData['Year'] <= 2024]
print(gameData.shape)

# Renaming publishers to be more compact.
gameData['Publisher'] = gameData['Publisher'].replace({
  'Sony Computer Entertainment': 'Sony',
  'Take-Two Interactive': 'Take Two',
  'Konami Digital Entertainment': 'Konami',
  'Namco Bandai Games': 'Bandai',
  'Microsoft Game Studios': 'Microsoft',
  'Warner Bros. Interactive Entertainment': 'Warner Bros.'
}) 

# Questions being answerd through queries
# Global searches
top10 = gameData.nlargest(10, 'Global_Sales')[['Name','Platform','Year','Global_Sales']] # Top 10 Games on the market
platforms = gameData.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False) # Best platforms globally
publishers = gameData.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False) # Publishers with most Sales
gnrGBL = gameData.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False) # Genres most sold
highest = gameData.groupby('Year')['Rank'].count().sort_values(ascending=False) # Years sorted by launches

# Regional
gnr = gameData.groupby('Genre')[['EU_Sales','NA_Sales','JP_Sales']].sum() # Genre sales in each regions 
region = gameData[['EU_Sales', 'JP_Sales', 'NA_Sales']].sum()
regionsby = (region/region.sum()*100).round(1) # Sales sorted in percentage for each region


if input("Show all searches in full?") == 'Yes':
  print("Top 10 Games based on sales\n", top10)
  print("Most Sucessfull Platforms based on Sales\n", platforms.head(10))
  print("Most sucessfull by genre globally\n", gnrGBL)
  print("Most sucessfull by genre in EU, NA and JP (% Style)\n", gnr)
  print("In percentage, Regional\n", regionsby)
  print("Years where the highest amount of games where released\n", highest)

# ============================================================
#   Charts
#   Each chart is done with a certain search
#   Each one is made to be easily readable
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Video Game Sales Analysis', fontsize=16, fontweight='bold')
 
# --- Chart 1: Sales by genre ---
ax1 = axes[0, 0]
gnrGBL.plot(kind='bar', ax=ax1, color='red', edgecolor='white')
ax1.set_title('Global Sales by Genre (millions)')
ax1.set_xlabel('Genre')
ax1.set_ylabel('Sales (M)')
ax1.tick_params(axis='x', rotation=45)

# --- Chart 2: Most Sucessfull publishers

ax2 = axes[0,1]
publishers.head(15).plot(kind="bar",ax=ax2, color="blue")
ax2.set_title('Sales sorted by publishers')
ax2.set_xlabel("Platforms")
ax2.set_ylabel('Sales (M)')
ax2.tick_params(axis='x', rotation=45)

# --- Chart 3: Regional Sales

ax3 = axes[1,0]
regionsby.head(15).plot(kind="pie",ax=ax3, color="blue", autopct='%1.1f%%', startangle=90, labels=['Europe', 'Japan', 'North America'])
ax3.set_title('Sales sorted by regions')

# --- Chart 4: Years with most launches

ax4 = axes[1,1]
highest.sort_index().plot(kind='line',ax=ax4, marker='o', markersize=3)
ax4.set_title("Years with most launches")

plt.tight_layout()
plt.savefig('charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as charts.png")