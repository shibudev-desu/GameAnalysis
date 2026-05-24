import pandas as pd
import matplotlib.pyplot as plt

try: 
  gameData = pd.read_csv("games.csv")
  print("Data arrived.")
except:
  print("Error reading data!")

print(gameData.shape)

# Cleaning Empty Data
gameData.dropna(subset=['Year', 'Publisher'])
gameData['Year'] = gameData['Year'].astype('Int32')
gameData = gameData[gameData['Year'] <= 2024]
print(gameData.shape)

""" Questions """

print("Top 10 Games based on sales")
top10 = gameData.nlargest(10, 'Global_Sales')[['Name','Platform','Year','Global_Sales']]
print(top10)

print("Most Sucessfull Platforms based on Sales")
plat = gameData.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False)
print(plat.head(10))

print("Most sucessfull by genre globally")
gnrGBL = gameData.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
print(gnrGBL)

print("Most sucessfull by genre in EU, NA and JP (% Style)")
gnr = gameData.groupby('Genre')[['EU_Sales','NA_Sales','JP_Sales']].sum()
print(gnr)

print("In percentage, Regional")
region = gameData[['EU_Sales', 'JP_Sales', 'NA_Sales']].sum()
regionsby = (region/region.sum()*100).round(1)
print(regionsby)

print("Years where the highest amount of games where released")
highest = gameData.groupby('Year')['Rank'].count().sort_values(ascending=False)
print(highest)