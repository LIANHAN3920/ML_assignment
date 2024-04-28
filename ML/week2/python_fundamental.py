import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
fig, ax = plt.subplots()

ax.scatter(df['Age'],df['Fare'])

plt.show()