import pandas as pd
import matplotlib.pyplot as plt

path = "dataset.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

categorical_cols = ["Country", "Weather", "Time of Day", "Season", "Activity", "Mood/Emotion"]

for col in categorical_cols:
    print(f"\n--- {col} (Top 10) ---")
    print(df[col].value_counts().head(10))

plt.figure()
df["Weather"].value_counts().plot(kind="bar")
plt.title("Weather Distribution")
plt.xlabel("Weather")
plt.ylabel("Count")
plt.show()

plt.figure()
df["Season"].value_counts().plot(kind="bar")
plt.title("Season Distribution")
plt.xlabel("Season")
plt.ylabel("Count")
plt.show()

cross = pd.crosstab(df["Weather"], df["Mood/Emotion"])
print("\nWeather x Mood (first 5 rows):\n")
print(cross.head())

print("\nTop 3 moods per Weather:\n")
for w in df["Weather"].value_counts().index:
    top_moods = df[df["Weather"] == w]["Mood/Emotion"].value_counts().head(3)
    print(f"\nWeather = {w}")
    print(top_moods)
