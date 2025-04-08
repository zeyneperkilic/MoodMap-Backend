import requests
import pandas as pd


API_URL = "http://localhost:8000"


local_data = pd.read_csv("processed_data.csv")

results = []

for cluster_id in range(4):
    for intensity in [1, 5, 10]:
        response = requests.get(f"{API_URL}/recommend/{cluster_id}?intensity={intensity}")
        if response.status_code == 200:
            songs = response.json()["songs"]
            for i, song in enumerate(songs, start=1):
                uri = song["uri"]
                row = local_data[local_data["uri"] == uri]
                if not row.empty:
                    energy = row.iloc[0]["energy"]
                    valence = row.iloc[0]["valence"]
                    results.append({
                        "cluster_id": cluster_id,
                        "intensity": intensity,
                        "rank": i,
                        "uri": uri,
                        "energy": energy,
                        "valence": valence
                    })


results_df = pd.DataFrame(results)
results_df.to_csv("recommendation_accuracy_report.csv", index=False)
print("✅ Created: recommendation_accuracy_report.csv")
