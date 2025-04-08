import requests

def test_recommendation(cluster_id, intensity):
    url = f"http://localhost:8000/recommend/{cluster_id}?intensity={intensity}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            songs = data.get('songs', [])
            print(f"✅ Cluster {cluster_id}, Intensity {intensity} - {len(songs)} şarkı önerildi.")
            if len(songs) == 0:
                print("⚠️  Uyarı: Hiç şarkı dönmedi!")
        else:
            print(f"❌ Hata: Status code {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ İstek sırasında hata oluştu: {str(e)}")

def run_tests():
    print("------ Öneri API Testleri Başlıyor ------")
    for cluster_id in range(4):  # Cluster 0, 1, 2, 3
        for intensity in [1, 5, 10]:
            test_recommendation(cluster_id, intensity)
    print("------ Testler Tamamlandı ------")

if __name__ == "__main__":
    run_tests()