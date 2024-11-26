# Kutubxonalarni import qilish
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging

# 1. Log konfiguratsiyasi
logging.basicConfig(filename='patient_clustering.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 2. Bemorlarni qo'lda kiritish
def get_patient_data():
    patients = []
    while True:
        try:
            # Bemorni kiritish
            print("Yangi bemor ma'lumotlarini kiriting:")
            blood_pressure = float(input("Qon bosimi (mmHg): "))
            blood_sugar = float(input("Qand miqdori (mg/dL): "))
            
            # Bemor ma'lumotlarini ro'yxatga qo'shish
            patients.append([blood_pressure, blood_sugar])
            
            # Yana bemor qo'shishni so'rash
            more = input("Yana bemor kiritmoqchimisiz? (ha/yo'q): ").lower()
            if more != 'ha':
                break
        except ValueError:
            print("Xato ma'lumot! Iltimos, raqamli qiymatlarni kiriting.")
    
    return np.array(patients)

# 3. DBSCAN yordamida bemorlarni klasterlash
def dbscan_clustering(X, eps=10, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

# 4. Natijalarni baholash va log faylga yozish
def evaluate_clustering(X, labels):
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_anomalies = np.sum(labels == -1)
    logging.info(f"Jami klasterlar soni: {num_clusters}")
    logging.info(f"Aniqlangan anomaliyalar soni: {num_anomalies}")

    # Klasterlar uchun Silhouette Score hisoblash
    if num_clusters > 1:
        silhouette_avg = silhouette_score(X[labels != -1], labels[labels != -1])
        logging.info(f"Silhouette Score: {silhouette_avg:.2f}")
        print(f"Silhouette Score: {silhouette_avg:.2f}")
    else:
        logging.warning("Klasterlash muvaffaqiyatsiz bo'ldi.")
        print("Klasterlash muvaffaqiyatsiz bo'ldi.")

    return num_clusters, num_anomalies

# 5. Vizualizatsiya funksiyasi
def visualize_results(X, labels):
    anomalies = X[labels == -1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[labels != -1, 0], X[labels != -1, 1], c=labels[labels != -1], cmap='viridis', s=50, label='Normal bemorlar')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', s=100, label='Anomaliyalar')
    plt.title("Bemorlarni klasterlash")
    plt.xlabel("Qon bosimi (mmHg)")
    plt.ylabel("Qand miqdori (mg/dL)")
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. Asosiy dastur oqimi
if __name__ == "__main__":
    # 1. Foydalanuvchidan bemorlarni ma'lumotlarini olish
    X = get_patient_data()

    # 2. Klasterlash (DBSCAN)
    eps_value = 10
    min_samples_value = 5
    labels = dbscan_clustering(X, eps=eps_value, min_samples=min_samples_value)

    # 3. Natijalarni baholash
    num_clusters, num_anomalies = evaluate_clustering(X, labels)

    # 4. Natijalarni konsolda chiqarish
    print(f"Jami klasterlar soni: {num_clusters}")
    print(f"Aniqlangan anomaliyalar soni: {num_anomalies}")

    # 5. Vizualizatsiya
    visualize_results(X, labels)