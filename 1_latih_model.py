import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

print("Sedang melatih model AI...")

# Data latih: [IPK, Total SKS]
X_train = np.array([
    [3.5, 144],
    [2.0, 100],
    [3.2, 140],
    [2.5, 110],
    [3.8, 150],

    # Data tambahan mahasiswa
    [1.0, 40],
    [1.2, 60],
    [1.5, 80],
    [1.8, 90],
    [2.0, 100],
    [2.2, 110],
    [2.8, 130],
    [3.0, 140],
    [3.5, 150],
    [3.9, 160]
])

# Label: 1 = Lulus, 0 = Tidak Lulus
y_train = np.array([
    1, 0, 1, 0, 1,
    0, 0, 0, 0, 0,
    0, 1, 1, 1, 1
])

# Membuat model Decision Tree
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Melatih model
model.fit(X_train, y_train)

# Menyimpan model
joblib.dump(model, 'model_kelulusan.pkl')

print("Sukses! Model berhasil disimpan.")