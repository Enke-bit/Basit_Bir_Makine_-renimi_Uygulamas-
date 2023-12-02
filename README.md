# Basit_Bir_Makine_-renimi_Uygulamas-

# 1.Adım-TensorFlow Kurulumu:

pip install tensorflow

# 2. Adım basit bir örnek:

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Örnek veri seti oluştur
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sinir ağı modeli oluştur
model = Sequential()
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modeli derle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Modelin performansını değerlendir
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
