import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from keras.models import Sequential
from keras.layers import Dense

# สร้างข้อมูล
X, y = make_blobs(n_samples=200, centers=[[2, 2], [3, 3]], cluster_std=0.75, random_state=42)

model = Sequential([Dense(1, input_dim=2, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=300, verbose=0)

# คำนวณ 
xx, yy = np.meshgrid(np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.1),np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot กราฟ
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.5)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.show()
