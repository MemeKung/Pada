from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0, 3.0),
                    cluster_std=0.25,
                    random_state=69)

X2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0, 2.0),
                    cluster_std=0.25,
                    random_state=69)

# รวมข้อมูลเข้าด้วยกัน
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

# คำนวณ centroid ของแต่ละกลุ่ม
centroid1 = np.mean(X1, axis=0)
centroid2 = np.mean(X2, axis=0)

# ฟังก์ชัน Decision Boundary
def decision_function(x1, x2):
    # ใช้เส้นแบ่งที่ตั้งฉากกับเส้นเชื่อม centroid ทั้งสอง
    slope = -(centroid2[0] - centroid1[0]) / (centroid2[1] - centroid1[1])
    intercept = (centroid1[1] + centroid2[1]) / 2 - slope * (centroid1[0] + centroid2[0]) / 2
    return slope * x1 - x2 + intercept

# สร้าง Grid สำหรับ Decision Plane
x1_range = np.linspace(-4, 5, 500)
x2_range = np.linspace(-4, 5, 500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

g_values = decision_function(x1_grid, x2_grid)


plt.figure(figsize=(8, 6))
# Decision Plane
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0, np.inf],
             colors=['red', 'blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black', linewidths=2)

# Plot Data Points
plt.scatter(X[:len(X1), 0], X[:len(X1), 1], c='purple', edgecolor='k', label='Class 1')
plt.scatter(X[len(X1):, 0], X[len(X1):, 1], c='yellow', edgecolor='k', label='Class 2')

# Plot Centroids
plt.scatter(centroid1[0], centroid1[1], c='purple', marker='x', s=100, label='Centroid 1')
plt.scatter(centroid2[0], centroid2[1], c='yellow', marker='x', s=100, label='Centroid 2')

plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Dynamic Decision Plane with Data Points')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
