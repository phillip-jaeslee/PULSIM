
import numpy as np
import matplotlib.pyplot as plt

# 1000개의 데이터 생성 (예: 사인파 데이터)
x_1000 = np.linspace(0, 10, 1000)  # x 좌표 (1000개의 데이터)
y_1000 = np.sin(x_1000)            # y 좌표 (1000개의 데이터)

# 600개의 새로운 x 좌표 생성
x_600 = np.linspace(0, 10, 600)    # 600개의 x 데이터
'''
# y 값 보간 (interpolation)으로 600포인트로 변환
y_600 = np.interp(x_600, x_1000, y_1000)

# 결과 시각화
plt.figure(figsize=(10, 5))

# 원본 그래프 (1000포인트)
plt.plot(x_1000, y_1000, label="1000 points", alpha=0.5)

# 축소된 그래프 (600포인트)
plt.plot(x_600, y_600, label="600 points", linestyle="--")

plt.legend()
plt.title("Resampled Graph from 1000 to 600 Points")
plt.show()
'''

from scipy.interpolate import CubicSpline

# Cubic Spline 보간기 생성
cs = CubicSpline(x_1000, y_1000)

# 600포인트에 대해 Cubic Spline 보간 적용
y_600_cs = cs(x_600)

# 결과 시각화
plt.figure(figsize=(10, 5))

# 원본 그래프 (1000포인트)
plt.plot(x_1000, y_1000, label="1000 points", alpha=0.5)

# 축소된 그래프 (Cubic Spline 600포인트)
plt.plot(x_600, y_600_cs, label="Cubic Spline 600 points", linestyle="--")

plt.legend()
plt.title("Cubic Spline Resampled Graph")
plt.show()