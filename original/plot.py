import matplotlib.pyplot as plt
record = [191.4, 149.2, 82.19047619047619, 48.74193548387097, 55.911764705882355, 48.583333333333336, 36.73076923076923, 39.98412698412698, 26.170212765957448, 17.85, 12.020408163265307, 13.0, 13.61038961038961, 14.4, 13.333333333333334]
x = range(15)
plt.plot(x,record)
plt.xlabel('level of environment')
plt.ylabel('accumulative reward')
plt.title('PPO Cart-pole FC Model')
plt.grid()
plt.show()
