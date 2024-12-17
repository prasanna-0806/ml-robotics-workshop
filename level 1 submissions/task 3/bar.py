import matplotlib.pyplot as plt
ids = [1, 2, 3, 4, 5]
heights = [170, 160, 180, 150, 165]
weights = [65, 55, 75, 50, 60]
plt.figure(figsize=(8, 6))
plt.bar([i - 0.2 for i in ids], heights, width=0.4, color='skyblue', label='Height (cm)')
plt.bar([i + 0.2 for i in ids], weights, width=0.4, color='lightgreen', label='Weight (kg)')
plt.xlabel('ID')
plt.ylabel('Values')
plt.title('Height and Weight of Individuals')
plt.xticks(ids)  
plt.legend()
plt.show()
