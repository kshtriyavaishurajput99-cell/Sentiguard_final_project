import matplotlib.pyplot as plt

generation = [1, 2, 3, 4, 5]
accuracy = [0.851, 0.867, 0.878, 0.889, 0.885]
features = [4500, 4300, 4150, 3998, 4100]

plt.figure(figsize=(7,4))
plt.plot(generation, accuracy, marker='o', color='blue', linewidth=2)
plt.title('GA - Generation vs Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('static/ga_accuracy.png')
plt.close()

plt.figure(figsize=(7,4))
plt.bar(generation, features, color='green')
plt.title('GA - Generation vs Features Selected')
plt.xlabel('Generation')
plt.ylabel('Features Selected')
plt.savefig('static/ga_features.png')
plt.close()

print("Ho gaya!")