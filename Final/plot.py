import numpy as np
import matplotlib.pyplot as plt


train = np.load('./train_accs.npy')
val = np.load('./val_accs.npy')

epochs = np.arange(1, len(train) + 1)

plt.plot(epochs, train, label='Training Set')
plt.plot(epochs, val, label='Validation Set')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
