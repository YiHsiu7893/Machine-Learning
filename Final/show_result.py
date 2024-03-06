
import os
import numpy as np

previous_val_acc = 'D:/CPA_dataset/resultOnly#1/val_acc.npy'
val_accs = np.load(previous_val_acc)

for i in range(len(val_accs)):
    if(val_accs[i]>75):
        print('{}: {}'.format(i,  val_accs[i]))
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# 載入.npy檔
re1 = np.load('D:/CPA_dataset/result#1/val_acc.npy')
re2 = np.load('D:/CPA_dataset/result#2/val_acc.npy')
re3 = np.load('D:/CPA_dataset/result#3/val_acc.npy')
re4 = np.load('C:/Users/user/Downloads/val_acc.npy')
#re2 = np.load(os.path.join(os.getcwd(), 'val_acc.npy'))

fig, ax = plt.subplots()

# 绘制数据
ax.plot(re1, label='Result #1')
ax.plot(re2, label='Result #2')
ax.plot(re3, label='Result #3')
ax.plot(re4, label='Result #4')

# 添加图例
ax.legend(loc='lower right')

ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Validation Accuracy')

ax.set_ylim(50, 90)

plt.show()

# 產生epoch的數量
epochs = np.arange(1, len(re2) + 1)
#plt.yticks(range(0, 90, 5))

# 繪製折線圖
#plt.plot(epochs, re1, label='result#1')
plt.plot(epochs, re2, label='result#2')
plt.plot(epochs, re3, label='result#3')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""
