import re

# Чтение данных из файла
with open('50.txt', 'r') as file:
    data = file.read()

# Регулярное выражение для поиска строк с потерями
pattern = r'Train Loss EPOCH \d+: (\d+\.\d+)\nValid Loss EPOCH \d+: (\d+\.\d+)'
matches = re.findall(pattern, data)

# Инициализация списков для хранения train и valid loss
train_loss = []
valid_loss = []

# Заполнение списков значениями потерь
for match in matches:
    train_loss.append(float(match[0]))
    valid_loss.append(float(match[1]))

import matplotlib.pyplot as plt

x=[]

for i in range(len(train_loss)):
    x.append(i)
# Примерные массивы данных


# Создание фигуры и оси
fig, ax = plt.subplots()

# Рисование первого графика (train)
ax.plot(x, train_loss, label='Train')

# Рисование второго графика (valid)
ax.plot(x, valid_loss, label='Valid')

# Настройка легенды
ax.legend()

# Добавление заголовка и подписей осей
plt.title('Train and Valid Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
