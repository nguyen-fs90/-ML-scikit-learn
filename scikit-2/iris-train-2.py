import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# Load file iris.csv
csv = pd.read_csv('iris.csv')
# print(csv)
# Lấy ra một hàng tùy ý
csv_data = csv[["sepal.length","sepal.width","petal.length","petal.width"]]
csv_label = csv["variety"]

# Phân loại ra thành dữ liệu dùng để train và dữ liệu dùng để test
train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

# Training data, dự đoán kết quả
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# Score
ac_score = metrics.accuracy_score(test_label, pre)
print("Correct Rate =", ac_score)
