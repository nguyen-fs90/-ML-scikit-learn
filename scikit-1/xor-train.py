from sklearn import svm

# Tập dữ liệu
xor_data = [
    # P, Q, Result
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# Phân loại data & label
data = []
label = []
for row in xor_data:
    p = row[0]
    q = row[1]
    r = row[2]
    data.append([p, q])
    label.append(r)

# Training 
clf = svm.SVC()
clf.fit(data, label)

# Dự đoán
predict = clf.predict(data)
print("result :", predict)

# Report
ok = 0; total = 0
for idx, answer in enumerate(label):
    p = predict[idx]
    if p == answer: ok += 1
    total += 1
print("Correct Rate :", ok, "/", total, "=", ok/total)
