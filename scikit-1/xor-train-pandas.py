import pandas as pd
from sklearn import svm, metrics

# Dữ liệu 
xor_input = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 1, 0]
]
# Phân loại data & label
xor_df = pd.DataFrame(xor_input)
xor_data = xor_df.loc[:,0:1]
xor_label = xor_df.loc[:,2]

# Training
clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)

# Score
ac_score = metrics.accuracy_score(xor_label, pre)
print("Correct Rate = ", ac_score)
