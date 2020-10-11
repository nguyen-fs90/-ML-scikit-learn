from sklearn import svm, metrics

# Load data từ file csv
def load_csv(fname):
    labels = []
    images = []
    with open(fname, "r") as f:
        for line in f:
            cols = line.split(",")
            if len(cols) < 2: continue
            labels.append(int(cols.pop(0)))
            vals = list(map(lambda n: int(n) / 256, cols))
            images.append(vals)
    return {"labels": labels, "images": images}
data = load_csv("./mnist/train.csv")
test = load_csv("./mnist/t10k.csv")

# Training
clf = svm.SVC()
clf.fit(data["images"], data["labels"])

# Dự đoán
predict = clf.predict(test["images"])

# Rate & Report
ac_core = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("Correct Rate =", ac_core)
print("Report =")
print(cl_report)
