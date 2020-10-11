from sklearn import svm
import joblib
import json

# Load data from json file
with open("./lang/freq.json", "r", encoding="utf-8") as fp:
    d = json.load(fp)
    data = d[0]

# Training
clf = svm.SVC()
clf.fit(data["freqs"], data["labels"])

# save data
joblib.dump(clf, "./cgi-bin/freq.pkl")
print("ok")
