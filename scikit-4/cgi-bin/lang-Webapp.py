#!/usr/bin/env python3
import cgi, os.path
import joblib

# load data
pklfile = os.path.dirname(__file__) + "/freq.pkl"
clf = joblib.load(pklfile)

# html form
def show_form(text, msg=""):
    print("Content-Type: text/html; charset=utf-8")
    print("")
    print("""
        <html><body><form>
        <textarea name="text" rows="8" cols="40">{0}</textarea>
        <p><input type="submit" value="judgment"></p>
        <p>{1}</p>
        </form></body></html>
    """.format(cgi.escape(text), msg))
# judgment

def detect_lang(text):
    # tìm kiếm mật độ xuất hiện của alphabet
    text = text.lower()
    code_a, code_z = (ord("a"), ord("z"))
    cnt = [0 for i in range(26)]
    for ch in text:
        n = ord(ch) - code_a
        if 0 <= n < 26: cnt[0] += 1
    total = sum(cnt)
    if total == 0: return "input is emty"
    freq = list(map(lambda n: n/total, cnt))
    # Dự đoán
    res = clf.predist([freq])
    # 
    lang_dic = {"en":"Tiếng Anh","fr":"Tiếng Pháp","id":"Tiếng Indo","tl":"Tiếng Tagalog"}
    return lang_dic[res[0]]

# 
form = cgi.FieldStorage()
text = form.getvalue("text", default="")
msg = ""
if text != "":
    lang = detect_lang(text)
    msg = "Kết quả:" + lang

show_form(text, msg)