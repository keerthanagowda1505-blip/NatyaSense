import re, torch, numpy as np
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

root = Path("data/raw/padabheda/kalakshetra")
classes = ["Anchitha", "Hayatham", "Kunchita", "Samapada", "Ekapada"]

m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
m.fc = torch.nn.Identity()
m.eval()

tf = transforms.Compose([
    transforms.Lambda(lambda im: im.crop((0, int(im.height*0.3), im.width, im.height))),
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def num(path):
    return int(re.findall(r"\d+", path.stem)[-1])

def embed(img):
    with torch.no_grad():
        return m(tf(img)[None])[0].numpy()

Xtr, ytr, Xte, yte = [], [], [], []
for cls in classes:
    for f in (root/cls).glob("*"):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = Image.open(f).convert("RGB")
        if num(f) <= 16:
            Xtr.append(embed(img)); ytr.append(cls)
            Xtr.append(embed(F.hflip(img))); ytr.append(cls)
            Xtr.append(embed(F.adjust_brightness(img, 1.25))); ytr.append(cls)
            Xtr.append(embed(F.adjust_brightness(img, 0.75))); ytr.append(cls)
        else:
            Xte.append(embed(img)); yte.append(cls)

Xtr, Xte = np.array(Xtr), np.array(Xte)
ytr, yte = np.array(ytr), np.array(yte)
print("Train photos:", len(ytr), "  Test photos:", len(yte))

clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
pred = clf.predict(Xte)
print("Accuracy: {:.2f}%".format((pred == yte).mean() * 100))
print(classification_report(yte, pred))
print("Confusion matrix (rows = true, cols = predicted)")
print("Labels:", classes)
print(confusion_matrix(yte, pred, labels=classes))

import joblib
Path("models").mkdir(exist_ok=True)
joblib.dump({"clf": clf, "classes": list(clf.classes_)}, "models/pada_model.joblib")
print("Saved models/pada_model.joblib")