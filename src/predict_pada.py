import sys, torch, joblib
from PIL import Image
from torchvision import models, transforms

saved = joblib.load("models/pada_model.joblib")
clf = saved["clf"]

m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
m.fc = torch.nn.Identity()
m.eval()

tf = transforms.Compose([
    transforms.Lambda(lambda im: im.crop((0, int(im.height*0.3), im.width, im.height))),
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict_pada(path):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        v = m(tf(img)[None])[0].numpy()
    probs = clf.predict_proba([v])[0]
    i = probs.argmax()
    return clf.classes_[i], float(probs[i])

if __name__ == "__main__":
    name, conf = predict_pada(sys.argv[1])
    print("Pada Bheda:", name, " Confidence: {:.0f}%".format(conf * 100))