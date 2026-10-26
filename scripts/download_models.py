import yaml
import urllib.request

cfg = yaml.safe_load(open("config.yaml"))["mediapipe"]

for url, path in [(cfg["pose_url"], cfg["pose_model"]),
                  (cfg["hand_url"], cfg["hand_model"])]:
    print("Downloading", path)
    urllib.request.urlretrieve(url, path)

print("Done")