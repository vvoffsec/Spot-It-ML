import inference
import os
import csv
import warnings
from collections import Counter

warnings.filterwarnings("ignore", module="inference")
warnings.filterwarnings("ignore", module="onnxruntime")

model = inference.get_model("spotitcards/9")

def extract_predictions(resp):
    if isinstance(resp, list):
        if resp and hasattr(resp[0], "predictions"):
            return resp[0].predictions
        return resp
    return resp.predictions

resp  = model.infer(image="dataset/1.jpg")
preds = extract_predictions(resp)
names = [p.class_name for p in preds]
print(names)
counts = Counter(names)
print(counts)
# dups = [name for name, cnt in counts.items() if cnt > 1]