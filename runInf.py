import inference
import os
import csv
import warnings
from collections import Counter

warnings.filterwarnings("ignore", module="inference")
warnings.filterwarnings("ignore", module="onnxruntime")

model = inference.get_model("spotitcards/8")

def extract_predictions(resp):
    if isinstance(resp, list):
        if resp and hasattr(resp[0], "predictions"):
            return resp[0].predictions
        return resp
    return resp.predictions

def find_duplicate(image_path):
    resp  = model.infer(image=image_path)
    preds = extract_predictions(resp)
    names = [p.class_name for p in preds]
    counts = Counter(names)
    dups = [name for name, cnt in counts.items() if cnt > 1]
    return dups[0] if dups else ""

if __name__ == "__main__":
    dataset_dir = "dataset"
    output_csv  = "submission.csv"

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Class"])

        for root, _, files in os.walk(dataset_dir):
            imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            imgs.sort(key=lambda fn: int(os.path.splitext(fn)[0]))

            # only do the first ten images
            imgs = imgs[:10]

            for fname in imgs:
                image_path = os.path.join(root, fname)
                dup = find_duplicate(image_path)
                writer.writerow([fname, dup])

    print(f"Wrote submission file with first 10 images to {output_csv}")