import pickle
import sys

# === Change this to your actual path ===
pkl_path = r"C:\Users\lenovo\Desktop\loma_loma\LOMA_LOMA\app\model\sk_model.pkl"  # or sys.argv[1] if you want CLI

with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

print("\n🔍 Type of loaded object:", type(obj))

if isinstance(obj, dict):
    print("\n📦 Keys in the dict:")
    for key in obj:
        print(" -", key)

    if "model" in obj:
        print("\n✅ Model type:", type(obj["model"]))
    if "labels" in obj:
        print("\n🏷️ Label mapping:", obj["labels"])

else:
    print("\n👀 Looks like it contains just a model:", type(obj))
