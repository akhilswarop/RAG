import torch

file_path = "onet_title_embeddings.pkl"

try:
    data = torch.load(file_path, map_location="cpu")
    if isinstance(data, dict):
        print("Keys:", data.keys())
    elif hasattr(data, "shape"):
        print("Shape:", data.shape)
    else:
        print("Data type:", type(data))
except Exception as e:
    print("Error loading file:", e)
