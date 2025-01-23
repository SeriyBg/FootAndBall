import torch
import pickle

if __name__ == '__main__':
    checkpoint = torch.load("models/model_20250122_1323_checkpoint.pth", weights_only=False)
    print(checkpoint["epoch"])
    print(checkpoint["loss"])
    checkpoint = torch.load("models/model_20250122_1449_checkpoint.pth", weights_only=False)
    print(checkpoint["epoch"])
    print(checkpoint["loss"])

    ts = pickle.load(open("training_stats_model_20250121_1418.pickle", "rb"))
    print(ts)
    ts = pickle.load(open("training_stats_model_20250121_1848.pickle", "rb"))
    print(ts)
    ts = pickle.load(open("training_stats_model_20250122_2029.pickle", "rb"))
    print(ts)