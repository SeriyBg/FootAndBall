import torch
import pickle

if __name__ == '__main__':
    # checkpoint = torch.load("models/model_20250122_1323_checkpoint.pth", weights_only=False)
    # print(checkpoint["epoch"])
    # print(checkpoint["loss"])
    # checkpoint = torch.load("models/model_20250122_1449_checkpoint.pth", weights_only=False)
    # print(checkpoint["epoch"])
    # print(checkpoint["loss"])

    checkpoint = torch.load("/Users/sergebishyr/PhD/models/no_changes_20e/model_20250123_1816_checkpoint.pth", weights_only=False, map_location=torch.device('cpu'))
    print(checkpoint["epoch"])
    print(checkpoint["loss"])

    ts = pickle.load(open("/Users/sergebishyr/PhD/models/no_changes_20e/training_stats_model_20250123_1816.pickle", "rb"))
    print(ts)
    ts = pickle.load(open("training_stats_model_20250121_1848.pickle", "rb"))
    print(ts)
    # ts = pickle.load(open("training_stats_model_20250122_2029.pickle", "rb"))
    # print(ts)