import torch
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # checkpoint = torch.load("/Users/sergebishyr/PhD/models/no_changes_20e/model_20250123_1816_checkpoint.pth", weights_only=False, map_location=torch.device('cpu'))
    # print(checkpoint["epoch"])
    # print(checkpoint["loss"])
    # checkpoint = torch.load("/Users/sergebishyr/PhD/models/model_20250125_1750_checkpoint.pth", weights_only=False, map_location=torch.device('cpu'))
    # print(checkpoint["epoch"])
    # print(checkpoint["loss"])
    #
    # ts = pickle.load(open("/Users/sergebishyr/PhD/models/no_changes_20e/training_stats_model_20250123_1816.pickle", "rb"))
    # print(ts)
    # ts = pickle.load(open("/Users/sergebishyr/PhD/models/training_stats_model_20250125_1750.pickle", "rb"))
    # print(ts)

    # Load the pickle file
    file_path = '/Users/sergebishyr/PhD/models/deeep_transp/training_stats_model_20250205_1439.pickle'  # Replace with your actual file path
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract training and validation losses
    train_losses = [entry['loss'] for entry in data['train']]
    val_losses = [entry['loss'] for entry in data['val']]


    file_path = '/Users/sergebishyr/PhD/models/no_changes_20e/training_stats_model_20250123_1816.pickle'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract training and validation losses
    train_losses_f = [entry['loss'] for entry in data['train']]
    val_losses_f = [entry['loss'] for entry in data['val']]

    # Plot the learning curve
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label='Training Loss +1 Transp', marker='o', linestyle='-')
    plt.plot(val_losses, label='Validation Loss +1 Transp', marker='s', linestyle='--')
    plt.plot(train_losses_f, label='Training Loss FPN', marker='^', linestyle='-')
    plt.plot(val_losses_f, label='Validation Loss FPN', marker='d', linestyle='--')

    # Add labels, legend, and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()