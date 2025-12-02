import torch
import torch.nn as nn
from torch import device
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from backbone import new_backbone
import torch.optim as optim
from dataset import ModifiedCIFAR100
# TODO save your best model and store it at './models/d1.pth'

def prepare_test():
    # TODO: Create an instance of your model here. Load the pre-trained weights and return your model.
    #  Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 100), where B is the batch size
    #  and 100 is the number of classes. The output of your model must be the prediction of your classifier,
    #  providing a score for each class, for each image in input

    backbone = new_backbone()
    model = ClassifierLayer(backbone, num_classes=100)

    # do not edit from here downwards
    weights_path = 'models/d1.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model

class ClassifierLayer(nn.Module):
    def __init__(self, backbone, input_dim=576, hidden_num=256, num_classes=100, dropout=0.5):
        super(ClassifierLayer, self).__init__()
        #
        # self.layers = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(input_dim, hidden_num),
        #     nn.BatchNorm1d(hidden_num),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_num, num_classes),
        # )
        self.backbone = backbone
        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, num_classes)
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.layers(x)

        # x = self.fc(x)
        return x

def calculate_top_k_accuracy(outputs, labels, k=5):
    _, predicted = torch.topk(outputs, k, dim=1)
    correct = predicted.eq(labels.view(-1, 1).expand_as(predicted))
    top_k_accuracy = correct.sum().item() / labels.size(0)
    return top_k_accuracy * 100

def test(name='d1'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = new_backbone()
    model = ClassifierLayer(backbone, num_classes=100).to(device)

    model.load_state_dict(torch.load(f'models/{name}.pth', map_location=device))
    model.eval()

    test_dataset = ModifiedCIFAR100(train=False, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0
    total = 0
    top_5_sum = 0

    for batch_idx, batch in enumerate(test_loader):
        images = batch['image'].to(device)
        labels = batch['fine_label'].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        top_5_sum += calculate_top_k_accuracy(outputs, labels, k=5)

    top_5_accuracy = top_5_sum / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test accuracy: {accuracy:.2f}% Top-5 accuracy: {top_5_accuracy:.2f}%')

    return {
        'test_accuracy': accuracy,
        'test_top_5_accuracy': top_5_accuracy
    }

def plot_metrics(historical_loss, historical_val_loss, historical_accuracy, historical_val_accuracy, historical_top_five):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(historical_loss, label='Training Loss')
    plt.plot(historical_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historical_accuracy, label='Training Accuracy')
    plt.plot(historical_val_accuracy, label='Validation Accuracy')
    plt.plot(historical_top_five, label='Validation Top-5 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()

def main(name='d1'):
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ModifiedCIFAR100(train=True, augment=True)

    dataset_size = len(train_dataset)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    backbone = new_backbone()
    model = ClassifierLayer(backbone, num_classes=100).to(device)

    unfreeze_epoch = 5

    for param in backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.AdamW(model.layers.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')

    historical_loss = []
    historical_accuracy = []
    historical_val_loss = []
    historical_val_accuracy = []
    historical_top_five = []

    for epoch in range(20):

        if epoch == unfreeze_epoch:
            print(f"Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Lower lr for backbone
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 5e-4},
                {'params': model.layers.parameters(), 'lr': 2e-4, 'weight_decay': 5e-4}
            ])

        model.train()
        total_loss = 0
        total_val_loss = 0

        training_total = 0
        training_correct = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['fine_label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            training_total += labels.size(0)
            training_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_total = 0
        val_correct = 0
        total_top_five = 0

        for batch_idx, batch in enumerate(val_loader):

            images = batch['image'].to(device)
            labels = batch['fine_label'].to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)

                top_five_accuracy = calculate_top_k_accuracy(outputs, labels, k=5)

            total_top_five += top_five_accuracy
            total_val_loss += loss.item()

        loss_avg = total_loss / len(train_loader)
        val_loss_avg = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        accuracy = 100 * training_correct / training_total
        top_five_accuracy = total_top_five / len(val_loader)

        historical_loss.append(loss_avg)
        historical_val_loss.append(val_loss_avg)
        historical_val_accuracy.append(val_accuracy)
        historical_accuracy.append(accuracy)
        historical_top_five.append(top_five_accuracy)

        print(f"Epoch [{epoch}] Loss: {loss_avg:.4f} Training Acc: {accuracy:.2f}% Val Loss: {val_loss_avg:.4f} Val Acc: {val_accuracy:.2f}% Top-5 Acc: {top_five_accuracy:.2f}%")

        plot_metrics(historical_loss, historical_val_loss, historical_accuracy, historical_val_accuracy, historical_top_five)

        plt.show()

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            no_improve = 0
            torch.save(model.state_dict(), f'models/{name}.pth')
            print("Saving model...")
        else:
            no_improve += 1

        if no_improve >= 15:
            print("Early stopping...")
            break

        scheduler.step(val_loss_avg)
    return {
        'loss': historical_loss,
        'val_loss': historical_val_loss,
        'accuracy': historical_accuracy,
        'val_accuracy': historical_val_accuracy,
        'top_five': historical_top_five
    }

def save_logs_and_plots(logs, run_name='d1'):
    df = pd.DataFrame(logs)
    df.to_csv(f'logs/{run_name}.csv', index=False)

    plot_metrics(logs['loss'], logs['val_loss'], logs['accuracy'], logs['val_accuracy'], logs['top_five'])
    plt.savefig(f'logs/{run_name}.png')


if __name__ == '__main__':
    logs = main(name='d1_20_augment_freeze_0')
    test_logs = test(name='d1_20_augment_freeze_0')
    logs = {**logs, **test_logs, 'mlp_classifer': False}

    save_logs_and_plots(logs, run_name='d1_20_augment_freeze_0')

