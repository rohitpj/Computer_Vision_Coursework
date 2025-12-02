import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from backbone import new_backbone
import torch.optim as optim
from dataset import ModifiedCIFAR100


# TODO save your best model and store it at './models/d3.pth'

def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output two tensors: the first of shape (B, 100), with the second of shape
    #  (B, 20). B is the batch size and 100/20 is the number of fine/coarse classes.
    #  The output is the prediction of your classifier, providing two scores for both fine and coarse classes,
    #  for each image in input

    backbone = new_backbone()
    model = DualClassifier(backbone)

    # do not edit from here downwards
    weights_path = 'models/d3.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model


class DualClassifier(nn.Module):
    def __init__(self, backbone, input_dim=576, fine_classes=100, coarse_classes=20, dropout=0.5):
        super(DualClassifier, self).__init__()

        self.backbone = backbone
        self.fine_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, fine_classes)
        )

        self.coarse_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, coarse_classes)
        )


    def forward(self, x):
        x = self.backbone(x)

        return self.fine_head(x), self.coarse_head(x)


def test(name='d1'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = new_backbone()
    model = DualClassifier(backbone).to(device)

    model.load_state_dict(torch.load(f'models/{name}.pth', map_location=device))
    model.eval()

    test_dataset = ModifiedCIFAR100(train=False, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    coarse_correct = 0
    fine_correct = 0
    exact_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            coarse_labels = batch['coarse_label'].to(device)
            fine_labels = batch['fine_label'].to(device)

            fine_out, coarse_out = model(images)
            _, coarse_predicted = torch.max(coarse_out, dim=1)
            _, fine_predicted = torch.max(fine_out, dim=1)

            total += fine_labels.size(0)
            coarse_correct += (coarse_predicted == coarse_labels).float().sum().item()
            fine_correct += (fine_predicted == fine_labels).float().sum().item()
            exact_correct += ((coarse_predicted == coarse_labels) & (fine_predicted == fine_labels)).float().sum().item()

    coarse_accuracy = 100 * coarse_correct / total
    fine_accuracy = 100 * fine_correct / total
    exact_accuracy = 100 * exact_correct / total

    print(f'Test Coarse Acc: {coarse_accuracy:.2f}% Fine Acc: {fine_accuracy:.2f}% Exact Match Acc: {exact_accuracy:.2f}%')

    return {
        'test_coarse_accuracy': coarse_accuracy,
        'test_fine_accuracy': fine_accuracy,
        'test_exact_accuracy': exact_accuracy
    }



def plot_metrics(historical_loss, historical_val_loss, historical_accuracy, historical_val_accuracy, historical_coarse, historical_fine, historical_exact):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(historical_loss, label='Training Loss')
    plt.plot(historical_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Plot the three requested validation accuracies plus training accuracy
    plt.plot(historical_accuracy, label='Training Fine Acc')
    plt.plot(historical_coarse, label='Val Coarse Acc')
    plt.plot(historical_fine, label='Val Fine Acc')
    plt.plot(historical_exact, label='Val Exact Match Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()


def main(name='d1'):
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ModifiedCIFAR100(train=True, augment=True)

    dataset_size = len(train_dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    backbone = new_backbone()
    model = DualClassifier(backbone).to(device)

    unfreeze_epoch = 5

    # freeze backbone initially
    for param in backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Only optimize the heads initially
    optimizer = optim.AdamW(list(model.fine_head.parameters()) + list(model.coarse_head.parameters()), lr=0.001, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')

    historical_loss = []
    historical_accuracy = []
    historical_val_loss = []
    historical_val_accuracy = []
    historical_coarse = []
    historical_fine = []
    historical_exact = []

    no_improve = 0

    for epoch in range(20):

        if epoch == unfreeze_epoch:
            print(f"Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 5e-4},
                {'params': list(model.fine_head.parameters()) + list(model.coarse_head.parameters()), 'lr': 2e-4, 'weight_decay': 5e-4}
            ])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        model.train()
        total_loss = 0.0

        training_total = 0
        training_correct = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            coarse_labels = batch['coarse_label'].to(device)
            fine_labels = batch['fine_label'].to(device)

            optimizer.zero_grad()
            fine_outputs, coarse_outputs = model(images)

            _, fine_predicted = torch.max(fine_outputs, dim=1)

            training_total += fine_labels.size(0)
            training_correct += (fine_predicted == fine_labels).float().sum().item()

            fine_loss = criterion(fine_outputs, fine_labels)
            coarse_loss = criterion(coarse_outputs, coarse_labels)
            loss = fine_loss + coarse_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        val_total = 0
        coarse_val_correct = 0
        fine_val_correct = 0
        exact_val_correct = 0
        total_val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                coarse_labels = batch['coarse_label'].to(device)
                fine_labels = batch['fine_label'].to(device)

                fine_outputs, coarse_outputs = model(images)

                _, coarse_predicted = torch.max(coarse_outputs, dim=1)
                _, fine_predicted = torch.max(fine_outputs, dim=1)

                b = coarse_labels.size(0)
                val_total += b
                coarse_val_correct += (coarse_predicted == coarse_labels).float().sum().item()
                fine_val_correct += (fine_predicted == fine_labels).float().sum().item()
                # exact match: both coarse and fine correct for the same sample
                exact_batch = ((coarse_predicted == coarse_labels) & (fine_predicted == fine_labels)).float().sum().item()
                exact_val_correct += exact_batch

                total_val_loss += (criterion(coarse_outputs, coarse_labels) + criterion(fine_outputs, fine_labels)).item()

        loss_avg = total_loss / len(train_loader)
        val_loss_avg = total_val_loss / len(val_loader)
        coarse_val_accuracy = 100 * coarse_val_correct / val_total if val_total > 0 else 0.0
        fine_val_accuracy = 100 * fine_val_correct / val_total if val_total > 0 else 0.0
        exact_val_accuracy = 100 * exact_val_correct / val_total if val_total > 0 else 0.0
        # keep val_accuracy as the average of coarse and fine (existing metric)
        val_accuracy = (coarse_val_accuracy + fine_val_accuracy) / 2
        accuracy = 100 * training_correct / training_total

        historical_loss.append(loss_avg)
        historical_val_loss.append(val_loss_avg)
        historical_val_accuracy.append(val_accuracy)
        historical_accuracy.append(accuracy)
        historical_coarse.append(coarse_val_accuracy)
        historical_fine.append(fine_val_accuracy)
        historical_exact.append(exact_val_accuracy)

        print(f"Epoch [{epoch}] Loss: {loss_avg:.4f} Training Fine Acc: {accuracy:.2f}% Val Loss: {val_loss_avg:.4f} Val Coarse: {coarse_val_accuracy:.2f}% Fine: {fine_val_accuracy:.2f}% Exact: {exact_val_accuracy:.2f}%")

        plot_metrics(historical_loss, historical_val_loss, historical_accuracy, historical_val_accuracy, historical_coarse, historical_fine, historical_exact)
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
        'coarse': historical_coarse,
        'fine': historical_fine,
        'exact': historical_exact
    }


def save_logs_and_plots(logs, run_name='d1'):
    df = pd.DataFrame(logs)
    df.to_csv(f'logs/{run_name}.csv', index=False)

    plot_metrics(logs['loss'], logs['val_loss'], logs['accuracy'], logs['val_accuracy'], logs['coarse'], logs['fine'], logs['exact'])
    plt.savefig(f'logs/{run_name}.png')


if __name__ == '__main__':
    logs = main(name='d3_20_augment_freeze_5')
    test_logs = test(name='d3_20_augment_freeze_5')
    logs = {**logs, **test_logs, 'mlp_classifer': False}

    save_logs_and_plots(logs, run_name='d3_20_augment_freeze_5')