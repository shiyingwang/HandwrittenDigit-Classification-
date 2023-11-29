import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CustomDataset
from model import CNN


class Trainer:
    def __init__(self, batch_size, learning_rate, num_epochs, dataset_root):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset_root = dataset_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomDataset(root_dir=self.dataset_root, transform=transform)

        # Split dataset into train, validation, and test sets
        num_data = len(dataset)
        num_train = int(0.8 * num_data)
        num_val = (num_data - num_train) // 2
        num_test = num_data - num_train - num_val

        train_set, val_set, test_set = random_split(
            dataset, [num_train, num_val, num_test]
        )

        self.train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size)

    def initialize_model(self):
        self.model = CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        best_accuracy = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Validation
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Epoch {epoch+1}/{self.num_epochs}, Validation Accuracy: {accuracy}")

            # Save the model with the best validation accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), "best_model.pth")

    def test(self):
        self.model.load_state_dict(torch.load("best_model.pth"))
        self.model.eval()

        # Testing loop
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Initialize Trainer
    trainer = Trainer(batch_size, learning_rate, num_epochs, "mnist")

    # Load dataset
    trainer.load_dataset()

    # Initialize model, loss, and optimizer
    trainer.initialize_model()

    # Train the model
    trainer.train()

    # Test the model
    trainer.test()
