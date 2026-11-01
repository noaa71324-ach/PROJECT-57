import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Algorithmic Approximation (Without hardware constraints) - Unquantized PWL
class UnquantizedPWLSigmoid(nn.Module):
    def forward(self, x):
        # Absolute value due to function symmetry
        abs_x = torch.abs(x)
        y = torch.zeros_like(abs_x)

        # Segment division according to the paper
        mask0 = abs_x < 0.6875
        mask1 = (abs_x >= 0.6875) & (abs_x < 1.5)
        mask2 = (abs_x >= 1.5)    & (abs_x < 2.25)
        mask3 = (abs_x >= 2.25)   & (abs_x < 3.0)
        mask4 = (abs_x >= 3.0)    & (abs_x < 4.0)
        mask5 = (abs_x >= 4.0)    & (abs_x < 5.0)
        mask6 = abs_x >= 5.0

        # Exact calculation of linear equations 
        # to test accuracy loss from transitioning to straight lines alone
        y[mask0] = 0.25 * abs_x[mask0] + 0.5
        y[mask1] = 0.1875 * abs_x[mask1] + 0.5390625
        y[mask2] = 0.125 * abs_x[mask2] + 0.625
        y[mask3] = 0.0625 * abs_x[mask3] + 0.765625
        y[mask4] = 0.03125 * abs_x[mask4] + 0.859375
        y[mask5] = 0.015625 * abs_x[mask5] + 0.921875
        y[mask6] = 1.0

        # Correction if original input was negative
        return torch.where(x < 0, 1.0 - y, y)

# 2. Hardware Simulation (Bit Accurate)
class BitAccurateSigmoid(nn.Module):
    def forward(self, x):
        # Multiply by 16 to transition to Q3.4
        x_scaled = torch.round(x * 16.0)
        
        # Truncate to 8 bits
        x_int = torch.clamp(x_scaled, -127, 127).to(torch.int32)

        # Extract sign bit and absolute value
        sign = (x_int < 0).to(torch.int32)
        abs_x = torch.abs(x_int)

        # Interval decoder as in the paper
        idx = torch.zeros_like(abs_x)
        idx = torch.where(abs_x >= 11, 1, idx)
        idx = torch.where(abs_x >= 24, 2, idx)
        idx = torch.where(abs_x >= 36, 3, idx)
        idx = torch.where(abs_x >= 48, 4, idx)
        idx = torch.where(abs_x >= 64, 5, idx)
        idx = torch.where(abs_x >= 80, 6, idx)

        intercept = torch.zeros_like(abs_x)
        shift_add = torch.zeros_like(abs_x)

        masks = [idx == i for i in range(7)]

        intercept[masks[0]] = 64 << 3
        shift_add[masks[0]] = (abs_x[masks[0]] << 6) >> 2

        intercept[masks[1]] = 69 << 3
        shift_add[masks[1]] = ((abs_x[masks[1]] << 6) >> 3) + ((abs_x[masks[1]] << 6) >> 4)

        intercept[masks[2]] = 80 << 3
        shift_add[masks[2]] = (abs_x[masks[2]] << 6) >> 3

        intercept[masks[3]] = 98 << 3
        shift_add[masks[3]] = (abs_x[masks[3]] << 6) >> 4

        intercept[masks[4]] = 110 << 3
        shift_add[masks[4]] = (abs_x[masks[4]] << 6) >> 5

        intercept[masks[5]] = 118 << 3
        shift_add[masks[5]] = (abs_x[masks[5]] << 6) >> 6

        intercept[masks[6]] = 128 << 3
        shift_add[masks[6]] = 0

        # Add slope and intercept
        mx_plus_c = shift_add + intercept
        
        # Truncate the 3 bottom bits to return to Q1.7 format 
        sigma_abs = mx_plus_c >> 3
        
        # Handle negative numbers
        out_int = torch.where(sign == 1, 128 - sigma_abs, sigma_abs)

        # Return to FLOAT
        return out_int.to(torch.float32) / 128.0

# 3. Test Network (Fashion MNIST)
class FashionMLP(nn.Module):
    def _init_(self):
        super(FashionMLP, self)._init_()
        self.flatten = nn.Flatten()
        
        # Small network with one hidden layer just to test our accelerator works
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Sigmoid(),  # This layer will be dynamically replaced later in the code
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)

def evaluate_model(model, test_loader, description):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # Running tests
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy with {description}: {accuracy:.2f}%")
    return accuracy

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = FashionMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training baseline model with Ideal Sigmoid (1 Epoch for quick test)...")
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("\nRunning Hardware-Software Co Design Evaluation")

    accuracies = []
    labels_graph = ["Ideal\n(PyTorch)", "Unquantized\n(Algorithm)", "Bit-Accurate\n(Hardware)"]

    # Test 1: Running the standard PyTorch baseline model
    acc1 = evaluate_model(model, test_loader, "Ideal Sigmoid")
    accuracies.append(acc1)

    # Test 2: Overriding the activation function with theoretical approximation
    model.classifier[1] = UnquantizedPWLSigmoid()
    acc2 = evaluate_model(model, test_loader, "Unquantized PWL")
    accuracies.append(acc2)

    # Test 3: Testing actual hardware performance with the same weights
    model.classifier[1] = BitAccurateSigmoid()
    acc3 = evaluate_model(model, test_loader, "Bit-Accurate PWL")
    accuracies.append(acc3)

    # 4. Display graph with results
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels_graph, accuracies, color=['#FFD1DC', '#AEC6CF', '#B19CD9'])

    plt.ylim(0, 100)

    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Fashion-MNIST Accuracy: Hardware vs Ideal Sigmoid', fontsize=14, fontweight='bold')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    print("\nGraph saved successfully as 'accuracy_comparison.png'")

if _name_ == '_main_':
    main()