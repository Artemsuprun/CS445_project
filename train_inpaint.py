# needed libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from mlp import MLP


# Training the model
def train(model, train, opt, epoch):
    transform = transforms.Compose([
        transforms.RandomErasing(1, (0.1, 0.4))
    ])
    for e in range(epoch):
        for images, targets in train:
            images = transform(images)
            images = images.view(-1, model.input).cuda()
            targets = targets.view(-1, model.input).cuda()

            opt.zero_grad()
            outputs = model(images)
            loss = model.loss(outputs, targets)
            loss.backward()
            opt.step()


def main():
    # Get the MNIST dataset
    train_MNIST = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    train_input = []
    train_target = []
    for img, _ in train_MNIST:
       input_img = img.clone()
       target_img = img.clone()
       train_input.append(input_img)
       train_target.append(target_img)
    train_input = torch.stack(train_input)
    train_target = torch.stack(train_target)

    train_set = TensorDataset(train_input, train_target)

    # Model setup and training
    input = 28*28
    output = 28*28
    hidden_layers = [input/2, output/50, output/2]
    model = MLP(input, hidden_layers, output, 'relu')
    model.cuda()

    lr = 0.001
    batch_size = 30
    num_epoch = 50

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    # set the loss function for the model
    model.loss = nn.MSELoss()

    model.display_layers()
    train(model, train_loader, opt, num_epoch)

    # Save the model
    torch.save(model.state_dict(), './savedModel.pth')


if __name__ == "__main__":
  main()
  print("Program has finished running!")

