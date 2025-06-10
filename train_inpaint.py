# needed libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mlp import MLP
from setup import setup_train


# Training the model
def train(model, train, opt, epoch):
    transform = transforms.Compose([
        transforms.RandomErasing(1, (0.2, 0.4))
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
    # get the training set from MNIST
    train_set = setup_train("MNIST")

    # Model setup and training
    input = 28*28
    output = 28*28
    hidden_layers = [input/2, input/3, input/10, output/3, output/2]
    model = MLP(input, hidden_layers, output, 'relu')
    model.cuda()

    lr = 0.001
    batch_size = 30
    num_epoch = 100

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

