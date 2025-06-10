import torch
import torch.nn as nn
from mlp import MLP
from setup import setup_test
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test(model, test, loss_func):
    transform = v2.Compose([
        v2.GaussianNoise(0.2, 0.2)
    ])
    error = 0.0
    for images, targets in test:
        images = transform(images)
        images = images.view(-1, model.input).cuda()
        targets = targets.view(-1, model.input).cuda()

        outputs = model(images)
        loss = loss_func(outputs, targets)
        error += loss.item()

        outputs = outputs.view(-1, 28, 28) # save the output images
        targets = targets.view(-1, 28, 28) # save the target images
        inputs = images.view(-1, 28, 28) # save the original input images
    
    print(error / len(test))
    return outputs, targets, inputs


def display_results(batch_size, sample, target, inputs):
    num_images = 5
    if batch_size > num_images:
        sample = sample[:num_images]
        target = target[:num_images]
        inputs = inputs[:num_images]
    for i in range(num_images):
        # Original
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(target[i].cpu().detach().numpy(), cmap="gray")
        plt.title(f"Original {i+1}")
        plt.axis("off")

        # Altered
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(inputs[i].cpu().detach().numpy(), cmap="gray")
        plt.title(f"Altered {i+1}")
        plt.axis("off")

        # Generated
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(sample[i].cpu().detach().numpy(), cmap="gray")
        plt.title(f"Generated {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    batch_size = 64
    input = 28*28
    output = 28*28

    hidden_layers = [input/2, input/3, input/10, output/3, output/2]
    model = MLP(input, hidden_layers, output, 'relu').cuda()
    model.load_state_dict(torch.load('./savedModel.pth'))

    test_set = setup_test("FMNIST")

    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    sample, target, inputs = test(model, test_loader, nn.MSELoss())

    display_results(batch_size, sample, target, inputs)



if __name__ == "__main__":
    main()
    print("Testing has finished.")

