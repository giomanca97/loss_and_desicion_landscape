import time
import torch
import torchvision
import torchvision.transforms as transforms
from robustbench import load_model
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def compute_loss_landscape(net, weights, directions, images, labels, xmin=-1.0, xmax=1.0,
                           ymin=-1.0, ymax=1.0, n_steps=31):
    """
        Calculate the loss values and accuracies.
    """

    with torch.no_grad():

        # Defining a set of coordinates were I am going to compute the loss
        xcoordinates = np.linspace(xmin, xmax, n_steps)
        ycoordinates = np.linspace(ymin, ymax, n_steps)

        # Initialize variables
        shape = (len(xcoordinates), len(ycoordinates))  # (51, 51) -> the size of the image
        losses = -np.ones(shape=shape)  # Initialize matrix that stores losses
        accuracies = -np.ones(shape=shape)  # Initialize also the accuracy matrix

        print('Computing')
        start_time = time.time()

        # Define criterion for the loss function
        criterion = nn.CrossEntropyLoss(reduction='none')

        # Loop over all uncalculated loss values
        for (countx, county), ind in np.ndenumerate(losses):
            # countx, county are indexes from 0 -> 50 to take all the elements in the losses matrix
            # ind contains an entire row of losses (list of 51 elements)

            # Get the coordinates of the loss value being calculated
            coords = xcoordinates[countx], ycoordinates[county]  # take the coordinate at index 'count'

            # Change the weights of the net depending on the directions
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * coords[0] + d1 * coords[1] for (d0, d1) in zip(dx, dy)]

            for (p, w, d) in zip(net.parameters(), weights, changes):
                p.data = w + torch.Tensor(d).type(type(w))

            # Compute the output scores
            outputs = net(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Compute the prediction
            _, prediction = torch.max(outputs.data, 1)
            # Compute the accuracy
            accuracy = prediction.eq(labels).sum().item()

            print("%d %d loss val: %.3f" % (countx, county, loss))

            # Record the result in the matrix
            losses[countx][county] = loss
            accuracies[countx][county] = accuracy

    total_time = time.time() - start_time
    print('Total time: %.2f' % total_time)

    return xcoordinates, ycoordinates, losses


def load_robustbench_model(outp_dir, model_name):

    # Load the robustbench model
    net = load_model(model_name=model_name, norm='Linf', model_dir=outp_dir)
    return net


def load_pytorch_cifar10(batch_size=1):

    # Load Cifar10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ts_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    ts_loader = torch.utils.data.DataLoader(ts_set, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    cls = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return ts_set, ts_loader, cls


def get_random_direction(weights):

    # Select a random tensor of weights as a direction for the two axis
    direction = [torch.randn(w.size()) for w in weights]

    # Normalize the direction
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))

    return direction


def plot_2d_loss(model_name, xcoord, ycoord, losses, vmin=0.1, vmax=10.0, vlevel=0.5):

    # Generate x, y, z coordinates
    x, y = np.meshgrid(xcoord, ycoord)
    z = losses
    n = len(xcoord)

    print('Plotting the 2D contour')
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))

    if len(x) <= 1 or len(y) <= 1:
        print('The length of coordinates is not enough for plotting contours')
        return

    # Plot 2D contours
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    cs = plt.contour(x, y, z, cmap='viridis', levels=np.arange(vmin, vmax, vlevel))
    plt.title('2D loss landscape')
    plt.clabel(cs, inline=1, fontsize=6)

    # Plot 2D filled contours
    plt.subplot(2, 2, 2)
    cs = plt.contourf(x, y, z, cmap='viridis', levels=np.arange(vmin, vmax, vlevel))
    plt.title('contourf')

    # Plot 2D heatmaps
    plt.subplot(2, 2, 3)
    sns_plot = sns.heatmap(z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    plt.title('heatmap')

    # Save the results
    sns_plot.get_figure().savefig(model_name + '-steps_' + str(n) + '_2dcontour-heatmap.pdf', dpi=300,
                                  bbox_inches='tight', format='pdf')
    plt.show()


def compute_decision_surface(net, weights, directions, image, label, xmin=-1.0, xmax=1.0,
                             ymin=-1.0, ymax=1.0, n_steps=31):
    """
        Calculate the decision values.
    """

    with torch.no_grad():

        # Defining a set of coordinates were I am going to compute the loss
        xcoordinates = np.linspace(xmin, xmax, n_steps)
        ycoordinates = np.linspace(ymin, ymax, n_steps)

        # Initialize variables
        shape = (len(xcoordinates), len(ycoordinates))  # (51, 51) -> the size of the image
        dec_functions = -np.ones(shape=shape)  # Initialize matrix that stores losses

        print('Computing')
        start_time = time.time()

        # Loop over all uncalculated loss values
        for (countx, county), ind in np.ndenumerate(dec_functions):
            # countx, county are indexes from 0 -> 50 to take all the elements in the losses matrix
            # ind contains an entire row of losses (list of 51 elements)

            # Get the coordinates of the loss value being calculated
            coords = xcoordinates[countx], ycoordinates[county]  # take the coordinate at index 'count'

            # Change the weights of the net depending on the directions
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * coords[0] + d1 * coords[1] for (d0, d1) in zip(dx, dy)]

            for (p, w, d) in zip(net.parameters(), weights, changes):
                p.data = w + torch.Tensor(d).type(type(w))

            # Compute the output scores
            outputs = net(image)
            true_score = outputs[0][label]
            best_class = np.argmax(outputs[0])
            if best_class == label:
                best_adv_score = outputs[0].sort()[0][-2]
            else:
                best_adv_score = outputs[0].max()
            # Compute the decision function
            decision_function = true_score - best_adv_score
            # Compute the prediction

            print("%d %d decision function: %.3f" % (countx, county, decision_function))

            # Record the result in the matrix
            dec_functions[countx][county] = decision_function

    total_time = time.time() - start_time
    print('Total time: %.2f' % total_time)

    return xcoordinates, ycoordinates, dec_functions


def plot_2d_decision_surface(model_name, xcoord, ycoord, decision_functions, vmin=-10.0, vmax=10.0, vlevel=0.5):

    # Generate x, y, z coordinates
    x, y = np.meshgrid(xcoord, ycoord)
    z = decision_functions
    n = len(xcoord)
    levels = np.arange(vmin, vmax, vlevel)
    zero_index = int(np.where(levels == 0)[0])

    print('Plotting the 2D contour')
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))

    if len(x) <= 1 or len(y) <= 1:
        print('The length of coordinates is not enough for plotting contours')
        return

    # Plot 2D contours
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    cs = plt.contour(x, y, z, cmap='viridis', levels=levels)
    plt.title('2D decision surface')
    cs.collections[zero_index].set_linestyle('dashed')
    cs.collections[zero_index].set_color('red')
    plt.clabel(cs, inline=1, fontsize=6)

    # Plot 2D filled contours
    plt.subplot(1, 2, 2)
    cs = plt.contourf(x, y, z, cmap='viridis', levels=levels)
    plt.title('contourf')

    # Save the results
    fig.savefig(model_name + '-steps_' + str(n) + '_2d_decision_surface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    plt.show()
