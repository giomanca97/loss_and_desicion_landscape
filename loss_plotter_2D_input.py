import time
import numpy as np
import torch
import torchvision
from robustbench import load_model
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms


def load_robustbench_model(model_name, model_dir='C:/Users/gioma/secml-data/models/robustbench/', norm='Linf'):

    # Load the robustbench model
    net = load_model(model_name=model_name, norm=norm, model_dir=model_dir)
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


class Plotter2dInput:
    def __init__(self, model_name, model, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, n_points=31):

        self.model_name = model_name
        self.net = model
        self.directions = None
        self.n_points = n_points

        self.criterion = None

        # Defining a set of coordinates were I am going to compute the loss

        self.xcoordinates = np.linspace(xmin, xmax, self.n_points)
        self.ycoordinates = np.linspace(ymin, ymax, self.n_points)

        # Initialize variables
        self.shape = (len(self.xcoordinates), len(self.ycoordinates))
        self.losses = None
        self.dec_functions = None

    def compute_loss_landscape(self, image, label, directions, alfa, beta, criterion='CrossEntropy', show=True,
                               vmin=0.0, vmax=10.0, vlevel=0.5):

        self.directions = directions

        # Define criterion for the loss function
        if criterion is 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            raise NotImplementedError

        self.losses = -np.ones(shape=self.shape)  # Initialize matrix that stores losses

        with torch.no_grad():
            print('Computing')
            start_time = time.time()
            # Loop over all uncalculated loss values
            for (countx, county), ind in np.ndenumerate(self.losses):
                # countx, county are indexes to take all the points in the losses matrix
                # ind contains an entire row of losses

                # Get the coordinates of the loss value being calculated
                coords = self.xcoordinates[countx], self.ycoordinates[county]  # take the coordinate at index 'count'

                # Change the input depending on the directions
                dx = self.directions[0]
                dy = self.directions[1]
                changes = [d0 * coords[0] + d1 * coords[1] for (d0, d1) in zip(dx, dy)]
                img = torch.empty((1, 3, 32, 32))

                for (i, d) in zip(image, changes):
                    img[0] = i + d
                # Compute the output scores
                outputs = self.net(img)
                # Compute the loss
                loss = self.criterion(outputs, label)

                print("%d %d loss val: %.3f" % (countx, county, loss))

                # Record the result in the matrix
                self.losses[countx][county] = loss

        total_time = time.time() - start_time
        print('Total time: %.2f' % total_time)

        if show is False:
            return self.xcoordinates, self.ycoordinates, self.losses
        else:
            Plotter2dInput.plot_2d_loss_landscape(self, alfa, beta, vmin, vmax, vlevel)

    def plot_2d_loss_landscape(self, alfa, beta, vmin=0.0, vmax=10.0, vlevel=0.5):

        # Generate x, y, z coordinates
        x, y = np.meshgrid(self.xcoordinates, self.ycoordinates)
        z = self.losses
        n = len(self.xcoordinates)

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
        plt.plot(alfa[0], beta[0], 'go')
        plt.plot(alfa[-1], beta[-1], 'ro')
        plt.plot(alfa[1:-1], beta[1:-1], 'r-')

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
        sns_plot.get_figure().savefig(self.model_name + '-steps_' + str(n) + '_2dcontour-heatmap.pdf', dpi=300,
                                      bbox_inches='tight', format='pdf')
        plt.show()

    def compute_decision_surface(self, image, label, directions, alfa, beta, show=True, vmin=-10.0, vmax=10.0, vlevel=0.5):

        self.directions = directions

        self.dec_functions = -np.ones(shape=self.shape)

        with torch.no_grad():
            print('Computing')
            start_time = time.time()

            # Loop over all uncalculated values
            for (countx, county), ind in np.ndenumerate(self.dec_functions):
                coords = self.xcoordinates[countx], self.ycoordinates[county]  # take the coordinate at index 'count'

                # Change the input depending on the directions
                dx = directions[0]
                dy = directions[1]
                changes = [d0 * coords[0] + d1 * coords[1] for (d0, d1) in zip(dx, dy)]
                img = torch.empty((1, 3, 32, 32))

                for (i, d) in zip(image, changes):
                    img[0] = i + d

                # Compute the output scores
                outputs = self.net(img)
                true_score = outputs[0][label]
                best_class = np.argmax(outputs[0])
                if best_class == label:
                    best_adv_score = outputs[0].sort()[0][-2]
                else:
                    best_adv_score = outputs[0].max()
                # Compute the decision function
                decision_function = true_score - best_adv_score

                print("%d %d decision function: %.3f" % (countx, county, decision_function))

                # Record the result in the matrix
                self.dec_functions[countx][county] = decision_function

        total_time = time.time() - start_time
        print('Total time: %.2f' % total_time)

        if show is False:
            return self.xcoordinates, self.ycoordinates, self.dec_functions
        else:
            Plotter2dInput.plot_2d_decision_surface(self, alfa, beta, vmin, vmax, vlevel)

    def plot_2d_decision_surface(self, alfa, beta, vmin=-10.0, vmax=10.0, vlevel=0.5):

        # Generate x, y, z coordinates
        x, y = np.meshgrid(self.xcoordinates, self.ycoordinates)
        z = self.dec_functions
        n = len(self.xcoordinates)
        levels = np.arange(vmin, vmax, vlevel).round(4)

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
        if np.where(levels == 0.0)[0]:
            zero_index = int(np.where(levels == 0.0)[0])
            cs.collections[zero_index].set_linestyle('dashed')
            cs.collections[zero_index].set_color('green')
        plt.clabel(cs, inline=1, fontsize=6)
        plt.plot(alfa[0], beta[0], 'go')
        plt.plot(alfa[-1], beta[-1], 'ro')
        plt.plot(alfa[1:-1], beta[1:-1], 'r-')

        # Plot 2D filled contours
        plt.subplot(1, 2, 2)
        cs = plt.contourf(x, y, z, cmap='viridis', levels=levels)
        plt.title('contourf')

        # Save the results
        fig.savefig(self.model_name + '-adv-steps_' + str(n) + '_2d_decision_surface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')
        plt.show()



