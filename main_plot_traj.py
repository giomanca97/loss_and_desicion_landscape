import matplotlib.pyplot as plt
import torch
import torchvision
from secml.adv.attacks import CFoolboxPGDLinf, CFoolboxPGDL2
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml import CClassifierPyTorch, CNormalizerMinMax
import numpy as np

from direction_solver import find_directions
from loss_plotter_2D_input import load_robustbench_model, Plotter2dInput

mod_name = 'Standard'
model = load_robustbench_model(mod_name)

tr, ts = CDataLoaderCIFAR10().load()
normalizer = CNormalizerMinMax().fit(tr.X)

dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
attack_ds = ts[1, :]
attack_ds.X = normalizer.transform(attack_ds.X)

input_shape = (3, 32, 32)

secml_model = CClassifierPyTorch(model, input_shape=input_shape, pretrained=True)

dmax = 3.0
lb, ub = 0., 1.
y_target = None

attack = CFoolboxPGDLinf(classifier=secml_model,
                         y_target=y_target,
                         lb=lb, ub=ub,
                         steps=100, abs_stepsize=0.01,
                         epsilons=dmax)

print("Orig. Label: {}".format(attack_ds.Y.item()))
print("Attacking...")
eva_y_pred, _, eva_adv_ds, _ = attack.run(attack_ds.X, attack_ds.Y)
print("Adv. Label: {}".format(eva_y_pred.item()))

path = attack.x_seq.tondarray()
x0 = attack_ds.X[0, :].tondarray()
x_adv = eva_adv_ds.X.tondarray()

path_torch = torch.from_numpy(path)

v1, v2 = find_directions(path_torch)

alfa = path_torch @ v2.T
beta = path_torch @ v1.T

# show images of the directions
for t in [v1, v2]:
    plt.figure()
    dir_name = "v1" if t is v1 else "v2"
    img = t.clone()
    img = torchvision.transforms.functional.to_pil_image(img.view(3, 32, 32))
    plt.imshow(img, vmin=-1, vmax=1, cmap='gray')
    plt.title(dir_name)

# plt.show()

# Range for the coordinates
xmn, xmx = -50.0, 50.0
ymn, ymx = -50.0, 50.0
# Number of points were the loss is calculated
steps = 11

image = as_tensor(attack_ds.X[0, :].reshape((1, 3, 32, 32)))
label = torch.tensor(as_tensor(attack_ds.Y), dtype=torch.long)

dirs = [v2.reshape((1, 3, 32, 32)), v1.reshape((1, 3, 32, 32))]

pl = Plotter2dInput(mod_name, model, xmin=xmn, xmax=xmx, ymin=ymn, ymax=ymx, n_points=steps)
pl.compute_loss_landscape(image, label, dirs, alfa, beta, vmax=15.0)
pl.compute_decision_surface(image, label, dirs, alfa, beta, vmin=-15.0, vmax=15.0)
