import os
import urllib
import eagerpy as ep
import torch
import resnet
import foolbox as fb


from loss_and_decision_landscape import compute_loss_landscape, load_pytorch_cifar10, load_robustbench_model, \
    plot_2d_loss, get_random_direction, compute_decision_surface, plot_2d_decision_surface

if __name__ == '__main__':

    # Load a robustbench model
    mod_name1 = 'Standard'
    mod_name2 = 'Carmon2019Unlabeled'
    mod_name3 = 'Ding2020MMA'
    mod_name4 = 'k-wta'

    model1 = load_robustbench_model('C:/Users/gioma/secml-data/models/robustbench/', mod_name1)
    model2 = load_robustbench_model('C:/Users/gioma/secml-data/models/robustbench/', mod_name2)
    model3 = load_robustbench_model('C:/Users/gioma/secml-data/models/robustbench/', mod_name3)

    import logging

    logger = logging.getLogger('kwtalogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('kwta_debug.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    filename = 'kwta_spresnet18_0.1_cifar_adv.pth'
    url = f'https://github.com/wielandbrendel/robustness_workshop/releases/download/v0.0.1/{filename}'

    if not os.path.isfile(filename):
        print('Downloading pretrained weights.')
        urllib.request.urlretrieve(url, filename)

    gamma = 0.1
    epsilon = 0.031
    filepath = f'kwta_spresnet18_{gamma}_cifar_adv.pth'

    model4 = resnet.SparseResNet18(sparsities=[gamma, gamma, gamma, gamma], sparse_func='vol')
    model4.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model4.eval();

    # Load the CIFAR10 test dataset
    testset, testloader, classes = load_pytorch_cifar10(batch_size=1)

    # Take batch_size images and the corresponding labels
    dataiter = iter(testloader)
    imgs, labs = dataiter.next()

    # Range for the coordinates
    xmn, xmx = -0.4, 0.4
    ymn, ymx = -0.4, 0.4
    # Number of points were the loss is calculated
    steps = 31

    # Create a list of tensors with net weights
    wgs = [p.data for p in model1.parameters()]

    # Take two random directions
    # dirs = [get_random_direction(wgs), get_random_direction(wgs)]

    x_dir = get_random_direction(wgs)

    # Here is the attack
    fmodel = fb.PyTorchModel(model1, bounds=(0, 1))

    image = ep.astensor(imgs)
    label = ep.astensor(labs)
    criterion = fb.criteria.Misclassification(label)

    print("Attack started")
    pgd_attack = fb.attacks.projected_gradient_descent.LinfProjectedGradientDescentAttack()
    adv_sample = pgd_attack.run(fmodel, image, criterion, epsilon=8/255)
    print("Attack completed")

    loss_function = pgd_attack.get_loss_fn(fmodel, label)
    _, gradients = pgd_attack.value_and_grad(loss_function, image)

    print(gradients.shape)

    y_dir = list(torch.as_tensor(gradients.raw, dtype=torch.float32))
    x_dir = get_random_direction(y_dir)

    dirs = [x_dir, y_dir]

    # Compute the loss landscape on the original image
    xc_std, yc_std, losses_matrix_std = compute_decision_surface(net=model1, weights=wgs, directions=dirs, image=imgs,
                                                                 label=labs, xmin=xmn, xmax=xmx, ymin=ymn,
                                                                 ymax=ymx, n_steps=steps)

    # Plot the loss landscape
    plot_2d_decision_surface(mod_name1+'1', xc_std, yc_std, losses_matrix_std, vmin=-10.0, vmax=10.0, vlevel=0.5)

    # wgs = [p.data for p in model2.parameters()]
    # dirs = get_random_directions(wgs)
    # xc_car, yc_car, losses_matrix_car = compute_decision_surface(net=model2, weights=wgs, directions=dirs, image=imgs,
    #                                                              label=labs, xmin=xmn, xmax=xmx, ymin=ymn,
    #                                                              ymax=ymx, n_steps=steps)
    # plot_2d_decision_surface(mod_name2, xc_car, yc_car, losses_matrix_car, vmin=-4.0, vmax=4.0, vlevel=0.2)

    # xc_din, yc_din, losses_matrix_din = compute_decision_surface(net=model3, weights=wgs, directions=dirs, image=imgs,
    #                                                              label=labs, xmin=xmn, xmax=xmx, ymin=ymn,
    #                                                              ymax=ymx, n_steps=steps)
    # plot_2d_decision_surface(mod_name3, xc_din, yc_din, losses_matrix_din, vmin=-20.0, vmax=10.0, vlevel=0.5)

    # xc_kwta, yc_kwta, losses_matrix_kwta = compute_decision_surface(net=model4, weights=wgs, directions=dirs,
    #                                                                 image=imgs, label=labs, xmin=xmn, xmax=xmx,
    #                                                                 ymin=ymn, ymax=ymx, n_steps=steps)
    # plot_2d_decision_surface(mod_name4+'sample2', xc_kwta, yc_kwta, losses_matrix_kwta, vmin=-30.0, vmax=10.0, vlevel=0.5)
