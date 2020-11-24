import torch

def get_optimizer(name, model, **kwargs):
    if name == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=kwargs["lr"],
                                        momentum=kwargs["momentum"],
                                        weight_decay=kwargs["weight_decay"])
    elif name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=kwargs["lr"],
        )
    else:
        print('Unknown solver: {}'.format(name))
        assert False

    return optimizer
