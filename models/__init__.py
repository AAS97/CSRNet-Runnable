from models.CSRNet import CSRNet, CSRNet_Back


def get_model(args):
    model_arch = _get_model_instance(args.arch)
    model_back = _get_model_back(args.model_backend)

    print('Fetching model %s - %s ' % (args.arch, args.model_backend))
    if args.arch == 'CSRNet':
        model = _get_model_instance('CSRNet')
        model = model(model_back, args.pretrained)
    else:
        raise 'Model {} not available'.format(args.arch)

    return model


def _get_model_instance(name):
    return {
        'CSRNet': CSRNet,
    }[name]


def _get_model_back(back):
    if len(back) == 1:
        dilation = {
            'A': [1, 1, 1, 1, 1, 1],
            'B': [2, 2, 2, 2, 2, 2],
            'C': [2, 2, 2, 4, 4, 4],
            'D': [4, 4, 4, 4, 4, 4]
        }[back]
    else:
        dilation = _parse_dilation(back)

    return CSRNet_Back(dilation)


def _parse_dilation(string):
    mapping = map(int, string.split(','))
    return list(mapping)
