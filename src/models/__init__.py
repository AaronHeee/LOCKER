from .locker_model import LOCKERModel

MODELS = {
    "locker": LOCKERModel
}

def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
