import yaml


def load_yaml_params(params_path):
    """
    Modified yaml safe loading to allow for yaml to load python lambdas given by !python/lambda
    """

    def yaml_lambda_constructor(loader, node):
        value = loader.construct_scalar(node)
        return eval(value)

    yaml.SafeLoader.add_constructor("!python/lambda", yaml_lambda_constructor)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params
