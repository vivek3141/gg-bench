import yaml
from jinja2 import Template


def load_yaml(path: str) -> dict:
    """
    Load and parse a YAML file.

    This function reads a YAML file from the given path and returns its contents as a dictionary.
    It also adds a custom constructor for the '!join' tag, which joins sequence elements into a string.

    Args:
        path (str): The path to the YAML file.

    Returns:
        dict: The parsed contents of the YAML file.
    """

    def join_sequence(loader, node):
        seq = loader.construct_sequence(node)
        return "".join([str(i) for i in seq])

    def render_jinja2_template(loader, node):
        values = loader.construct_sequence(node)
        template_string = values[0]
        if isinstance(values[1], dict):
            return Template(template_string).render(values[1])
        else:
            return Template(template_string).render(values=values[1:])

    yaml.add_constructor("!jinja2", render_jinja2_template)
    yaml.add_constructor("!join", join_sequence)

    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
