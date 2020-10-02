from pathlib import Path
import yaml

class Configuration:
    config = dict()

    def load(self, dir='./config'):
        for path in Path(dir).rglob('*.yml'):
            self.configure_yaml(path)
        for path in Path(dir).rglob('*.yaml'):
            self.configure_yaml(path)
        return self

    def configure_yaml(self, path):
        with open(path) as f:
            self.config.update(yaml.load(f, Loader=yaml.FullLoader))
            # self.config.update(yaml.load(f, Loader=yaml.SafeLoader))

    def get_config(self):
        return self.config
