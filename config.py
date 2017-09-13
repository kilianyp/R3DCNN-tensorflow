"""Configuration class that interfaces the config file."""
import json
import os
from hardware.hardware import Hardware

class Config(object):
    DATA_FOLDER = "dataset"
    NAME = "config.json"
    def __init__(self, path):
        # check if json file given or folder
        if path[-5:] != ".json":
            path = os.path.join(path, self.NAME)

        with open(path) as file:
            self.config = json.load(file)
        self.label_configs = self.config['Actions']
        self.labels = ["No gesture"]
        for index, label_config in enumerate(self.config['Actions']):
            self.labels.append(label_config['Name'])
        self.labels.append("blank")
        self.num_labels = len(self.labels)
        self.hardware = Hardware(self.config['hardware'])
        self.streams = self.hardware.from_module("STREAMS")
        self.size = {}
        for stream in self.streams:
            width = self.hardware.from_module(stream + "_WIDTH")
            height = self.hardware.from_module(stream + "_HEIGHT")
            self.size[stream] = (width, height)
        self.data_dir = os.path.join("datasets", self.hardware.name)

    def dump(self, location):
        """Dump the current config."""
        name = os.path.join(location, self.NAME)
        with open(name, 'w') as out:
            json.dump(self.config, out)
