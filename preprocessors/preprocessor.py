import os
import json
import glob
import sys
import time
import cv2
from PIL import Image, ImageOps

import numpy as np

# python 2,3 compability
from six import string_types

from sklearn.model_selection import KFold
sys.path.insert(0, '.')
from helper import sort_nicely
from config import Config

#
# Preprocessor class that takes in a configuration file and only returns training,
# test and validation set
# If one  configuration has been checked and segmentation done, we use this segmentation for all
# upcomping configuration to allow comparison
# static permutation
# everything dependent on the configuration should be a class method
# TODO switch to scikit learn

# num_images x num_frames  x image_height x image_width x num_channels

# all information about the existing data should be contained in the config


class Preprocessor(object):
    """The base structure for all preprocessors"""

    @property
    def train_X(self):
        """The training data."""
        return self._train_X

    @property
    def train_y(self):
        """The training labels."""
        return self._train_y

    @property
    def val_X(self):
        """The validation data."""
        return self._val_X

    @property
    def val_y(self):
        """The validation labels."""
        return self._val_y

    @property
    def test_X(self):
        """The test data."""
        return self._test_X

    @property
    def test_y(self):
        """The test labels."""
        return self._test_y


    FRAME_ENDING = '.data'
    DELIMITER = '_'  # DELIMITER between FILE_NAME and recording_id
    IMAGE_ENDING = '.bmp'
    PIXEL_DEPTH = 255.0  # Number of levels per pixel.

    # this function should only contain code specific to loading and processing
    # images from folders
    def __init__(self, config, options):

        self.config = Config(config)
        self.num_labels = self.config.num_labels
        self.options = options
        self.depth_threshold = options.depth_threshold
        self.labels = self.config.labels
        self.data_dir = self.config.data_dir

        # needed for crossvalidation
        # number of current dataset
        # calls to get data
        # specific to configuration
        self.used_datasets = 0


    # loads the complete dataset defined in the data set
    def _load_dataset(self):
        # only train data is stretched
        # not test data
        dataset = np.ndarray(shape=(0, self.options.num_frames, self.image_height,
                                    self.image_width, self.num_channels), dtype=np.float32)

        # save extended dataset in
        self.extended_X = []
        self.extended_y = []
        data_size = []
        self.files = np.ndarray(shape=(0))

        for label_config in self.labels:
            # check if folder exists
            label = label_config['Name']
            print(label)
            folder = self.data_dir + label + '/'
            if not os.path.exists(folder):
                print("{} folder does not exist! Skipping".format(label))
            # create frame objects and images for current gesture
            data = self.__load_data(folder)
            extended_X, extended_y = self._extend_data(data, label_config)
            self.extended_X += extended_X
            self.extended_y += extended_y
            dataset = np.append(dataset, data, axis=0)
            data_size.append(len(data))

        # one-hot label encoding
        # einheits matrix
        labels = np.eye(self.num_labels, dtype=int)
        # repeat each one with the number in data_size
        labels = np.repeat(labels, data_size, axis=0)

        # randomize before segmenting into training and testing set
        # save permutation so we can check which images are wrong

        # use existing permutation or generate new one

        dataset, labels = self._randomize(dataset, labels, load=True)
        self.X, self.y = dataset, labels

    # loads data for a single label in folder
    def __load_data(self, folder):
        frame_files = glob.glob(folder + "*" + self.FRAME_ENDING)
        print(len(frame_files))

        data = np.ndarray(shape=(len(frame_files), self.options.num_frames, self.image_height,
                                 self.image_width, self.num_channels), dtype=np.float32)

        for frame_index, frame_file in enumerate(frame_files):
            images = self.__load_images(frame_file)
            data[frame_index, :, :] = images

        return data

    # loads images for a single recording
    # @param_in frame_file is a file name of a frame recordings
    def __load_images(self, frame_file):
        # find all images belonging to this recording by using string
        # manipulation

        # ensure new recording is chosen (otherwise mix up between recording_2_
        # and recording_20_
        image_filter = frame_file[
            :-len(self.FRAME_ENDING)] + self.DELIMITER + "*" + self.IMAGE_ENDING
        image_files = glob.glob(image_filter)
        sort_nicely(image_files)
        # calculate sample rate
        sample_period = len(image_files) / float(self.options.num_frames)
        # select samples
        samples = [int(x * sample_period) for x in range(self.options.num_frames)]

        images = np.ndarray(shape=(self.options.num_frames, self.image_height,
                                   self.image_width, self.num_channels), dtype=np.float32)
        for sample_index, sample in enumerate(samples):
            image = Image.open(image_files[sample])
            # first resize with a ratio so that shorter side already
            # has the correct size
            ratio = self.image_width / \
                image.width if image.width < image.height else self.image_height / image.height
            image.thumbnail((int(image.width * ratio),
                             int(image.height * ratio)))
            # cut everything else off
            image = ImageOps.fit(image, (self.image_width, self.image_height))

            if self.num_channels == 1:
                shape = (image.height, image.width, 1)
                # reshape to force 1 dimension
                image = np.array(image)
                image = np.reshape(image, shape)

            images[sample_index] = image
        return images

    # currentl not used
    def __load_frames(self, folder):
        frame_files = glob.glob(folder + "*" + self.FRAME_ENDING)
        frames = []
        for frame_file in frame_files:
            with open(frame_file, "rb") as file:
                frames.append(deserialize(file))
        return frames

    # only one return function that handles all data managment
    def get_data(self):

        kf = KFold(n_splits=5)
        # print(self.labels)
        for train, test in kf.split(self.X):
            # for(index in train):
            #    self.extended_dataset

            # extended data X and y for training, not needed for testing
            train_extended_X = np.array(self.extended_X)[train]
            shape = list(train_extended_X.shape)
            # after selecting merge first two dimensions which is sample x
            # num_of_augmentations
            train_extended_X = np.reshape(
                train_extended_X, [shape[0] * shape[1]] + shape[2::])

            train_extended_y = np.array(self.extended_y)[train]
            train_extended_y = np.reshape(train_extended_y,
                                         (train_extended_y.shape[0] *
                                          train_extended_y.shape[1],
                                          train_extended_y.shape[2])
                                         )

            # regular data for training
            train_X = self.X[train]
            train_y = self.y[train]

            # data for testing
            test_X = self.X[test]
            test_y = self.y[test]

            # concatenate
            train_X = np.concatenate((train_X, train_extended_X))
            train_y = np.concatenate((train_y, train_extended_y))
            train_X, train_y = self._randomize(train_X, train_y, False)


            yield {'train_X': train_X, 'train_y': train_y,
                   'valid_X': 0, 'valid_y': 0,
                   'test_X': test_X, 'test_y': test_y}

    # return all files name with permutation
    def get_files(self):
        return self.files[self.permutation]


    # extend data and save with same indexes as the original dataset
    def _extend_data(self, dataset, config):
        extended_X = []
        extended_y = []
        for clip in dataset:
            # reverse the order of the images in the clip
            extended_clip_X = []
            extended_clip_y = []
            if 'reverse' in config:
                # print(clip[::-1].shape)
                extended_clip_X.append(clip[::-1])
                extended_clip_y.append(self.hot_labels[config['reverse']])

            # mirror the images
            if 'mirror' in config:
                mirror_clip = []
                for image in clip:
                    # print(image.shape)
                    mirror_clip.append(np.fliplr(image))

                mirror_clip = np.array(mirror_clip)
                extended_clip_X.append(mirror_clip)
                extended_clip_y.append(self.hot_labels[config['mirror']])

            # rotate
            # TODO: this is not ideal as the cropped image is rotated,
            # some information might be lost
            rotate_clip = []
            for image in clip:
                im = np.reshape(
                    image, (image.shape[0], image.shape[1])).astype(np.uint8)
                im = Image.fromarray(im)

                # im.show()
                im = im.rotate(10)
                # TODO look for cleaner solution
                narray = np.array(im)
                shape = list(narray.shape)
                shape.append(1)
                narray = np.reshape(narray, shape)
                rotate_clip.append(narray)

            extended_clip_X.append(rotate_clip)
            extended_clip_y.append(self.hot_labels[config['Name']])

            extended_X.append(extended_clip_X)
            extended_y.append(extended_clip_y)
        return extended_X, extended_y

    # form for convolutional network
    def reformat(self, dataset, labels):
        #-1 unknown dimension
        # read out shape from dataset
        shape = dataset.shape
        dataset = dataset.reshape(
            (-1, self.image_height, self.image_width, self.num_channels)).astype(np.float32)
        return dataset, labels

    # normalize
    def normalize(self, dataset):
        # normalize for images
        dataset -= self.PIXEL_DEPTH / 2
        dataset /= self.PIXEL_DEPTH

    # preview a folder with the current settings

    def preview(self, folder, show=True, save=False):
        if os.path.exists(folder):
            frame_files = glob.glob(folder + "*" + self.FRAME_ENDING)
        else:
            print("Folder %s was not found" % folder)

        for frame in frame_files:
            # print(frame)
            data = self.__load_images(frame)
            # print(data.shape)
            # create one big image
            # assuming num_frames x height x width x channels
            # new takes size = (width, height)
            # but saves it in an array (height, width)
            new_image = Image.new(
                'RGB', (data.shape[2], data.shape[1] * data.shape[0]))
            for index, npimage in enumerate(data):
                im = Image.fromarray(np.reshape(
                    npimage, (data.shape[1], data.shape[2])))
                new_image.paste(im, (0, data.shape[1] * index))
            if show:
                new_image.show()
            if save:
                if not os.path.exists(folder + "overview"):
                    os.makedirs(folder + "overview")
                last = frame.find("recording")
                name = frame[last:]
                new_image.save(folder + "overview/" + name + ".jpg")

    def _increase_index(self, clip_indx, frame_indx):
        """increases the indices
        Args:
            clip_indx: The index for the current clip within the  video
            frame_indx: The index for the frame within the current clip
        Returns:
            increases frame and clip index
        """
        frame_indx += 1
        frame_indx = frame_indx % self.options.num_frames
        if frame_indx == 0:
            clip_indx += 1
        return clip_indx, frame_indx

    def _show_dataset(self, dataset, wait):
        """Displays the dataset
        Args:
            Wait time in ms to display one frame
        """
        dataset = dataset.reshape(-1, self.image_width, self.image_height, self.num_channels)
        for data in dataset:
            cv2.imshow('data', data)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

    @property
    def name(self):
        """The name of the preprocessor which is used in the folder name."""
        return "default"
    # save permutation to be able to reuse old model for further testing
    PERMUTATION_NAME = 'permutation.npy'

    @classmethod
    def load_permutation(cls, name=PERMUTATION_NAME):
        permutation_name = os.path.join(cls.DIR_NAME, name)
        print(permutation_name)
        if os.path.isfile(permutation_name):
            with open(permutation_name, 'rb') as file:
                print("restored permutation")
                return np.load(file)
        print(permutation_name)
        return np.ndarray(shape=(0))

    @classmethod
    def save_permutation(cls, permutation, name=PERMUTATION_NAME):
        permutation_name = os.path.join(cls.DIR_NAME, name)
        with open(permutation_name, 'wb+') as file:
            np.save(file, permutation)

    """
    this function is called twice, once to randomize the dataset before segmenting
    and again after extending
    we only need the permutation before segmenting to recover debugging
    information
    there are cases:
    case 1 - after extending:
    a new permutation is created
    case 2 - before segmenting not using a pretrained model
    create a new permutation for any first run, after that reuse this permutation
    while tuning hyperperparametrs
    case 2 - before segmenting using a pretrained model
    reuse in any case an already created permutation
    """
    @classmethod
    def _randomize(cls, dataset, labels, load=False):
        print("Randomize...")
        if load:
            permutation = cls.load_permutation()
            if not permutation.size:
                if cls.USE_PRETRAINED_MODEL:
                    print("Despite using a pretrained model, no permutation could be found! Exiting")
                    sys.exit()

                permutation = np.random.permutation(labels.shape[0])
                cls.save_permutation(permutation)
        else:
            permutation = np.random.permutation(labels.shape[0])

        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    # reset permutation
    @classmethod
    def reset(cls):
        cls.permutation = np.ndarray(shape=(0))
