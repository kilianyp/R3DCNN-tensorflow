"""Preprocessor for a dataset recorded from a Kinect v1 Camera."""
from preprocessor import Preprocessor

import numpy as np
import cv2
import pdb
import sys
import os
from random import randint


class R3DCNNPreprocessor(Preprocessor):
    @property
    def name(self):
        return "R3DCNN"
    def __init__(self, config, options):
        super(R3DCNNPreprocessor, self).__init__(config, options)
        self.num_clips = self.options.frames // self.options.num_frames
        self.image_width = 112
        self.image_height = 112
        self.num_channels = 1

    # TODO Unit test
    def _should_load(self, subjects, path, include):
        """Check if this folder should be used for training.

        First check if subject in path.
        Depending on include this subject is then included or not.
        If subject is not in path, it means that we do the opposite of before.

        Args:
            include indicates if the elemtents in subjects should be loaded
            or left out

        """
        for subject in subjects:
            if subject in path:
                """
                if include:
                    return True
                else:
                    return False
                """
                return include

        """if include:
            return False
        else:
            return True
        """
        return not include

    def load_train_data(self, leave_out=[]):
        """Load training data where leave_out is left out.

        Args:
            leave_out: list of subjects that are not used for training
        """
        self._train_X, self._train_y = self._load_data(self.image_width,
                                                       self.image_height,
                                                       leave_out,
                                                       include=False,
                                                       extend=True)
        print("Train data")
        print(self.train_X.shape)
        print(self.train_y.shape)

    def load_pretrain_data(self, leave_out=[]):
        """Load training data and then extract 16 frames from it."""
        X, y = self._load_data(self.image_width, self.image_height,
                               leave_out, include=False, extend=True)

        shape = [X.shape[0]] + [-1] + list(X.shape[3:])
        X = X.reshape(shape)
        # resample to num_frames
        sample_period = self.options.frames / float(self.options.num_frames)
        samples = [int(x * sample_period) for x in range(self.options.num_frames)]
        X = X[:, samples, :]
        print(X.shape)
        self.pretrain_X = X
        self.pretrain_y = y



    def load_test_data(self):
        pass

    def load_subjects(self, subjects, extend):
        """Load specific subjects."""
        data_X, data_y = self._load_data(self.image_width, self.image_height,
                                         subjects, include=True, extend=extend)
        print(data_X.shape, data_y)
        return data_X, data_y

    def load_val_data(self, subjects=[]):
        """Load validation data where leave_out is used as validation.

        Args:
            leave_out: list of subjects that are used for validation
        """
        self._val_X, self._val_y = self._load_data(self.image_width,
                                                   self.image_height,
                                                   subjects,
                                                   include=True,
                                                   extend=False)
        print("Validation data")
        print(self.val_X.shape)
        print(self.val_y.shape)

    def _load_data(self, image_width, image_height,
                   subjects=[], include=False, extend=False):
        """Load data into numpy array.

        Args:
            image_width: The width the original image is resized to
            image_height: The height the original image is resized to
            include: Flag that indicates if the elements in subjects are
                     used or not used
            subjects: A list of subjects that are loaded or not loaded.

        Default is to load all available subjects (not exclude any).
        """
        # we train each gesture with 96 frames.
        # TODO best solution would be to use init state to allow training of
        # any clip lenth.
        # or use sparse

        data_X = []
        data_y = []

        # files of this subject are used in the test set
        for label, label_name in enumerate(self.labels):
            # check if folder exists
            path = os.path.join(self.data_dir, label_name)
            if not os.path.exists(path):
                print("{} folder does not exist! Skipping".format(path))
            # data for one gesture
            for directory, _, video_files in os.walk(path):
                if len(video_files):
                    if self._should_load(subjects, directory, include):
                        for video_file in video_files:
                            # print(path)

                            path = os.path.join(directory, video_file)
                            X, y = self._load_data_from_file(path, [label],
                                                             image_width,
                                                             image_height,
                                                             self.options.frames,
                                                             extend=extend)
                            data_X.extend(X)
                            data_y.append(y)
                            if self.options.debug:
                                break

        data_X = np.array(data_X)
        data_y = np.array(data_y)
        # force unkown dimension to be able to concatenate later with csv
        data_y = data_y.reshape((data_y.shape[0],))

        # self._show_dataset(self.train_X, 10)
        # self._show_dataset(self.test_X, 10)

        return data_X, data_y

    def load_original(self, subjects):
        """Load samples for subjects from dataset in original size."""
        return self._load_data(640, 480, subjects, True)

    def _load_data_from_file(self, path, label, image_width,
                              image_height, frames, extend=False):
        """Load video file from path.

        Args:
            path: path to video file
            image_width: width of resized video
            image_height: height of resized video
            frames: number of frames this file is strechted or cut to
            extend: bool if this data should be augmented
            label: a list of labels
        Returns:
            labels: label for file
            video_container: numpy array containing the video file(s)
        """
        cap = cv2.VideoCapture(path)

        if extend:
            temp_rotate_scat = randint(-5, 5)
            temp_scale_scat = randint(-5, 5)
        else:
            temp_rotate_scat = 0
            temp_scale_scat = 0

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #only do temporal scattering if enough film material
        if frames > length:
            buffer_add = max(0, temp_rotate_scat, temp_scale_scat)
            buffer_sub = min(0, temp_rotate_scat, temp_scale_scat)
        else:
            buffer_add = 0
            buffer_sub = 0

        # number of frames that are necessary to load
        # depending on the video length we need to expand or shorten it
        if extend:
            num_frames = frames - buffer_sub + buffer_add
            num_clips = frames // self.options.num_frames
        else:
            num_clips = length // self.options.num_frames
            num_frames = num_clips * self.options.num_frames
            frames = num_frames
        # extend if necessary or cut
        # three timelines:
        # the one in the original clip
        # the one in the new complete clip, meaning containing all the frames
        # including the buffer frames
        # the one in the actual clip, only containing the actual frames for the augmentation
        # place the start and end frame are the position where the video
        # starts and ends in the
        # 0 - buf_sub - start - 96 frames - end - buf_add - end
        # print(buffer_add, buffer_sub)
        # start is where the clip begins, end where the new clip ends
        # load_from is the in the new clip
        # start_frame from is the position in the original clip should be loaded from
        # variable that holds the actual start in the video
        start_frame = 0
        if length < num_frames:
            # we have to do temporal padding
            start = 0 - buffer_sub
            end = start + frames
            load_from = (frames - length) // 2
            load_end = load_from + length
        else:
            # place in the middle including buffer
            load_from = int((length - num_frames) / 2)
            start = load_from - buffer_sub
            end = start + frames
            load_end = load_from + num_frames
            start_frame = load_from

        # timeline of the original clip
        load_range = range(load_from, load_end)
        # new time line without augmentation
        original_range = range(start, end)
        # new time line with rotation
        rotate_range = range(start + temp_rotate_scat,
                             end + temp_rotate_scat)
        # new time line with scaling
        scale_range = range(start + temp_scale_scat,
                            end + temp_scale_scat)
        # this is the same as load range when the clip is long enough
        frames_to_load = range(start_frame,
                               start_frame + num_frames)

        # print(load_range)
        # print(original_range)
        # print(rotate_range)
        # print(scale_range)
        # print(frames_to_load)

        original_container = np.zeros(
            (num_clips, self.options.num_frames, image_height,
             image_width, self.num_channels), dtype=np.uint8)
        scaled_container = np.zeros(
            (num_clips, self.options.num_frames, image_height,
             image_width, self.num_channels), dtype=np.uint8)
        rotated_container = np.zeros(
            (num_clips, self.options.num_frames, image_height,
             image_width, self.num_channels), dtype=np.uint8)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        copied = 0

        # holds the frame that should be copied if necessary
        copy_frame = np.ndarray(shape=[0])
        copy_scaled = np.ndarray(shape=[0])
        copy_rotated = np.ndarray(shape=[0])

        # we save index for each individually as they have their own temporal scattering
        rotate_frame_indx = 0
        scale_frame_indx = 0
        original_frame_indx = 0
        rotate_clip_indx = 0
        scale_clip_indx = 0
        original_clip_indx = 0

        # we need to set the copies to allow copies before the clip
        first = True

        #
        rot_scat = randint(-15, 15)
        scale_scat = randint(0, 20)

        orig_width = 160
        orig_height = 120

        margin_width = (orig_width-image_width)
        margin_height = (orig_height-image_height)

        crop_rot_width = randint(0, margin_width)
        crop_rot_height = randint(0, margin_height)

        crop_scale_width = randint(0, margin_width)
        crop_scale_height = randint(0, margin_height)

        if extend:
            crop_orig_width = randint(0, margin_width)
            crop_orig_height = randint(0, margin_height)
        else:
            crop_orig_width = margin_width // 2
            crop_orig_height = margin_height // 2


        for indx, frameIndx in enumerate(frames_to_load):

            if frameIndx in load_range:
                """ if image is needed for any argumentation and the frame
                exists in the original clip"""
                ret, frame = cap.read()
            elif first:
                """ if this is the first iteration pre generate the image
                to be able to copy it in front of the clip
                """
                ret, frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            else:
                """ if the frame does not exist in the original clip and we have a
                copy of the first used frame """
                ret = False

            if ret:

                rows, cols, _ = frame.shape
                # rotation between -15 and 15 degrees
                if (frameIndx in rotate_range and extend) or first:
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), rot_scat, 1)
                    rotated_frame = cv2.warpAffine(frame, M, (cols, rows))
                    rotated_frame = cv2.resize(rotated_frame,
                                               (orig_width, orig_height))
                    rotated_frame = rotated_frame[crop_rot_height:crop_rot_height + image_height,
                                                crop_rot_width:crop_rot_width + image_width]
                    rotated_frame = rotated_frame[..., 0]
                    rotated_frame = rotated_frame[..., np.newaxis]
                    copy_rotated = rotated_frame
                    if frameIndx in rotate_range:
                        rotated_container[rotate_clip_indx,
                                          rotate_frame_indx, ...] = rotated_frame
                        rotate_clip_indx, rotate_frame_indx = self._increase_index(rotate_clip_indx, rotate_frame_indx)

                # scale image
                # we scale the image by cropping it but resizing it to the same size afterwards
                if (frameIndx in scale_range and extend) or first:
                    factor = 1 - scale_scat/100
                    scaled_rows = rows * factor
                    scaled_cols = cols * factor
                    # => crop from middle
                    new_rows = int((rows - scaled_rows)/2)
                    new_cols = int((cols - scaled_cols)/2)
                    scaled_frame = frame[new_rows:rows-new_rows, new_cols:rows-new_cols]
                    scaled_frame = cv2.resize(scaled_frame, (orig_width, orig_height))
                    scaled_frame = scaled_frame[crop_scale_height:crop_scale_height + image_height,
                                                crop_scale_width:crop_scale_width + image_width]
                    scaled_frame = (scaled_frame * (1 + scale_scat/50)).astype(np.uint8)
                    scaled_frame = scaled_frame[..., 0]
                    scaled_frame = scaled_frame[..., np.newaxis]
                    copy_scaled = scaled_frame
                    if frameIndx in scale_range:
                        scaled_container[scale_clip_indx, scale_frame_indx, ...] = scaled_frame
                        scale_clip_indx, scale_frame_indx = self._increase_index(scale_clip_indx, scale_frame_indx)


                if frameIndx in original_range or first:
                    # resize
                    frame = cv2.resize(frame, (orig_width, orig_height))
                    frame = frame[crop_orig_height:crop_orig_height + image_height,
                                  crop_orig_width:crop_orig_width + image_width]
                    frame = frame[..., 0]
                    frame = frame[..., np.newaxis]
                    copy_frame = frame
                    if frameIndx in original_range:
                            original_container[original_clip_indx, original_frame_indx, ...] = frame
                            original_clip_indx, original_frame_indx = self._increase_index(original_clip_indx, original_frame_indx)

                # we get our clip, from now on copy this one if necessary
                first = False

            else:
                # if copy exists it means that also other copies exist
                if copy_frame.size:
                    copied += 1
                    if frameIndx in original_range:
                        original_container[original_clip_indx, original_frame_indx, ...] = copy_frame
                        original_clip_indx, original_frame_indx = self._increase_index(original_clip_indx, original_frame_indx)
                    if frameIndx in scale_range and extend:
                        scaled_container[scale_clip_indx, scale_frame_indx, ...] = copy_scaled
                        scale_clip_indx, scale_frame_indx = self._increase_index(scale_clip_indx, scale_frame_indx)

                    if frameIndx in rotate_range and extend:
                        rotated_container[rotate_clip_indx, rotate_frame_indx, ...] = copy_rotated
                        rotate_clip_indx, rotate_frame_indx = self._increase_index(rotate_clip_indx, rotate_frame_indx)

                else:
                    print("Empty frame")
                    print(path)
        cap.release()
        if self.options.debug:
            print("copied %s frames" % (copied))
        # TODO: quite inefficient as they are all created
        if extend:
            return [original_container, scaled_container, rotated_container], label * 3
        else:
            return [original_container], label

    def process(self, video):
        """Shrink frame."""
        # TODO dynamic
        processed_video = np.ndarray(shape=(1, 8, 112, 112, 1))
        for clip_idx, clip in enumerate(video):
            for frame_idx, frame in enumerate(clip):
                # frame[frame > threshold] = 0
                frame = cv2.resize(frame, (160, 120))
                # crop from middle
                # TODO dynamic
                frame = frame[4:116, 24:136]

                frame = frame[..., np.newaxis]
                processed_video[clip_idx, frame_idx, :] = frame
        return processed_video
