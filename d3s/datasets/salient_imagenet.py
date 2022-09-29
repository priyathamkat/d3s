import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from d3s.constants import (IMAGENET_PATH, MTURK_RESULTS_CSV_PATH,
                           SALIENT_IMAGENET_MASKS_PATH)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class SalientImageNet(Dataset):
    def __init__(
        self,
        class_index,
        feature_indices,
        images_path=IMAGENET_PATH,
        masks_path=SALIENT_IMAGENET_MASKS_PATH,
        resize_size=256,
        crop_size=224,
    ):
        with open(
            Path(__file__).parent.parent / "metadata/imagenet_classes.json", "r"
        ) as f:
            self.classes = {int(k): v.split(",")[0] for k, v in json.load(f).items()}

        with open(
            Path(__file__).parent.parent / "metadata/imagenet_dictionary.json", "r"
        ) as f:
            self.dictionary = {int(k): v for k, v in json.load(f).items()}

        self._rng = np.random.default_rng()

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ]
        )

        wordnet_dict = eval(open(masks_path / "wordnet_dict.py").read())
        wordnet_id = wordnet_dict[class_index]

        self.images_path = images_path / "train" / wordnet_id
        self.masks_path = masks_path / wordnet_id

        image_names_file = self.masks_path / "image_names_map.csv"
        image_names_df = pd.read_csv(image_names_file)

        image_names = []
        feature_indices_dict = defaultdict(list)
        for feature_index in feature_indices:
            image_names_feature = image_names_df[str(feature_index)].to_numpy()

            for i, image_name in enumerate(image_names_feature):
                image_names.append(image_name)
                feature_indices_dict[image_name].append(feature_index)

        self.image_names = np.unique(np.array(image_names))
        self.feature_indices_dict = feature_indices_dict

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        curr_image_path = self.images_path / (image_name + ".JPEG")

        image = Image.open(curr_image_path).convert("RGB")
        image_tensor = self.transform(image)

        feature_indices = self.feature_indices_dict[image_name]

        all_mask = np.zeros(image_tensor.shape[1:])
        for feature_index in feature_indices:
            curr_mask_path = (
                self.masks_path
                / ("feature_" + str(feature_index))
                / (image_name + ".JPEG")
            )

            mask = np.asarray(Image.open(curr_mask_path))
            mask = mask / 255.0

            all_mask = np.maximum(all_mask, mask)

        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)
        return image_tensor, mask_tensor

    def get_random(self, _):
        index = self._rng.choice(len(self))
        return self[index]


class MTurkResults:
    def __init__(self, csv_path=MTURK_RESULTS_CSV_PATH):
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)

        self.aggregate_results(self.dataframe)

        self.class_feature_maps()
        self.core_spurious_labels_dict()
        self.spurious_feature_lists()

    def aggregate_results(self, dataframe):
        answers_dict = defaultdict(list)
        reasons_dict = defaultdict(list)
        feature_rank_dict = defaultdict(int)
        wordnet_dict = defaultdict(str)

        for row in dataframe.iterrows():
            index, content = row
            WorkerId = content["WorkerId"]

            class_index = int(content["Input.class_index"])
            feature_index = int(content["Input.feature_index"])
            feature_rank = int(content["Input.feature_rank"])

            wordnet_dict[class_index] = content["Input.wordnet_id"]

            key = str(class_index) + "_" + str(feature_index)

            main_answer = content["Answer.main"]
            confidence = content["Answer.confidence"]
            reasons = content["Answer.reasons"]

            answers_dict[key].append((WorkerId, main_answer, confidence, reasons))
            reasons_dict[key].append(reasons)

            feature_rank_dict[key] = feature_rank

        self.answers_dict = answers_dict
        self.feature_rank_dict = feature_rank_dict
        self.reasons_dict = reasons_dict
        self.wordnet_dict = wordnet_dict

    def core_spurious_labels_dict(self):
        answers_dict = self.answers_dict

        core_features_dict = defaultdict(list)
        spurious_features_dict = defaultdict(list)

        core_spurious_dict = {}
        core_list = []
        spurious_list = []
        for key, answers in answers_dict.items():
            class_index, feature_index = key.split("_")
            class_index, feature_index = int(class_index), int(feature_index)

            num_spurious = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer in ["separate_object", "background"]:
                    num_spurious = num_spurious + 1

            if num_spurious >= 3:
                spurious_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = "spurious"
                spurious_list.append(key)

            else:
                core_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = "core"
                core_list.append(key)

        self.core_spurious_dict = core_spurious_dict
        self.core_list = core_list
        self.spurious_list = spurious_list

        self.core_features_dict = core_features_dict
        self.spurious_features_dict = spurious_features_dict

    def spurious_feature_lists(self):
        answers_dict = self.answers_dict

        background_list = []
        separate_list = []
        ambiguous_list = []
        for key, answers in answers_dict.items():
            num_background = 0
            num_separate = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer == "background":
                    num_background = num_background + 1
                elif main_answer == "separate_object":
                    num_separate = num_separate + 1

            if num_background >= 3:
                background_list.append(key)
            elif num_separate >= 3:
                separate_list.append(key)
            elif (num_background + num_separate) >= 3:
                ambiguous_list.append(key)

        self.background_list = background_list
        self.separate_list = separate_list
        self.ambiguous_list = ambiguous_list

    def class_feature_maps(self):
        answers_dict = self.answers_dict

        keys_list = answers_dict.keys()

        feature_to_classes_dict = defaultdict(list)
        class_to_features_dict = defaultdict(list)

        for key in keys_list:
            class_index, feature_index = key.split("_")
            class_index = int(class_index)
            feature_index = int(feature_index)

            feature_to_classes_dict[feature_index].append(class_index)
            class_to_features_dict[class_index].append(feature_index)

        self.class_to_features_dict = class_to_features_dict
        self.feature_to_classes_dict = feature_to_classes_dict
