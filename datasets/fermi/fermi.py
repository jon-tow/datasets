# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""Fermi Problems: A New Reasoning Challenge for AI."""


import json

import datasets


_CITATION = """
@article{kalyan2021much,
  title={How Much Coffee Was Consumed During EMNLP 2019? Fermi Problems: A New Reasoning Challenge for AI},
  author={Kalyan, Ashwin and Kumar, Abhinav and Chandrasekaran, Arjun and Sabharwal, Ashish and Clark, Peter},
  journal={arXiv preprint arXiv:2110.14207},
  year={2021}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Fermi Problems are questions whose answer is a number that can only be reasonably
estimated as a precise measurement of the value is either impossible or impractical.
"""

_HOMEPAGE = "https://allenai.org/data/fermi"

_LICENSE = "https://creativecommons.org/licenses/by/4.0/"


class FermiConfig(datasets.BuilderConfig):
    """BuilderConfig for Fermi."""

    def __init__(self, urls, features, **kwargs):
        """BuilderConfig for Fermi.

        Args:
        urls: *string*, the urls to the specific subset of the Fermi Challenge dataset.
        features: *list[string]*, list of the features that will appear in the
            feature dict.
        """
        # Version history:
        super().__init__(**kwargs)
        self.urls = urls
        self.features = features


class Fermi(datasets.GeneratorBasedBuilder):
    """The Fermi Challenge."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        FermiConfig(
            name="realFP",
            version=datasets.Version("1.0.0"),
            description="A collection of 928 fermi problems and their solutions expressed in the form a program.",
            urls={
                "train": "https://raw.githubusercontent.com/allenai/fermi/main/data/realFP/train_realfp.json",
                "validation": "https://raw.githubusercontent.com/allenai/fermi/main/data/realFP/val_realfp.json",
                "test": "https://raw.githubusercontent.com/allenai/fermi/main/data/realFP/test_realfp.json",
            },
            features={
                "question": datasets.Value("string"),
                "program": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "context": datasets.Value("string"),
            },
        ),
        FermiConfig(
            name="realFP_distractor",
            version=datasets.Version("1.0.0"),
            description="A collection of 928 fermi problems and their solutions expressed in the form a program. This set contains distractor contexts.",
            urls={
                "train": "https://raw.githubusercontent.com/allenai/fermi/main/data/realFP/distractor_setting/train_distractor_realfp.json",
                "validation": "https://raw.githubusercontent.com/allenai/fermi/main/data/realFP/distractor_setting/val_distractor_realfp.json",
                "test": "https://raw.githubusercontent.com/allenai/fermi/main/data/realFP/distractor_setting/test_distractor_realfp.json",
            },
            features={
                "question": datasets.Value("string"),
                "program": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "context": datasets.Value("string"),
                "fact_transform": datasets.features.Sequence(datasets.features.Value("int32")),
            },
        ),
        FermiConfig(
            name="synthFP",
            version=datasets.Version("1.0.0"),
            description="An auxilliary set of 10000 templated fermi questions, created by the authors.",
            urls={
                "train": "https://raw.githubusercontent.com/allenai/fermi/main/data/synthFP/train_synthfp.json",
                "validation": "https://raw.githubusercontent.com/allenai/fermi/main/data/synthFP/val_synthfp.json",
                "test": "https://raw.githubusercontent.com/allenai/fermi/main/data/synthFP/test_synthfp.json",
            },
            features={
                "question": datasets.Value("string"),
                "program": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "context": datasets.Value("string"),
                "template": datasets.Value("string"),
                "hop": datasets.Value("int32"),
            },
        ),
        FermiConfig(
            name="synthFP_distractor",
            version=datasets.Version("1.0.0"),
            description="An auxilliary set of 10000 templated fermi questions, created by the authors. This set contains distractor contexts.",
            urls={
                "train": "https://raw.githubusercontent.com/allenai/fermi/main/data/synthFP/distractor_setting/train_distractor_synthfp.json",
                "validation": "https://raw.githubusercontent.com/allenai/fermi/main/data/synthFP/distractor_setting/val_distractor_synthfp.json",
                "test": "https://raw.githubusercontent.com/allenai/fermi/main/data/synthFP/distractor_setting/test_distractor_synthfp.json",
            },
            features={
                "question": datasets.Value("string"),
                "program": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "context": datasets.Value("string"),
                "template": datasets.Value("string"),
                "hop": datasets.Value("int32"),
                "fact_transform": datasets.features.Sequence(datasets.features.Value("int32")),
            },
        ),
    ]

    DEFAULT_CONFIG_NAME = "realFP"

    def _info(self):
        return datasets.DatasetInfo(
            description=f"{_DESCRIPTION}\n{self.config.description}",
            features=datasets.Features(self.config.features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = self.config.urls
        data_dir = dl_manager.download_and_extract(urls)
        print(data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir["test"], "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.loads(f.read())
            for key, row in enumerate(data):
                example = {
                    "question": row["question"],
                    "program": row["program"],
                    "answer": row["answer"],
                    "context": row["context"],
                }
                if "synth" in self.config.name:
                    example["template"] = row["template"]
                    example["hop"] = row["hop"]
                if "distractor" in self.config.name:
                    example["fact_transform"] = list(row["fact_transform"].keys())
                yield key, example
