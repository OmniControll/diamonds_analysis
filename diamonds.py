"""Diamond Dataset"""

from typing import List
from functools import partial

import datasets

import pandas


VERSION = datasets.Version("1.0.0")
_BASE_FEATURE_NAMES = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "price",
    "observation_point_on_axis_x",
    "observation_point_on_axis_y",
    "observation_point_on_axis_z"
]

_ENCODING_DICS = {
    "cut": {
        "Fair": 0,
        "Good": 1,
        "Very Good": 2,
        "Premium": 3,
        "Ideal": 4
    },
    "clarity": {
        "IF": 0,
        "VVS1": 1,
        "VVS2": 2,
        "VS1": 3,
        "VS2": 4,
        "SI1": 5,
        "SI2": 6,
        "I1": 7
    }
}

DESCRIPTION = "Diamond quality dataset."
_HOMEPAGE = "https://www.kaggle.com/datasets/ulrikthygepedersen/diamonds"
_URLS = ("https://www.kaggle.com/datasets/ulrikthygepedersen/diamonds")
_CITATION = """"""

# Dataset info
urls_per_split = {
    "train": "https://huggingface.co/datasets/mstz/diamonds/raw/main/diamonds.csv",
}
features_types_per_config = {
    "encoding": {
        "feature": datasets.Value("string"),
        "original_value": datasets.Value("string"),
        "encoded_value":  datasets.Value("int8"),
    },
    
    "cut": {
        "carat": datasets.Value("float32"),
        "color": datasets.Value("string"),
        "clarity": datasets.Value("float32"),
        "depth": datasets.Value("float32"),
        "table": datasets.Value("float32"),
        "price": datasets.Value("float32"),
        "observation_point_on_axis_x": datasets.Value("float32"),
        "observation_point_on_axis_y": datasets.Value("float32"),
        "observation_point_on_axis_z": datasets.Value("float32"),
        "cut": datasets.ClassLabel(num_classes=5, names=("Fair", "Good", "Very Good", "Premium", "Ideal"))
    },

    "cut_binary": {
        "carat": datasets.Value("float32"),
        "color": datasets.Value("string"),
        "clarity": datasets.Value("float32"),
        "depth": datasets.Value("float32"),
        "table": datasets.Value("float32"),
        "price": datasets.Value("float32"),
        "observation_point_on_axis_x": datasets.Value("float32"),
        "observation_point_on_axis_y": datasets.Value("float32"),
        "observation_point_on_axis_z": datasets.Value("float32"),
        "cut": datasets.ClassLabel(num_classes=2, names=("no", "yes"))
    },
}
features_per_config = {k: datasets.Features(features_types_per_config[k]) for k in features_types_per_config}


class DiamondConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(DiamondConfig, self).__init__(version=VERSION, **kwargs)
        self.features = features_per_config[kwargs["name"]]


class Diamond(datasets.GeneratorBasedBuilder):
    # dataset versions
    DEFAULT_CONFIG = "cut"
    BUILDER_CONFIGS = [
        DiamondConfig(name="encoding", description="Encoding dictionaries for discrete features."),
        DiamondConfig(name="cut", description="5-ary classification, predict the cut quality of the diamond."),
        DiamondConfig(name="cut_binary", description="Binary classification."),
    ]


    def _info(self):
        info = datasets.DatasetInfo(description=DESCRIPTION, citation=_CITATION, homepage=_HOMEPAGE,
                                    features=features_per_config[self.config.name])

        return info
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloads = dl_manager.download_and_extract(urls_per_split)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloads["train"]}),
        ]
    
    def _generate_examples(self, filepath: str):
        if self.config.name == "encoding":
            data = self.encoding_dics()
        else:
            data = pandas.read_csv(filepath)
            data = self.preprocess(data, config=self.config.name)

        for row_id, row in data.iterrows():
            data_row = dict(row)

            yield row_id, data_row

    def preprocess(self, data: pandas.DataFrame, config: str = "cut") -> pandas.DataFrame:
        data["clarity"] = data.clarity.apply(lambda x: x.replace("b", "").replace("'", ""))
        data["cut"] = data.cut.apply(lambda x: x.replace("b", "").replace("'", ""))
        data["color"] = data.color.astype(str)
        data["color"] = data.color.apply(lambda x: x[2]).replace("\"", "")

        for feature in _ENCODING_DICS:
            encoding_function = partial(self.encode, feature)
            data[feature] = data[feature].apply(encoding_function)
         
        data.columns = _BASE_FEATURE_NAMES
        data = data.drop_duplicates(subset=["carat", "color", "clarity", "depth", "table", "price", "cut"])

        if self.config.name == "cut_binary":
            data.cut = data.cut.apply(lambda x: 0 if x <= 2 else 1)            
        
        
        return data[list(features_types_per_config["cut"].keys())]

    def encode(self, feature, value):
        if feature in _ENCODING_DICS:
            return _ENCODING_DICS[feature][value]
        raise ValueError(f"Unknown feature: {feature}")

    def encoding_dics(self):
        data = [pandas.DataFrame([(feature, original, encoded) for original, encoded in d.items()])
                for feature, d in _ENCODING_DICS.items()]
        data = pandas.concat(data, axis="rows").reset_index()
        data.drop("index", axis="columns", inplace=True)
        data.columns = ["feature", "original_value", "encoded_value"]

        return data
