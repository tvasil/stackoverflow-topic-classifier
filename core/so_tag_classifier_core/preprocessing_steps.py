import re
from typing import List

import numpy as np

try:
    from nltk.corpus import stopwords

    _STOPWORDS = set(stopwords.words("english"))
except:
    import nltk

    nltk.download("stopwords")
    from nltk.corpus import stopwords

    _STOPWORDS = set(stopwords.words("english"))

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

_RANDOM_STATE = 42

_TAGS_TO_KEEP = [
    "python",
    "javascript",
    "java",
    "android",
    "c#",
    "html",
    "reactjs",
    "php",
    "python3x",
    "nodejs",
    "r",
    "c++",
    "css",
    "sql",
    "flutter",
    "angular",
    "ios",
    "pandas",
    "jquery",
    "mysql",
    "swift",
    "django",
    "c",
    "typescript",
    "arrays",
    "json",
    "laravel",
    "reactnative",
    "sqlserver",
    "amazonwebservices",
    "springboot",
    "firebase",
    "docker",
    "azure",
    "dart",
    "dataframe",
    "excel",
    "kotlin",
    "linux",
    "spring",
    "vuejs",
    "postgresql",
    "wordpress",
    "numpy",
    "string",
    "tensorflow",
    "mongodb",
    "windows",
    "net",
    "vba",
    "regex",
    "aspnetcore",
    "bash",
    "androidstudio",
    "api",
    "git",
    "database",
    "powershell",
    "xcode",
    "aspnet",
    "list",
    "selenium",
    "kubernetes",
    "machinelearning",
    "visualstudiocode",
    "express",
    "rubyonrails",
    "xml",
    "netcore",
    "macos",
    "apachespark",
    "function",
    "oracle",
    "csv",
    "ajax",
    "algorithm",
    "matplotlib",
    "unity3d",
    "image",
    "googlecloudfirestore",
    "swiftui",
    "keras",
    "wpf",
    "visualstudio",
    "woocommerce",
    "dictionary",
    "amazons3",
    "datetime",
    "maven",
    "loops",
    "shell",
    "npm",
    "flask",
    "apache",
    "ruby",
    "googlechrome",
    "googlecloudplatform",
    "pyspark",
    "opencv",
    "ubuntu",
]

_REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
_BAD_SYMBOLS_RE = re.compile(r"[^0-9a-z #+_]")
_STOPWORDS = set(stopwords.words("english"))


def text_prepare(arr: np.ndarray, join_symbol: str) -> List[str]:
    """Prepare both X and Y text by lowering, removing stopwords and
    replacing special characters. Return a list of strings joined by 'join_symbol'
    """

    def process(text: str, join_symbol: str) -> str:
        """Actual processing function"""
        text = text.lower()
        text = re.sub(_REPLACE_BY_SPACE_RE, " ", text)

        text = re.sub(_BAD_SYMBOLS_RE, "", text)
        text = re.sub(r"\s+", " ", text)

        return f"{join_symbol}".join([i for i in text.split() if i not in _STOPWORDS])

    return [process(text, join_symbol) for text in arr]


def transform_y(arr: np.ndarray, join_symbol: str = ",") -> List[set]:
    """Specific function to transform the Y tags and return as a list of sets,
    the format required by MultiLabelBinarizer"""
    arr = text_prepare(arr, join_symbol=join_symbol)
    return [set(i.split(",")) for i in arr]


def binarize_ys(
    y_train: np.ndarray, y_test: np.ndarray, binarizer=MultiLabelBinarizer(classes=_TAGS_TO_KEEP)
):
    """Return binarized y_train and y_test. The default will filter the labels
    down to the 100 most popular ones (global var)"""

    y_train_binarized = binarizer.fit_transform(transform_y(y_train))
    y_test_binarized = binarizer.transform(transform_y(y_test))
    return binarizer, y_train_binarized, y_test_binarized


def tokenize_and_stem(text: str) -> List[str]:
    """Tokenizes a sentence and then stems each word in the sentence.
    Returns a list of tokens/stems"""
    tokenized_list = word_tokenize(text)
    snowball = SnowballStemmer(language="english")
    return [snowball.stem(word) for word in tokenized_list]
