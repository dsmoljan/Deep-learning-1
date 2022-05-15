from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
    "Class which represents one training instance of data"
    token_list: List[str]
    label: str
