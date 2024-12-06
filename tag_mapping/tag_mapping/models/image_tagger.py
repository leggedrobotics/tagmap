from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from PIL import Image


class ImageTagger(ABC):
    """
    Abstract base class for all image taggers
    """

    @abstractmethod
    def tag_image(self, image: Image.Image) -> Tuple[List, List]:
        """
        Forwards the tagging model and returns the tags and confidences
        """
        raise NotImplementedError

    @abstractmethod
    def filtered_tag_image(
        self, image: Image.Image, params: Dict[str, Any]
    ) -> Tuple[List, List]:
        """
        Forwards the model and applies additional inference filtering to remove unlikely tags
        """
        raise NotImplementedError
