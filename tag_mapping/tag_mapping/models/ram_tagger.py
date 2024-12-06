import numpy as np

from typing import Dict, List, Tuple, Any
from PIL import Image

from ram import get_transform, inference_ram
from ram.models import ram

from tag_mapping.models.image_tagger import ImageTagger
from tag_mapping.filtering import compute_unlikely_tags_center_crop_ensemble


class RAMTagger(ImageTagger):
    """
    Wrapper for the Recognize-Anything tagging model
    """

    def __init__(self, config) -> None:
        self._init_model(config)

    def _init_model(self, config) -> None:
        self._device = config["device"]

        self._model = ram(
            pretrained=config["ram_pretrained_path"],
            image_size=config["ram_image_size"],
            vit=config["vit"],
        )
        self._model.to(self._device)
        self._model.eval()

        self._transform = get_transform(config["ram_image_size"])

    def tag_image(self, image: Image.Image) -> Tuple[List, List]:
        """
        Forwards the tagging model and returns the tags and confidences
        """
        tags, confidences = inference_ram(
            self._transform(image).unsqueeze(0).to(self._device), self._model
        )
        tags = tags.split(" | ")
        return {"tags": tags, "confidences": confidences}

    def override_class_thresholds(self, thresholds: Dict[str, float]) -> None:
        for cls, t in thresholds.items():
            try:
                self._model.override_class_threshold(cls, t)
            except Exception as e:
                print("Couldn't override threshold for {} because: {}".format(cls, e))

    def filtered_tag_image(
        self, image: Image.Image, params: Dict[str, Any]
    ) -> Tuple[List, List]:
        """
        Forwards the model and applies additional inference filtering to remove unlikely tags
        """
        out = self.tag_image(image)
        tags, confidences = (out["tags"], out["confidences"])

        # filter tags
        unlikely_tags = compute_unlikely_tags_center_crop_ensemble(
            image,
            tags,
            params["crop_border_proportions"],
            self,
        )

        filtered_tags = [tag for tag in tags if tag not in unlikely_tags]
        filtered_tag_confidences = [
            conf for tag, conf in zip(tags, confidences) if tag not in unlikely_tags
        ]
        return filtered_tags, filtered_tag_confidences
