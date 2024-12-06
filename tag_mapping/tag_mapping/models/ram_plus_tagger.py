from tag_mapping.models.ram_tagger import RAMTagger

from ram import get_transform
from ram.models import ram_plus


class RAMPlusTagger(RAMTagger):
    def _init_model(self, config) -> None:
        # override RAM model to load RAM++ model
        self._device = config["device"]

        self._model = ram_plus(
            pretrained=config["ram_pretrained_path"],
            image_size=config["ram_image_size"],
            vit=config["vit"],
        )
        self._model.to(self._device)
        self._model.eval()

        self._transform = get_transform(config["ram_image_size"])
