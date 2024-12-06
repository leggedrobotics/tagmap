from typing import Iterable

from PIL import Image


def compute_unlikely_tags_center_crop_ensemble(
    image: Image.Image,
    image_tags: Iterable[str],
    cc_proportions: Iterable[float],
    tagging_model,
) -> Iterable[str]:
    """
    Finds unlikely tags in a set of tags for an image by running the
    model on center cropped versions of the original image

    Args:
        image: original image
        image_tags: tags of the original image
        cc_proportions: list of border crop proportions
        tagging_model: tagging model

    Returns:
        set of unlikely tags
    """

    def center_crop(img, crop_border_proportion):
        assert crop_border_proportion < 0.5
        width, height = img.size
        return img.crop(
            (
                crop_border_proportion * width,
                crop_border_proportion * height,
                width * (1 - crop_border_proportion),
                height * (1 - crop_border_proportion),
            )
        )

    cc_images = [center_crop(image, ccp) for ccp in cc_proportions]

    unlikely_tags_set = set()
    for cc_image in cc_images:
        cc_image_tags = tagging_model.tag_image(cc_image)["tags"]

        unlikely_tags = [tag for tag in image_tags if tag not in cc_image_tags]
        unlikely_tags_set.update(unlikely_tags)

    return unlikely_tags_set
