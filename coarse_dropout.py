'''
Modified from albumentations.dropout.coarse_dropout
'''

import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import torch

# __all__ = ["CoarseDropout"]


def cutout_holes(
    img: torch.Tensor, holes: Iterable[Tuple[int, int, int, int]], fill_value: Union[int, float] = 0
) -> torch.Tensor:
    # Make a copy of the input image since we don't want to modify it directly
    img = img.clone()
    for x1, y1, x2, y2 in holes:
        img[:, y1:y2, x1:x2] = fill_value
    return img


def cutout_circles(
    img: torch.Tensor, circles: Iterable[Tuple[int, int, int]], fill_value: Union[int, float] = 0
) -> torch.Tensor:
    if len(circles) == 0:
        return img

    _, height, width = img.shape

    # Create a meshgrid
    xx, yy = torch.meshgrid(torch.arange(0, width, device=img.device), torch.arange(0, height, device=img.device))

    for center_x, center_y, circle_radius in circles:
        distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        circle = distance <= circle_radius ** 2
        img[:, circle] = fill_value

    return img


class CoarseDropout():
    """CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.

        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        max_holes: int = 3,
        max_height: Union[int, float] = 0.7,
        max_width: Union[int, float] = 0.7,
        min_holes: Optional[int] = None,
        min_height: Optional[Union[int, float]] = None,
        min_width: Optional[Union[int, float]] = None,
        fill_value: int = -1,
        mask_fill_value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 0.7,
        fully_drop_p: float = 0.1,
        max_circles: int = 3,
        min_circles: Optional[int] = None,
        max_radius: Union[int, float] = 0.7,
        min_radius: Optional[Union[int, float]] = None,
        p_circle: float = 0.5,
    ):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

        self.max_circles = max_circles
        self.min_circles = min_circles if min_circles is not None else max_circles
        self.max_radius = max_radius
        self.min_radius = min_radius if min_radius is not None else max_radius

        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))

        self.check_range(self.max_height)
        self.check_range(self.min_height)
        self.check_range(self.max_width)
        self.check_range(self.min_width)

        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

        self.always_apply = always_apply
        self.p = p
        self.fully_drop_p = fully_drop_p
        self.p_circle = p_circle
        if not self.p + self.fully_drop_p <= 1:
            raise ValueError("Invalid combination of p and fully_drop_p, their sum should be minus 1. Got: {}".format([p, fully_drop_p]))

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
            )

    def apply(
        self,
        img: torch.Tensor,
        fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        circles: Iterable[Tuple[int, int, int]] = (),
    ) -> torch.Tensor:
        hole_cutout_img = cutout_holes(img, holes, fill_value)
        circle_cutout_img = cutout_circles(hole_cutout_img, circles, fill_value)
        return circle_cutout_img

    def apply_to_mask(
        self,
        img: torch.Tensor,
        mask_fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = ()
    ) -> torch.Tensor:
        if mask_fill_value is None:
            return img
        return cutout_holes(img, holes, mask_fill_value)

    def get_params_dependent_on_targets(self, img: torch.Tensor):
        # img = params["image"]
        b, _, height, width = img.shape

        holes_list = []
        for _b in range(b):
            holes = []
            for _n in range(random.randint(self.min_holes, self.max_holes)):
                if all(
                    [
                        isinstance(self.min_height, int),
                        isinstance(self.min_width, int),
                        isinstance(self.max_height, int),
                        isinstance(self.max_width, int),
                    ]
                ):
                    hole_height = random.randint(self.min_height, self.max_height)
                    hole_width = random.randint(self.min_width, self.max_width)
                elif all(
                    [
                        isinstance(self.min_height, float),
                        isinstance(self.min_width, float),
                        isinstance(self.max_height, float),
                        isinstance(self.max_width, float),
                    ]
                ):
                    hole_height = int(height * random.uniform(self.min_height, self.max_height))
                    hole_width = int(width * random.uniform(self.min_width, self.max_width))
                else:
                    raise ValueError(
                        "Min width, max width, \
                        min height and max height \
                        should all either be ints or floats. \
                        Got: {} respectively".format(
                            [
                                type(self.min_width),
                                type(self.max_width),
                                type(self.min_height),
                                type(self.max_height),
                            ]
                        )
                    )

                y1 = random.randint(0, height - hole_height)
                x1 = random.randint(0, width - hole_width)
                y2 = y1 + hole_height
                x2 = x1 + hole_width
                holes.append((x1, y1, x2, y2))
            holes_list.append(holes)

        min_edge = min(height, width)
        circles_list = []
        for _b in range(b):
            circles = []
            for _n in range(random.randint(self.min_circles, self.max_circles)):
                if all(
                    [
                        isinstance(self.max_radius, int),
                        isinstance(self.min_radius, int),
                    ]
                ):
                    circle_radius = random.randint(self.min_radius, self.max_radius)
                elif all(
                    [
                        isinstance(self.max_radius, float),
                        isinstance(self.min_radius, float),
                    ]
                ):
                    circle_radius = int(min_edge * random.uniform(self.min_radius, self.max_radius))
                else:
                    raise ValueError(
                        "Min radius and max radius \
                        should all either be ints or floats. \
                        Got: {} respectively".format(
                            [
                                type(self.min_radius),
                                type(self.max_radius),
                            ]
                        )
                    )

                center_x, center_y = random.randint(0, width), random.randint(0, height)
                circles.append((center_x, center_y, circle_radius))

            circles_list.append(circles)

        return {"holes_list": holes_list, "circles_list": circles_list}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
            "always_apply",
            "p",
            "fully_drop_p",
            "max_circles",
            "min_circles",
            "max_radius",
            "min_radius",
            "p_circle",
        )

    def __call__(self, img):
        params = self.get_params_dependent_on_targets(img)
        holes_list = params["holes_list"]
        circles_list = params["circles_list"]

        for b in range(len(holes_list)):
            r = random.random()
            if self.always_apply or r < self.p + self.fully_drop_p:
                if r < self.p:
                    r_circle = random.random()
                    if r_circle > self.p_circle:
                        circles_list[b] = []
                    img[b, :, :, :] = self.apply(img[b, :, :, :], fill_value=self.fill_value, holes=holes_list[b], circles=circles_list[b])
                    # img = self.apply_to_mask(img, mask_fill_value=self.mask_fill_value)
                    # img = self.apply_to_keypoints(img)
                elif r > self.p and r < self.p + self.fully_drop_p:
                    img[b, :, :, :] = torch.ones_like(img[b, :, :, :]) * self.fill_value
        return img
