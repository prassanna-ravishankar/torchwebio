import urllib.request
from typing import List, Optional, Tuple

import timm
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn, no_grad, topk
from torchvision.transforms import Compose


class ImageClassificationModel:
    def __init__(
        self,
        model_name: Optional[str] = "",
        number_of_results: Optional[int] = 5,
        model: Optional[nn.Module] = None,
        category_list: Optional[str] = None,
    ):
        self._number_of_results = number_of_results

        if not model:
            self._model = self._get_model(model_name)
        else:
            self._model = model
        self._transformation = self._prepare_transformation()
        if category_list:
            self._categories = self._get_categories(category_list)
        else:
            self._categories = self._get_categories()

    @staticmethod
    def _get_model(model_name: str) -> nn.Module:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        return model

    def _prepare_transformation(self) -> Compose:
        config = resolve_data_config({}, model=self._model)
        return create_transform(**config)

    @staticmethod
    def _get_categories(
        category_list="https://raw.githubusercontent.com/"
        "pytorch/hub/master/imagenet_classes.txt",
    ):
        filename, _ = urllib.request.urlretrieve(category_list)
        with open(filename, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        return categories

    def pre_process(self, img: Image) -> Tensor:
        return self._transformation(img).unsqueeze(0)

    def post_process(self, probabilities: Tensor) -> List[Tuple[str, float]]:
        top5_prob, top5_catid = topk(probabilities, self._number_of_results)
        return [
            (self._categories[top5_catid[i]], top5_prob[i].item())
            for i in range(top5_prob.size(0))
        ]

    def forward(self, img_tensor: Tensor) -> Tensor:
        with no_grad():
            out = self._model(img_tensor)
        return nn.functional.softmax(out[0], dim=0)

    def process_img(self, img) -> List[Tuple[str, float]]:
        img_tensor = self.pre_process(img)
        probabilities = self.forward(img_tensor)
        return self.post_process(probabilities)


MODEL_NAMES = [
    {"name": "Adversarial Inception v3", "model_name": "adv_inception_v3"},
    {"name": "AdvProp (EfficientNet)", "model_name": "tf_efficientnet_b0_ap"},
    {"name": "Big Transfer (BiT)", "model_name": "resnetv2_101x1_bitm"},
    {"name": "CSP-DarkNet", "model_name": "cspdarknet53"},
    {"name": "CSP-ResNet", "model_name": "cspresnet50"},
    {"name": "CSP-ResNeXt", "model_name": "cspresnext50"},
    {"name": "DenseNet", "model_name": "densenet121"},
]
