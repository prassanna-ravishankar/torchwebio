import urllib.request
from typing import List, Optional, Tuple

import timm
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn, no_grad, topk
from torchvision.transforms import Compose


class ImageClassificationModel:
    """
    An adapter class to expose all Image classification functionality
    for models that follow the interface of pytorch-image-models (timm).

    In a nutshell, this class exposes the following functionality :
    pre-process, forward and post-process.
    The process method does all three at a go, and is pretty much the only
    method that needs to be called externally.

    Additionally, there are some helper static methods that are
    grouped here only for encapsulation purposes

    """

    def __init__(
        self,
        model_name: Optional[str] = "",
        number_of_results: Optional[int] = 5,
        model: Optional[nn.Module] = None,
        category_list: Optional[str] = None,
    ):
        """Initialisation of an image classification model

        Parameters
        ----------
        model_name : Optional[str], optional
            timm model string to load. Only required if the model
            parameter is not None, by default ""
        number_of_results : int, optional
            Number of results to return. K in topK if this is treated as a
            retrieval problem, by default 5
        model : nn.Module, optional
            Pytorch image classification model. If this is passed,
            model_name is ignored, by default None
        category_list : str, optional
            URL to pull down the category list from, by default None
        """
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
        """Pull down model from TIMM's model hub, in eval mode

        Parameters
        ----------
        model_name : str
            model_name as accepted by `timm.create_model(..)`

        Returns
        -------
        nn.Module
            Pytorch model in evaluation mode
        """
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        return model

    def _prepare_transformation(self) -> Compose:
        """Prepare transformations for the corresponding model

        Returns
        -------
        Compose
            List of transformations to apply to
            the image, as a pytorch transformers compose object
        """
        config = resolve_data_config({}, model=self._model)
        return create_transform(**config)

    @staticmethod
    def _get_categories(
        category_list="https://raw.githubusercontent.com/"
        "pytorch/hub/master/imagenet_classes.txt",
    ):
        """Get category lists from a remote URL.

        Parameters
        ----------
        category_list : str, optional
            URL for caegory lists.
            Imagenet by default,
            by default
            "https://raw.githubusercontent.com/"
            "pytorch/hub/master/imagenet_classes.txt"

        Returns
        -------
        List[str]
            Ordered list of categories where the
            index is the category index.
        """
        filename, _ = urllib.request.urlretrieve(category_list)
        with open(filename, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        return categories

    def pre_process(self, img: Image) -> Tensor:
        """Preprocess the image. Basically processes the image
        through the transformations.
        Assumes batch size = 1

        Parameters
        ----------
        img : Image
            Image to pre-process.

        Returns
        -------
        Tensor
            pre-processed image as a tensor
        """
        return self._transformation(img).unsqueeze(0)

    def post_process(self, probabilities: Tensor) -> List[Tuple[str, float]]:
        """Post process the soft-max output of the model.
        Extracts top-K results from the last layer,
         and maps them to the corresponding categories.

        Parameters
        ----------
        probabilities : Tensor
            Tensor of probabilities (output of the network)

        Returns
        -------
        List[Tuple[str, float]]
            List of pairs of Category label and category score
        """
        top5_prob, top5_catid = topk(probabilities, self._number_of_results)
        return [
            (self._categories[top5_catid[i]], top5_prob[i].item())
            for i in range(top5_prob.size(0))
        ]

    def forward(self, img_tensor: Tensor) -> Tensor:
        """Forward pass of the pytorch model

        Parameters
        ----------
        img_tensor : Tensor
            input image after pre-processing.

        Returns
        -------
        Tensor
            Output probabilities from the network
        """
        with no_grad():
            out = self._model(img_tensor)
        return nn.functional.softmax(out[0], dim=0)

    def process_img(self, img) -> List[Tuple[str, float]]:
        """Processes a single image. Calls
        pre_process, forward and post_process in succession

        Parameters
        ----------
        img : PIL.Image
            Input image as a PIL object

        Returns
        -------
        List[Tuple[str, float]]
            Output scores and categories
        """
        img_tensor = self.pre_process(img)
        probabilities = self.forward(img_tensor)
        return self.post_process(probabilities)


# TODO: Verify whether this is required.
MODEL_NAMES = [
    {"name": "Adversarial Inception v3", "model_name": "adv_inception_v3"},
    {"name": "AdvProp (EfficientNet)", "model_name": "tf_efficientnet_b0_ap"},
    {"name": "Big Transfer (BiT)", "model_name": "resnetv2_101x1_bitm"},
    {"name": "CSP-DarkNet", "model_name": "cspdarknet53"},
    {"name": "CSP-ResNet", "model_name": "cspresnet50"},
    {"name": "CSP-ResNeXt", "model_name": "cspresnext50"},
    {"name": "DenseNet", "model_name": "densenet121"},
]
