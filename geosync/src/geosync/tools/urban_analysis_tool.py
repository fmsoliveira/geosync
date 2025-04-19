import torch
import numpy as np

from PIL import Image
from pydantic import BaseModel
from crewai.tools import BaseTool
from pydantic import PrivateAttr
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class UrbanGrowthInput(BaseModel):
    image_path_1: str
    image_path_2: str

class UrbanGrowthAnalyzerTool(BaseTool):
    name: str = "Urban Growth Analyzer"
    description: str = "Detects and counts new buildings between two Sentinel-2 images using torchvision DeepLabV3."
    args_schema: type = UrbanGrowthInput

    _model: any = PrivateAttr()
    _preprocess: any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocess = AutoImageProcessor.from_pretrained(
            "yeray142/finetune-instance-segmentation-ade20k-mini-mask2former_backbone_frozen_1"
        )
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "yeray142/finetune-instance-segmentation-ade20k-mini-mask2former_backbone_frozen_1"
        )


    def segment_buildings(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self._preprocess(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (batch, num_classes, H, W)
            pred_seg = logits.argmax(dim=1)[0].cpu().numpy()
            buildings_mask = (pred_seg == 12)  # ADE20K: 12 = building
            return buildings_mask

    def _run(self, image_path_1: str, image_path_2: str) -> str:
        mask1 = self.segment_buildings(image_path_1)
        mask2 = self.segment_buildings(image_path_2)

        # Conta polígonos (construções) usando OpenCV
        import cv2, os
        mask1_uint8 = (mask1 * 255).astype(np.uint8)
        mask2_uint8 = (mask2 * 255).astype(np.uint8)
        contours1, _ = cv2.findContours(mask1_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n_buildings_1 = len(contours1)
        n_buildings_2 = len(contours2)
        new_buildings = n_buildings_2 - n_buildings_1

        # Imagem de diferença
        diff_mask = (mask2.astype(int) - mask1.astype(int)) > 0
        #diff_img_path = "output/new_buildings_diff.png"
        diff_img_path = os.path.join(os.getcwd(), "output", "new_buildings_diff.png")
        os.makedirs("output", exist_ok=True)
        Image.fromarray((diff_mask * 255).astype(np.uint8)).save(diff_img_path)

        return {
            "Edifícios na data 1": n_buildings_1,
            "Edifícios na data 2": n_buildings_2,
            "Novas construções": new_buildings,
            "Imagem de diferença guardada em": diff_img_path
        }