#import torch
import numpy as np
import cv2
import os
import tensorflow as tf
import segmentation_models as sm

from PIL import Image
from pydantic import BaseModel, PrivateAttr
from crewai.tools import BaseTool
#from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class UrbanGrowthInput(BaseModel):
    image_path_1: str
    image_path_2: str

class UrbanGrowthAnalyzerTool(BaseTool):
    name: str = "Urban Growth Analyzer"
    description: str = "Detects and counts new buildings between two images using a Keras trained Unet model."
    args_schema: type = UrbanGrowthInput

    _model: any = PrivateAttr()
    _preprocess: any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Carrega o modelo Keras
        self._model = tf.keras.models.load_model(
            "tools/my_building_segmentation_model.h5",
            custom_objects={
                "bce_jaccard_loss": sm.losses.bce_jaccard_loss,
                "iou_score": sm.metrics.iou_score
            }
        )
        # Define o preprocessamento igual ao usado no treino
        self._preprocess = sm.get_preprocessing('resnet34')


    def segment_buildings(self, image_path):
        # Carrega e pré-processa a imagem
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        img_np = np.array(img)
        img_np = np.expand_dims(img_np, axis=0)
        img_np = self._preprocess(img_np)
        # Faz a predição
        pred_mask = self._model.predict(img_np)[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        return pred_mask

    def _run(self, image_path_1: str, image_path_2: str) -> str:
        mask1 = self.segment_buildings(image_path_1)
        mask2 = self.segment_buildings(image_path_2)

        # Supondo que sabes o tamanho original (podes obter via PIL ou passar como argumento)
        # original_size = (101, 101)

        # # Redimensiona as máscaras preditas
        # mask1_resized = cv2.resize(mask1, original_size, interpolation=cv2.INTER_NEAREST)
        # mask2_resized = cv2.resize(mask2, original_size, interpolation=cv2.INTER_NEAREST)

        # Conta polígonos (construções) usando OpenCV
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