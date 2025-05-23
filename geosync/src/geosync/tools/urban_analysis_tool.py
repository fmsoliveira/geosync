import numpy as np
import cv2
import os
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel, PrivateAttr
from crewai.tools import BaseTool

class UrbanGrowthInput(BaseModel):
    image_path_1: str
    image_path_2: str

class UrbanGrowthAnalyzerTool(BaseTool):
    name: str = "Urban Growth Analyzer"
    description: str = "Detects and counts new buildings between two satellite images using a SegFormer model."
    args_schema: type = UrbanGrowthInput
    
    _model: any = PrivateAttr()
    _input_name: str = PrivateAttr()
    _output_name: str = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Carregar o modelo ONNX
        model_path = os.path.join(os.path.dirname(__file__), "../models/segformer_dynamic.onnx")
        self._model = ort.InferenceSession(model_path)
        self._input_name = self._model.get_inputs()[0].name
        self._output_name = self._model.get_outputs()[0].name

    def preprocess_image(self, image_path):
        # Carregar a imagem
        img = Image.open(image_path).convert("RGB")
        img = img.resize((512, 512))  # Ajuste para o tamanho esperado pelo SegFormer
        
        img_np = np.array(img, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)

        img_np = img_np.astype(np.float32)

        return img_np, img.size     

    def segment_buildings(self, image_path):
        # Pré-processar a imagem
        input_data, original_size = self.preprocess_image(image_path)
        
        # Executar inferência com o modelo ONNX
        outputs = self._model.run([self._output_name], {self._input_name: input_data})
        
        # Processar a saída para obter a máscara binária de construções
        logits = outputs[0]
        
        if len(logits.shape) == 4:  # [batch, num_classes, height, width]
            building_class_index = 0 
            pred_mask = logits[0, building_class_index]
        else:  # Se já for uma máscara binária
            pred_mask = logits[0, 0]
        
        # Converter para máscara binária e redimensionar para 256x256
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return pred_mask

    def _run(self, image_path_1: str, image_path_2: str) -> str:
        mask1 = self.segment_buildings(image_path_1)
        mask2 = self.segment_buildings(image_path_2)

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
        os.makedirs("output", exist_ok=True)
        diff_img_path = os.path.join(os.getcwd(), "output", "new_buildings_diff.png")
        Image.fromarray((diff_mask * 255).astype(np.uint8)).save(diff_img_path)

        # Guardar imagens originais com contornos desenhados
        orig1 = cv2.imread(image_path_1)
        orig2 = cv2.imread(image_path_2)
        # Redimensionar para 256x256 para coincidir com a máscara
        orig1 = cv2.resize(orig1, (256, 256))
        orig2 = cv2.resize(orig2, (256, 256))
        # Desenhar contornos a vermelho
        cv2.drawContours(orig1, contours1, -1, (0, 0, 255), 2)
        cv2.drawContours(orig2, contours2, -1, (0, 0, 255), 2)
        buildings_img1_path = os.path.join(os.getcwd(), "output", "buildings_detected_first_image.png")
        buildings_img2_path = os.path.join(os.getcwd(), "output", "buildings_detected_second_image.png")
        cv2.imwrite(buildings_img1_path, orig1)
        cv2.imwrite(buildings_img2_path, orig2)

        return {
            "Edifícios na data 1": n_buildings_1,
            "Edifícios na data 2": n_buildings_2,
            "Novas construções": new_buildings,
            "Imagem de diferença guardada em": diff_img_path,
            "Edifícios identificados na primeira imagem": buildings_img1_path,
            "Edifícios identificados na segunda imagem": buildings_img2_path
        }