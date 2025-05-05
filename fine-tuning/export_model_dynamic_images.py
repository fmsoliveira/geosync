import torch
import torch.onnx
from pathlib import Path
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Caminho para o modelo salvo
model_path = "./modelo_segformer_finalizado"

model = SegformerForSemanticSegmentation.from_pretrained(model_path)
processor = SegformerImageProcessor.from_pretrained(model_path)

model.eval()

model.to('cpu')

dummy_input = {
    "pixel_values": torch.randn(1, 3, 512, 512),
}

print("Exportando modelo SegFormer para ONNX com dimensões dinâmicas...")

class SegformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs.logits

try:
    
    # Definir eixos dinâmicos para permitir diferentes tamanhos de imagens
    dynamic_axes = {
        'input': {
            0: 'batch_size',  # tamanho do batch variável
            2: 'height',      # altura variável
            3: 'width'        # largura variável
        },
        'output': {
            0: 'batch_size',  # tamanho do batch variável
            2: 'height',      # altura variável
            3: 'width'        # largura variável
        }
    }
    
    # Export para ONNX com eixos dinâmicos
    torch.onnx.export(
        model,                                 
        dummy_input["pixel_values"],           
        "segformer_dynamic.onnx",             
        export_params=True,                    
        opset_version=12,                      
        do_constant_folding=True,              
        input_names=['input'],                 
        output_names=['output'],               
        dynamic_axes=dynamic_axes  # especifica os eixos que podem mudar de tamanho
    )
    print("Modelo exportado com sucesso para segformer_dynamic.onnx")
    print("Este modelo aceita imagens de diferentes dimensões!")
    
except Exception as e:
    print(f"Erro ao exportar com dimensões dinâmicas: {e}")
    
    try:
        print("Tentando abordagem alternativa...")
        
        wrapped_model = SegformerWrapper(model)
        
        torch.onnx.export(
            wrapped_model,
            dummy_input["pixel_values"],
            "segformer_dynamic_alt.onnx",
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        print("Modelo exportado com sucesso para segformer_dynamic_alt.onnx")
        
    except Exception as e2:
        print(f"Segunda tentativa também falhou: {e2}")