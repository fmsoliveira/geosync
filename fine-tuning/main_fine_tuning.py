from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from pathlib import Path
from PIL import Image
from transformers.onnx import export

import numpy as np
import os
import torch
import torch.nn as nn

# Configuração do dispositivo - verifica se MPS está disponível no Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Usando MPS (Metal Performance Shaders) no Apple Silicon")
else:
    device = torch.device("cpu")
    print("MPS não disponível, usando CPU")

def safe_cross_entropy(input, target, **kwargs):
    """
    Implementação segura de cross entropy que garante tensores contíguos
    e usa reshape em vez de view para evitar problemas de stride
    """
    # Garante que os tensores estão contíguos
    input = input.contiguous()
    target = target.contiguous()
    
    # Usa reshape em vez de view - stride error
    input = input.permute(0, 2, 3, 1).reshape(-1, input.shape[1])
    target = target.reshape(-1)
    
    return nn.functional.cross_entropy(input, target, **kwargs)

class SegformerWithSafeLoss(SegformerForSemanticSegmentation):
    """
    Modelo SegFormer personalizado com função de perda segura
    para lidar com problemas de tensores não contíguos
    """
    def compute_loss(self, inputs, return_outputs=False):
        outputs = self(**inputs)
        logits = outputs.logits
        
        # Garantir tensores contíguos
        logits = logits.contiguous()
        labels = inputs["labels"].contiguous()
        
        loss = safe_cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Definições principais do modelo
NUM_CLASSES = 2  # Fundo (0) e edifício (1)
IMG_SIZE = (512, 512)  # Tamanho para redimensionar imagens e máscaras

# Carrega modelo e processor
model = SegformerWithSafeLoss.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
# Move o modelo para o dispositivo adequado (MPS ou CPU)
model = model.to(device)

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

def preprocess(example):
    """
    Pré-processamento de um único exemplo (imagem + máscara)
    """
    image = Image.open(example["image"]).convert("RGB")
    mask = Image.open(example["mask"])

    # Garante que a máscara é Grayscale (single channel)
    if mask.mode != "L":
        mask = mask.convert("L")

    mask_np = np.array(mask)

    # Mapeia os valores da máscara (assumindo 0=fundo, 255=edifício)
    building_class_value = 1  # O índice que queremos para 'edifício'
    original_building_value = 255  # O valor usado no ficheiro da máscara
    mask_np[mask_np == original_building_value] = building_class_value

    # Verificação opcional de valores únicos na máscara
    # print(f"Valores únicos na máscara {os.path.basename(example['mask'])}: {np.unique(mask_np)}")

    # Converte para o tipo correto antes de passar para o processor
    mask_np = mask_np.astype(np.int64)

    # O processor lida com redimensionamento e normalização
    processed = processor(image, segmentation_maps=mask_np, return_tensors="pt")

    # Remove a dimensão batch adicionada pelo processor e garante tensores contíguos
    return {
        "pixel_values": processed["pixel_values"][0].float().contiguous(),
        "labels": processed["labels"][0].long().contiguous()
    }

def load_images_and_masks(img_dir, mask_dir, img_size=None):
    """
    Carrega os caminhos das imagens e das máscaras
    """
    X, Y = [], []
    for fname in os.listdir(img_dir):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        # Verifica se ambos os arquivos existem
        if os.path.exists(img_path) and os.path.exists(mask_path):
            X.append(img_path)
            Y.append(mask_path)
        else:
            print(f"Aviso: Imagem ou máscara não encontrada para {fname}")
            
    print(f"Carregados {len(X)} pares de imagem/máscara de {img_dir} e {mask_dir}")
    return X, Y

def load_examples(img_dir, mask_dir, img_size):
    """
    Prepara os dados no formato esperado pelo Dataset
    """
    x, y = load_images_and_masks(img_dir, mask_dir, img_size)
    data = []
    for img_path, mask_path in zip(x, y):
        data.append({"image": img_path, "mask": mask_path})
    return data

def collate_fn(batch):
    """
    Função de coleta para o DataLoader que garante que todos os tensores
    sejam do tipo correto e contíguos
    """
    def ensure_tensor(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(dtype).contiguous()
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype).contiguous()
        else:
            return torch.tensor(x, dtype=dtype).contiguous()
    
    pixel_values = torch.stack([ensure_tensor(x["pixel_values"], torch.float32) for x in batch])
    labels = torch.stack([ensure_tensor(x["labels"], torch.int64) for x in batch])
    
    return {
        "pixel_values": pixel_values, 
        "labels": labels
    }

def main():
    # Carrega os dados
    train_data = load_examples('data/train/images', 'data/train/gt', IMG_SIZE)
    val_data = load_examples('data/val/images', 'data/val/gt', IMG_SIZE)

    # Cria datasets Hugging Face
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

    # Aplica preprocessamento
    dataset = dataset.map(preprocess)

    # Argumentos de treino
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        save_steps=100,
        save_total_limit=3,
        logging_dir="./logs",
        logging_steps=10,
        # evaluation_strategy="steps",
        # eval_steps=100,
        no_cuda=True,
        dataloader_pin_memory=False,  # Desativa pin_memory que não é suportado no MPS
        # report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=collate_fn,
    )

    # Treina o modelo
    print("Iniciando treinamento...")
    trainer.train()
    print("Treinamento concluído!")

    # Guardar o modelo treinado
    model.save_pretrained("./modelo_segformer_finalizado")
    processor.save_pretrained("./modelo_segformer_finalizado")
    print("Modelo salvo em ./modelo_segformer_finalizado")

if __name__ == "__main__":
    main()