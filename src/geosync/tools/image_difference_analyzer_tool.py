import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import zipfile
import shutil

from pydantic import BaseModel
from typing import Type
from crewai.tools import BaseTool
from pathlib import Path
from typing import List

class ImageDiffInput(BaseModel):
    image_path_1: str
    image_path_2: str

class ImageDifferenceAnalyzerTool(BaseTool):
    name: str = "Satellite Image Difference Analyzer"
    description: str = "Compares two satellite images and produces a visual difference raster."
    args_schema: Type[BaseModel] = ImageDiffInput

    def extract_tifs_if_zip(self, path: str) -> List[str]:
        """Extrai ficheiros .tif se o input for um ficheiro .zip. 
        Caso contrário, devolve o próprio caminho se for um .tif."""
        if zipfile.is_zipfile(path):
            extracted_paths = []
            output_dir = Path("temp_extracted") / Path(path).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            for file in output_dir.glob("*.tif"):
                extracted_paths.append(str(file.resolve()))

            if not extracted_paths:
                raise ValueError(f"Nenhum ficheiro .tif encontrado no zip: {path}")

            return extracted_paths
        else:
            if path.lower().endswith(".tif"):
                return [path]
            else:
                raise ValueError(f"Formato não suportado: {path}")

    def find_band(self, paths: List[str], band_id: str) -> str:
        for path in paths:
            name = Path(path).name.lower()
            if (
                f"_b{band_id}.tif" in name or
                f".b{band_id}.tif" in name or
                f"b{band_id}.tif" in name or
                f"_b{int(band_id):02}.tif" in name or
                f"b{int(band_id):02}.tif" in name
                ):
                return path
        raise ValueError(f"Banda B{band_id} não encontrada.")

    def cleanup_temp_extracted(self):
        temp_dir = Path("temp_extracted")
        if temp_dir.exists() and temp_dir.is_dir():
            shutil.rmtree(temp_dir)

    def analyze_difference(self, image_path_1: str, image_path_2: str) -> str:

        try:
            tifs_1 = self.extract_tifs_if_zip(image_path_1)
            tifs_2 = self.extract_tifs_if_zip(image_path_2)

            # Procurar bandas RGB e NIR
            bands_1 = {
                "B2": self.find_band(tifs_1, "2"),  # Azul
                "B3": self.find_band(tifs_1, "3"),  # Verde
                "B4": self.find_band(tifs_1, "4"),  # Vermelho
                "B8": self.find_band(tifs_1, "8"),  # Infravermelho próximo
            }

            bands_2 = {
                "B2": self.find_band(tifs_2, "2"),
                "B3": self.find_band(tifs_2, "3"),
                "B4": self.find_band(tifs_2, "4"),
                "B8": self.find_band(tifs_2, "8"),  # Infravermelho próximo
            }

            os.makedirs("output", exist_ok=True)

            # Função para carregar uma banda e normalizar
            def read_normalized(path):
                with rasterio.open(path) as src:
                    arr = src.read(1).astype(np.float32)
                    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

            # Criar imagem RGB antiga (image 1)
            rgb1 = np.stack([read_normalized(bands_1["B4"]),
                            read_normalized(bands_1["B3"]),
                            read_normalized(bands_1["B2"])], axis=-1)
            plt.imsave("output/rgb_antiga.png", rgb1)

            # Criar imagem RGB recente (image 2)
            rgb2 = np.stack([read_normalized(bands_2["B4"]),
                            read_normalized(bands_2["B3"]),
                            read_normalized(bands_2["B2"])], axis=-1)
            plt.imsave("output/rgb_recente.png", rgb2)

            # Criar imagem NIR antiga
            nir_old = read_normalized(bands_1["B8"])
            plt.imsave("output/nir_antiga.png", nir_old, cmap="gray")

            # Criar imagem NIR recente
            nir_recent = read_normalized(bands_2["B8"])
            plt.imsave("output/nir_recente.png", nir_recent, cmap="gray")

            def calculate_ndvi(red_band_src, nir_band_src):
                # Calcular NDVI: (B8 - B4) / (B8 + B4)
                with rasterio.open(nir_band_src) as nir_src, rasterio.open(red_band_src) as red_src:
                    nir_band = nir_src.read(1).astype(np.float32)
                    red_band = red_src.read(1).astype(np.float32)

                    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-9)
                    ndvi_normalized = (ndvi + 1) / 2  # para valores entre 0 e 1

                    meta = red_src.meta.copy()
                    meta.update(count=1)

                    #plt.imsave("output/ndvi.png", ndvi_normalized, cmap="RdYlGn")
                    return ndvi_normalized, meta

            def calculate_ndbi(swir_band_src, nir_band_src):
                with rasterio.open(swir_band_src) as swir_src, rasterio.open(nir_band_src) as nir_src:
                    swir_band = swir_src.read(1).astype(np.float32)
                    nir_band = nir_src.read(1).astype(np.float32)
                    ndbi = (swir_band - nir_band) / (swir_band + nir_band + 1e-9)
                    ndbi_normalized = (ndbi + 1) / 2
                    meta = swir_src.meta.copy()
                    meta.update(count=1)
                    return ndbi_normalized, meta

            ndvi_old, meta_old = calculate_ndvi(red_band_src=bands_1["B4"], nir_band_src=bands_1["B8"])
            ndvi_recent, meta_recent = calculate_ndvi(red_band_src=bands_2["B4"], nir_band_src=bands_2["B8"])

            # Calcular a diferença
            ndvi_diff = ndvi_recent - ndvi_old

            # Guardar o resultado da diferença ndvi como png
            plt.imsave("output/ndvi_diff.png", ndvi_diff, cmap="RdYlGn")

            # Guardar o resultado da diferença ndvi como tif
            with rasterio.open("output/ndvi_diff.tif", "w", **meta_old) as dst:
                dst.write(ndvi_diff.astype(rasterio.float32), 1)

            return "output/ndvi_diff.tif"  # ou outro ficheiro que o agente vá usar por padrão
        except Exception as e:
            raise Exception(f"Erro ao analisar diferenças entre as imagens: {str(e)}")
        finally:
            self.cleanup_temp_extracted()

    def _run(self, image_path_1: str, image_path_2: str) -> str:
        return self.analyze_difference(image_path_1, image_path_2)