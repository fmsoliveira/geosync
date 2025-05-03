import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import zipfile
import shutil
import json
from rasterio.merge import merge

from pydantic import BaseModel
from typing import Type, Dict, List, Union, Optional, Tuple
from crewai.tools import BaseTool
from pathlib import Path

class ImageDiffInput(BaseModel):
    first_date_images: Dict[str, str]
    second_date_images: Dict[str, str]

class ImageDifferenceAnalyzerTool(BaseTool):
    name: str = "Satellite Image Difference Analyzer"
    description: str = "Compares two satellite images (multiple quadrants) and produces a visual difference raster."
    args_schema: Type[BaseModel] = ImageDiffInput

    def extract_tifs_if_zip(self, path: str, prefix: str = "") -> List[str]:
        """Extrai ficheiros .tif se o input for um ficheiro .zip. 
        Caso contrário, devolve o próprio caminho se for um .tif."""
        if zipfile.is_zipfile(path):
            extracted_paths = []
            # Add prefix to avoid overwriting files from different dates
            output_dir = Path("temp_extracted") / f"{prefix}_{Path(path).stem}"
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

    def merge_rasters(self, raster_paths: List[str], output_path: str) -> str:
        """Combina múltiplos rasters em um único arquivo"""
        if len(raster_paths) == 0:
            raise ValueError("Nenhum raster para mesclar")
        
        if len(raster_paths) == 1:
            # Usar shutil.copyfile em vez de shutil.copy para copiar apenas o conteúdo
            shutil.copyfile(raster_paths[0], output_path)
            return output_path
        
        src_files_to_mosaic = []
        for raster_path in raster_paths:
            try:
                src = rasterio.open(raster_path)
                src_files_to_mosaic.append(src)
            except Exception as e:
                print(f"Erro ao abrir raster {raster_path}: {str(e)}")
        
        if not src_files_to_mosaic:
            raise ValueError("Nenhum raster pôde ser aberto para mesclagem")
        
        try:
            mosaic, out_trans = merge(src_files_to_mosaic)
            
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "driver": "GTiff"
            })
            
            # Escreve o raster unificado para o disco
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            print(f"Mescla concluída: {output_path}, shape: {mosaic.shape}, paths usados: {raster_paths}")
            
        except Exception as e:
            print(f"Erro durante a mesclagem: {str(e)}")
            raise
        finally:
            for src in src_files_to_mosaic:
                src.close()
            
        return output_path

    def extract_and_merge_bands(self, quadrant_paths: Dict[str, str], band_id: str, date_prefix: str) -> str:
        """Extrai e une uma banda específica de múltiplos quadrantes"""
        band_paths = []
        
        # Cria diretório temporário com prefixo para evitar conflitos
        temp_dir = Path(f"temp_merged_{date_prefix}")
        temp_dir.mkdir(exist_ok=True)
        
        # Para cada quadrante, extrair os TIFFs e encontrar a banda
        for quadrant_name, zip_path in quadrant_paths.items():
            try:
                # Usar o prefixo da data para evitar conflitos entre datas diferentes
                tifs = self.extract_tifs_if_zip(zip_path, prefix=f"{date_prefix}_{quadrant_name}")
                band_path = self.find_band(tifs, band_id)
                band_paths.append(band_path)
                print(f"Banda {band_id} encontrada no quadrante {quadrant_name} para {date_prefix}: {band_path}")
            except Exception as e:
                print(f"Erro ao processar quadrante {quadrant_name} para banda {band_id}: {str(e)}")
        
        if not band_paths:
            raise ValueError(f"Nenhuma banda {band_id} encontrada em qualquer quadrante")
        
        # Unir as bandas extraídas com prefixo único para cada data
        output_path = f"{temp_dir}/merged_band_{band_id}.tif"
        self.merge_rasters(band_paths, output_path)
        
        return output_path

    def read_normalized(self, path: str) -> np.ndarray:
        """Carrega uma banda e normaliza seus valores para o intervalo [0,1]"""
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            print(f"[DEBUG] {path}: min={arr.min()}, max={arr.max()}, mean={arr.mean()}")
            
            if arr.max() == arr.min():
                print(f"AVISO: Dados constantes na banda {path}: valor={arr.min()}")
                return arr * 0  # Retorna uma matriz de zeros do mesmo tamanho
                
            # Normalização
            valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
            if np.sum(valid_mask) > 0:
                min_val = np.percentile(arr[valid_mask], 2)  # Ignora outliers inferiores
                max_val = np.percentile(arr[valid_mask], 98) # Ignora outliers superiores
                
                # Evitar a divisão por zero
                range_val = max_val - min_val
                if range_val < 1e-6:
                    range_val = 1
                
                # Normalização
                norm_arr = np.clip((arr - min_val) / range_val, 0, 1)
                
                print(f"Normalização: min={min_val}, max={max_val}, range={range_val}")
                print(f"Valores normalizados: min={norm_arr.min()}, max={norm_arr.max()}")
                
                return norm_arr
            else:
                print(f"AVISO: Nenhum valor válido na banda {path}")
                return arr * 0

    def calculate_ndvi(self, red_band_path: str, nir_band_path: str) -> Tuple[np.ndarray, dict]:
        """Calcula NDVI: (NIR - RED) / (NIR + RED)"""
        try:
            with rasterio.open(nir_band_path) as nir_src, rasterio.open(red_band_path) as red_src:
                nir_band = nir_src.read(1).astype(np.float32)
                red_band = red_src.read(1).astype(np.float32)
                
                # Substituir valores não válidos
                nir_band[np.isnan(nir_band) | np.isinf(nir_band)] = 0
                red_band[np.isnan(red_band) | np.isinf(red_band)] = 0
                
                # Evitar divisão por zero
                denominator = nir_band + red_band
                valid_mask = denominator > 1e-6
                
                # Inicializar o NDVI com zeros
                ndvi = np.zeros_like(nir_band)
                
                # Calcular NDVI apenas onde o denominador é válido
                ndvi[valid_mask] = (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]
                
                # NDVI está no intervalo [-1, 1], normalizar para [0, 1]
                ndvi_normalized = (ndvi + 1) / 2
                
                print(f"NDVI calculado: min={ndvi.min()}, max={ndvi.max()}")
                print(f"NDVI normalizado: min={ndvi_normalized.min()}, max={ndvi_normalized.max()}")

                meta = red_src.meta.copy()
                meta.update(count=1, dtype='float32')

                return ndvi_normalized, meta
        except Exception as e:
            print(f"Erro no cálculo do NDVI: {str(e)}")
            raise

    def save_raster_with_colormap(self, data: np.ndarray, output_path: str, cmap_name: str = 'RdYlGn', vmin: float = 0, vmax: float = 1):
        """Salva um raster como imagem PNG com um mapa de cores específico e limites definidos"""
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap=cmap_name, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Valor')
        plt.title(f'Visualização: {Path(output_path).stem}')
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Imagem salva: {output_path}")

    def analyze_difference(self, first_date_images: Dict[str, str], second_date_images: Dict[str, str]) -> Dict:
        """Analisa as diferenças entre imagens de satélite de duas datas, com múltiplos quadrantes"""
        print("Processando imagens de múltiplos quadrantes...")

        expected_quadrants = {"NE", "NO", "SO", "SE"}
        if set(first_date_images.keys()) != expected_quadrants or set(second_date_images.keys()) != expected_quadrants:
            raise ValueError("Input inválido: Esperado dicionário com as chaves NE, NO, SO, SE para cada data.")

        for key, value in first_date_images.items():
            print("Element in firstdateimages key: ", key, ", value: ", value)
        for key, value in second_date_images.items():
            print("Element in seconddateimages key: ", key, ", value: ", value)
        
        try:
            os.makedirs("output", exist_ok=True)
            
            # Criar diretórios temporários separados para cada data
            os.makedirs("temp_merged_first", exist_ok=True)
            os.makedirs("temp_merged_second", exist_ok=True)
            
            # Extrair e mesclar as bandas para cada data com prefixos diferentes
            print("Extraindo e mesclando bandas da primeira data...")
            bands_1 = {
                "B2": self.extract_and_merge_bands(first_date_images, "2", "first"),  # Azul
                "B3": self.extract_and_merge_bands(first_date_images, "3", "first"),  # Verde
                "B4": self.extract_and_merge_bands(first_date_images, "4", "first"),  # Vermelho
                "B8": self.extract_and_merge_bands(first_date_images, "8", "first"),  # Infravermelho próximo
            }

            print("Extraindo e mesclando bandas da segunda data...")
            bands_2 = {
                "B2": self.extract_and_merge_bands(second_date_images, "2", "second"),
                "B3": self.extract_and_merge_bands(second_date_images, "3", "second"),
                "B4": self.extract_and_merge_bands(second_date_images, "4", "second"),
                "B8": self.extract_and_merge_bands(second_date_images, "8", "second"),
            }

            print("Criando imagens RGB...")
            # Criar imagem RGB antiga (image 1)
            rgb1 = np.stack([
                self.read_normalized(bands_1["B4"]),
                self.read_normalized(bands_1["B3"]),
                self.read_normalized(bands_1["B2"])
            ], axis=-1)
            plt.imsave("output/rgb_antiga.png", rgb1)

            # Criar imagem RGB recente (image 2)
            rgb2 = np.stack([
                self.read_normalized(bands_2["B4"]),
                self.read_normalized(bands_2["B3"]),
                self.read_normalized(bands_2["B2"])
            ], axis=-1)
            plt.imsave("output/rgb_recente.png", rgb2)

            # Criar imagem NIR antiga
            print("Criando imagens NIR...")
            nir_old = self.read_normalized(bands_1["B8"])
            plt.imsave("output/nir_antiga.png", nir_old, cmap="gray")

            # Criar imagem NIR recente
            nir_recent = self.read_normalized(bands_2["B8"])
            plt.imsave("output/nir_recente.png", nir_recent, cmap="gray")

            # Calcular NDVI
            print("Calculando NDVI...")
            ndvi_old, meta_old = self.calculate_ndvi(bands_1["B4"], bands_1["B8"])
            ndvi_recent, meta_recent = self.calculate_ndvi(bands_2["B4"], bands_2["B8"])
            
            # Salvar imagens NDVI individuais
            self.save_raster_with_colormap(ndvi_old, "output/ndvi_antiga.png", cmap_name="RdYlGn")
            self.save_raster_with_colormap(ndvi_recent, "output/ndvi_recente.png", cmap_name="RdYlGn")

            # Calcular a diferença
            print("Calculando diferença NDVI...")
            ndvi_diff = ndvi_recent - ndvi_old

            # Verificar se existe variação significativa
            diff_std = np.std(ndvi_diff)
            print(f"Diferença NDVI: min={ndvi_diff.min()}, max={ndvi_diff.max()}, std={diff_std}")

            # Usar percentis para definir os limites com base nos dados reais
            p_low, p_high = np.percentile(ndvi_diff[~np.isnan(ndvi_diff)], [5, 95])
            # Garantir simetria em torno de zero para o mapa de cores
            limit = max(abs(p_low), abs(p_high))
            limit = max(limit, 0.05)  # Garantir um mínimo de contraste

            # Guardar o resultado com limites adaptados aos dados
            self.save_raster_with_colormap(
                ndvi_diff, 
                "output/ndvi_diff.png", 
                cmap_name="RdBu_r",  # Mapa de cores alternativo com bom contraste
                vmin=-limit,
                vmax=limit
            )

            # Se a diferença for muito pequena, amplificar o sinal para melhor visualização
            if diff_std < 0.01:
                print("AVISO: Diferença muito pequena entre as imagens NDVI!")
                print("Amplificando diferenças pequenas para melhor visualização")
                
                # Amplificar diferenças por um fator (ex: 5x)
                amplification_factor = 5.0
                ndvi_diff_enhanced = ndvi_diff * amplification_factor
                
                self.save_raster_with_colormap(
                    ndvi_diff_enhanced, 
                    "output/ndvi_diff_enhanced.png", 
                    cmap_name="RdBu_r",
                    vmin=-limit * amplification_factor,
                    vmax=limit * amplification_factor
                )
                
                result_dict = {
                    "image_path_1": "output/rgb_antiga.png",
                    "image_path_2": "output/rgb_recente.png",
                    "ndvi_antiga": "output/ndvi_antiga.png",
                    "ndvi_recente": "output/ndvi_recente.png",
                    "ndvi_diff": "output/ndvi_diff.png",
                    "ndvi_diff_enhanced": "output/ndvi_diff_enhanced.png"
                }
            else:
                result_dict = {
                    "image_path_1": "output/rgb_antiga.png",
                    "image_path_2": "output/rgb_recente.png",
                    "ndvi_antiga": "output/ndvi_antiga.png",
                    "ndvi_recente": "output/ndvi_recente.png",
                    "ndvi_diff": "output/ndvi_diff.png"
                }
            
            # Guardar o resultado da diferença ndvi como tif
            print("Salvando NDVI como TIF...")
            with rasterio.open("output/ndvi_diff.tif", "w", **meta_old) as dst:
                dst.write(ndvi_diff.astype(rasterio.float32), 1)
            
            return result_dict
            
        except Exception as e:
            print(f"Erro detalhado: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Erro ao analisar diferenças entre as imagens: {str(e)}")
        finally:
            self.cleanup_temp_extracted()
            # Limpar diretórios temporários
            for temp_dir in ["temp_merged_first", "temp_merged_second"]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    def _run(self, first_date_images: Dict[str, str], second_date_images: Dict[str, str]) -> str:
        """Executa a análise de diferença entre imagens de múltiplos quadrantes"""
        return self.analyze_difference(first_date_images, second_date_images)