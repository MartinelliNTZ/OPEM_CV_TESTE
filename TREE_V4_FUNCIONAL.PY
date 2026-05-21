
import rasterio
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import os
import sys
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# =============================================================================
# CONFIGURAÇÕES GLOBAIS (AJUSTE CONFORME SEU AMBIENTE)
# =============================================================================

# Caminho para a imagem GeoTIFF de entrada
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"

# Caminho para o arquivo de pontos de treino (GeoJSON ou Shapefile)
# Essencial para o Cenário 2 (Watershed)
TRAINING_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"

# Caminho para o diretório de saída
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v4"

# Nome do arquivo de saída
OUTPUT_FILENAME = "arvores_segmentadas_watershed.geojson"

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================

def calculate_exg(img_rgb):
    """Calcula o Índice de Vegetação Excessiva (ExG)"""
    r, g, b = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
    # Evitar divisão por zero
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    
    r_n, g_n, b_n = r/sum_rgb, g/sum_rgb, b/sum_rgb
    exg = 2*g_n - r_n - b_n
    return exg

def detect_trees_watershed(tiff_path, points_path, output_dir):
    if not os.path.exists(tiff_path):
        print(f"Erro: Imagem não encontrada: {tiff_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs = src.crs
        img_data = src.read([1, 2, 3])
        img_rgb = np.moveaxis(img_data, 0, -1)
        
        print(f"Lendo pontos de treino: {points_path}")
        points_gdf = gpd.read_file(points_path)
        
        # 1. Gerar Máscara de Vegetação (ExG)
        print("Calculando máscara de vegetação...")
        exg = calculate_exg(img_rgb)
        # Limiar adaptativo ou fixo para vegetação
        mask = exg > 0.05 
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        # 2. Criar Marcadores (Seeds) a partir dos pontos
        print(f"Criando marcadores para {len(points_gdf)} pontos...")
        markers = np.zeros(img_rgb.shape[:2], dtype=int)
        
        count = 0
        for idx, row in points_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                py, px = src.index(row.geometry.x, row.geometry.y)
                if 0 <= py < markers.shape[0] and 0 <= px < markers.shape[1]:
                    count += 1
                    # Atribuir um ID único para cada árvore
                    markers[py, px] = count
        
        # 3. Algoritmo Watershed
        # O Watershed expande os marcadores até preencherem a máscara
        # Usamos o inverso do ExG como 'superfície' (vales onde a árvore é mais verde)
        print("Executando segmentação Watershed (isso pode demorar em imagens grandes)...")
        distance = -exg # Quanto mais verde, mais 'fundo' é o vale
        
        # Watershed precisa de marcadores e uma máscara
        labels = watershed(distance, markers, mask=mask)
        
        # 4. Vetorização dos Polígonos
        print("Vetorizando resultados...")
        polygons = []
        # Processar cada label (ID de árvore)
        # Nota: Para 2000+ pontos, iterar sobre labels únicos é mais eficiente que findContours na imagem toda
        for i in range(1, count + 1):
            tree_mask = (labels == i).astype(np.uint8)
            if np.any(tree_mask):
                contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Pegar o maior contorno para essa label
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 10: # Filtro mínimo
                        geo_coords = []
                        for pt in cnt.reshape(-1, 2):
                            lon, lat = rasterio.transform.xy(transform, pt[1], pt[0])
                            geo_coords.append((lon, lat))
                        if len(geo_coords) >= 3:
                            polygons.append(Polygon(geo_coords))

        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
            out_path = os.path.join(output_dir, OUTPUT_FILENAME)
            gdf.to_file(out_path, driver='GeoJSON')
            print(f"Sucesso! {len(polygons)} polígonos gerados em: {out_path}")
        else:
            print("Nenhum polígono gerado.")

if __name__ == "__main__":
    # Instalar dependências se necessário (scikit-image e scipy)
    # pip install scikit-image scipy
    detect_trees_watershed(INPUT_IMAGE_PATH, TRAINING_POINTS_PATH, OUTPUT_DIR)
