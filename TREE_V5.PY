import rasterio
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pathlib import Path
import os
import sys
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# =============================================================================
# CONFIGURAÇÕES GLOBAIS (AJUSTE CONFORME SEU AMBIENTE)
# =============================================================================

# Caminho para a imagem GeoTIFF de entrada (mosaico de drone)
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"

# Caminho para o arquivo de pontos de treino de ÁRVORES (GeoJSON ou Shapefile)
TRAINING_TREE_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"

# Caminho para o arquivo de pontos de treino de SOLO (GeoJSON ou Shapefile)
TRAINING_SOIL_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp"

# Tamanho do buffer em metros para gerar polígonos de treino a partir dos pontos
BUFFER_SIZE_METERS = 1 # 1 metro

# Caminho para o diretório de saída dos resultados
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\resultados_rf_watershed2"

# Nome do arquivo GeoJSON de saída final (polígonos das árvores)
OUTPUT_POLYGONS_FILENAME = "arvores_segmentadas_rf_watershed.geojson"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def log_message(message):
    """Imprime uma mensagem com timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def calculate_exg(img_rgb):
    """Calcula o Índice de Vegetação Excessiva (ExG) para uma imagem RGB."""
    r, g, b = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1 # Evitar divisão por zero
    
    r_n, g_n, b_n = r/sum_rgb, g/sum_rgb, b/sum_rgb
    exg = 2*g_n - r_n - b_n
    return exg

def get_pixel_coords(src, lon, lat):
    """Converte coordenadas geográficas para coordenadas de pixel (col, row)."""
    return src.index(lon, lat)

def extract_features_from_buffers(tiff_path, tree_points_path, soil_points_path, buffer_size_m, output_process_dir):
    log_message("Iniciando extração de features de buffers...")
    output_process_dir = Path(output_process_dir)
    output_process_dir.mkdir(parents=True, exist_ok=True)

    tree_points_path = Path(tree_points_path)
    soil_points_path = Path(soil_points_path)

    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs = src.crs
        
        # Calcular GSD (Ground Sample Distance) em metros por pixel
        gsd = abs(transform[0]) # GSD em unidades do CRS (metros)
        buffer_size_pixels = max(1, int(np.ceil(buffer_size_m / gsd)))
        log_message(f"GSD da imagem: {gsd:.2f} metros/pixel. Buffer de {buffer_size_m}m = {buffer_size_pixels} pixels.")

        img_data = src.read([1, 2, 3])
        img_rgb = np.moveaxis(img_data, 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)

        all_features = []
        all_labels = []

        # Processar pontos de ÁRVORES
        if tree_points_path.is_file():
            log_message(f"Processando pontos de árvores de {tree_points_path}")
            tree_gdf = gpd.read_file(tree_points_path)
            for idx, row in tree_gdf.iterrows():
                if row.geometry.geom_type == 'Point':
                    col_px, row_px = get_pixel_coords(src, row.geometry.x, row.geometry.y)
                    
                    min_row = max(0, int(row_px - buffer_size_pixels))
                    max_row = min(src.height, int(row_px + buffer_size_pixels))
                    min_col = max(0, int(col_px - buffer_size_pixels))
                    max_col = min(src.width, int(col_px + buffer_size_pixels))

                    if min_row < max_row and min_col < max_col:
                        patch_rgb = img_rgb[min_row:max_row, min_col:max_col]
                        patch_hsv = img_hsv[min_row:max_row, min_col:max_col]
                        patch_exg = img_exg[min_row:max_row, min_col:max_col]
                        patch_features = np.concatenate(
                            [patch_rgb.reshape(-1, 3), patch_hsv.reshape(-1, 3), patch_exg.reshape(-1, 1)],
                            axis=1
                        )
                        all_features.append(patch_features)
                        all_labels.append(np.full(patch_features.shape[0], 1, dtype=np.uint8))
        else:
            log_message(f"Aviso: Arquivo de pontos de árvores não encontrado em {tree_points_path}")

        # Processar pontos de SOLO
        if soil_points_path.is_file():
            log_message(f"Processando pontos de solo de {soil_points_path}")
            soil_gdf = gpd.read_file(soil_points_path)
            for idx, row in soil_gdf.iterrows():
                if row.geometry.geom_type == 'Point':
                    col_px, row_px = get_pixel_coords(src, row.geometry.x, row.geometry.y)
                    
                    min_row = max(0, int(row_px - buffer_size_pixels))
                    max_row = min(src.height, int(row_px + buffer_size_pixels))
                    min_col = max(0, int(col_px - buffer_size_pixels))
                    max_col = min(src.width, int(col_px + buffer_size_pixels))

                    if min_row < max_row and min_col < max_col:
                        patch_rgb = img_rgb[min_row:max_row, min_col:max_col]
                        patch_hsv = img_hsv[min_row:max_row, min_col:max_col]
                        patch_exg = img_exg[min_row:max_row, min_col:max_col]
                        patch_features = np.concatenate(
                            [patch_rgb.reshape(-1, 3), patch_hsv.reshape(-1, 3), patch_exg.reshape(-1, 1)],
                            axis=1
                        )
                        all_features.append(patch_features)
                        all_labels.append(np.full(patch_features.shape[0], 0, dtype=np.uint8))
        else:
            log_message(f"Aviso: Arquivo de pontos de solo não encontrado em {soil_points_path}")

    if all_features:
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
    else:
        all_features = np.empty((0, 7), dtype=np.float32)
        all_labels = np.empty((0,), dtype=np.uint8)

    log_message(f"Extração de features concluída. Total de amostras: {len(all_features)}")
    return all_features, all_labels

def train_random_forest(features, labels):
    log_message("Iniciando treinamento do Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    log_message(f"Acurácia do modelo Random Forest: {accuracy:.4f}")
    return model

def classify_image_and_watershed(
    tiff_path,
    rf_model,
    tree_points_path,
    output_geojson_path,
    output_process_dir,
    chunk_size=2000, # Processar em chunks para evitar travamento
    min_polygon_area=50, # Área mínima do polígono final em pixels
    min_circularity=0.4,
    watershed_threshold=0.5 # Limiar de probabilidade para a máscara do Watershed
):
    log_message("Iniciando classificação da imagem e segmentação Watershed...")
    
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.shape
        log_message(f"Dimensões da imagem: {height}x{width} pixels. Processamento em chunks de {chunk_size}x{chunk_size}.")

        # Carregar pontos de árvores para marcadores do Watershed
        log_message(f"Carregando pontos de árvores para marcadores Watershed de {tree_points_path}")
        tree_markers_gdf = gpd.read_file(tree_points_path)
        
        # Inicializar imagem de classificação e marcadores
        classified_image_proba = np.zeros((height, width), dtype=np.float32) # Armazenar probabilidades float
        watershed_markers = np.zeros((height, width), dtype=int)
        
        marker_id_counter = 1
        for idx, row in tree_markers_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                col_px, row_px = get_pixel_coords(src, row.geometry.x, row.geometry.y)
                if 0 <= row_px < height and 0 <= col_px < width:
                    watershed_markers[int(row_px), int(col_px)] = marker_id_counter
                    marker_id_counter += 1
        log_message(f"Total de marcadores Watershed criados: {marker_id_counter - 1}")

        # Processar a imagem em chunks
        for row_start in range(0, height, chunk_size):
            row_end = min(row_start + chunk_size, height)
            for col_start in range(0, width, chunk_size):
                col_end = min(col_start + chunk_size, width)
                
                log_message(f"Processando chunk: linha {row_start}-{row_end}, coluna {col_start}-{col_end}")
                
                window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
                chunk_data = src.read([1, 2, 3], window=window)
                
                chunk_rgb = np.moveaxis(chunk_data, 0, -1)
                # Garantir que chunk_rgb seja uint8 para cvtColor
                if chunk_rgb.dtype != np.uint8:
                    chunk_rgb_uint8 = np.clip(chunk_rgb, 0, 255).astype(np.uint8)
                else:
                    chunk_rgb_uint8 = chunk_rgb

                chunk_hsv = cv2.cvtColor(chunk_rgb_uint8, cv2.COLOR_RGB2HSV)
                chunk_exg = calculate_exg(chunk_rgb_uint8)

                flat_rgb = chunk_rgb_uint8.reshape(-1, 3)
                flat_hsv = chunk_hsv.reshape(-1, 3)
                flat_exg = chunk_exg.reshape(-1, 1)
                chunk_features_np = np.hstack([flat_rgb, flat_hsv, flat_exg])

                if chunk_features_np.size > 0:
                    # Prever probabilidades para a classe 'árvore' (classe 1)
                    chunk_predictions_proba = rf_model.predict_proba(chunk_features_np)[:, 1]
                    classified_image_proba[row_start:row_end, col_start:col_end] = chunk_predictions_proba.reshape(chunk_rgb.shape[0], chunk_rgb.shape[1])

        # Salvar mapa de probabilidade intermediário
        output_proba_path = os.path.join(output_process_dir, "rf_probability_map.tif")
        with rasterio.open(
            output_proba_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=classified_image_proba.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(classified_image_proba, 1)
        log_message(f"Mapa de probabilidade do Random Forest salvo em: {output_proba_path}")

        # Criar máscara para o Watershed: apenas pixels com probabilidade > threshold
        # Isso ajuda a evitar que o Watershed se espalhe para áreas de não-árvore
        watershed_mask = (classified_image_proba > watershed_threshold).astype(np.uint8)
        
        # Aplicar Watershed na imagem de probabilidade
        log_message("Aplicando Watershed na imagem de probabilidade...")
        # A imagem de distância para o Watershed deve ter 'vales' onde queremos segmentar.
        # Quanto maior a probabilidade de árvore, mais 'fundo' deve ser o vale.
        # Usamos 1 - probabilidade para que alta probabilidade (próximo de 1) se torne um valor baixo (próximo de 0).
        distance_map = 1 - classified_image_proba
        
        # Certificar que os marcadores estão dentro da máscara
        # Aumentar os marcadores para que sejam pequenos 'lagos' em vez de pontos únicos
        # Isso pode ajudar a iniciar o Watershed de forma mais robusta
        dilated_markers = cv2.dilate(watershed_markers.astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
        
        # O Watershed precisa de marcadores e uma máscara. A máscara restringe a expansão.
        # Usamos o mapa de distância e os marcadores, aplicando a máscara para limitar a segmentação.
        labels = watershed(distance_map, dilated_markers, mask=watershed_mask)

        # Salvar labels do Watershed intermediário
        output_labels_path = os.path.join(output_process_dir, "watershed_labels.tif")
        with rasterio.open(
            output_labels_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=labels.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(labels, 1)
        log_message(f"Labels do Watershed salvos em: {output_labels_path}")
        
        # Vetorização dos Polígonos
        log_message("Vetorizando resultados do Watershed...")
        polygons = []
        for label in np.unique(labels):
            if label == 0: # 0 é o background do Watershed
                continue
            
            label_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                # Adicionar um filtro de área mínima para evitar ruído
                if area < min_polygon_area: 
                    continue

                circularity = 0
                if perimeter > 0: # Evitar divisão por zero
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Adicionar um filtro de circularidade mínima para remover formas alongadas que não são árvores
                if circularity < min_circularity:
                    continue

                geo_coords = []
                for point in cnt.reshape(-1, 2):
                    lon, lat = rasterio.transform.xy(transform, point[1], point[0])
                    geo_coords.append((lon, lat))
                
                if len(geo_coords) >= 3:
                    try:
                        polygons.append(Polygon(geo_coords))
                    except Exception as e:
                        log_message(f"Aviso: Não foi possível criar polígono válido para um contorno: {e}")

        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
            final_output_path = os.path.join(output_process_dir, output_geojson_path) # Salvar no subdiretório 'process'
            gdf.to_file(final_output_path, driver='GeoJSON')
            log_message(f"Sucesso! {len(polygons)} polígonos de árvores salvos em {final_output_path}")
        else:
            log_message("Nenhum polígono de árvore detectado com os parâmetros atuais.")

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    log_message("Iniciando script de detecção de árvores v6 (corrigido)...")

    # Criar diretório de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Criar subdiretório 'process' para arquivos intermediários
    output_process_dir = os.path.join(OUTPUT_DIR, "process")
    if not os.path.exists(output_process_dir):
        os.makedirs(output_process_dir)

    # 1. Extrair features de buffers de treino
    features, labels = extract_features_from_buffers(
        tiff_path=INPUT_IMAGE_PATH,
        tree_points_path=TRAINING_TREE_POINTS_PATH,
        soil_points_path=TRAINING_SOIL_POINTS_PATH,
        buffer_size_m=BUFFER_SIZE_METERS,
        output_process_dir=output_process_dir
    )

    if len(features) == 0:
        log_message("Erro: Nenhuma feature extraída para treinamento. Verifique os caminhos dos pontos de treino.")
        sys.exit(1)

    # 2. Treinar modelo Random Forest
    rf_model = train_random_forest(features, labels)

    # 3. Classificar imagem completa e aplicar Watershed
    classify_image_and_watershed(
        tiff_path=INPUT_IMAGE_PATH,
        rf_model=rf_model,
        tree_points_path=TRAINING_TREE_POINTS_PATH, # Usar os pontos de árvore como marcadores para o Watershed
        output_geojson_path=OUTPUT_POLYGONS_FILENAME,
        output_process_dir=output_process_dir,
        chunk_size=2000, # Ajuste conforme a memória disponível
        min_polygon_area=20, # Reduzir área mínima para capturar árvores menores
        min_circularity=0.6, # Aumentar circularidade para focar em formas mais redondas
        watershed_threshold=0.7 # Aumentar limiar para máscara do Watershed (apenas pixels com alta prob. de árvore)
    )

    log_message("Script concluído.")
