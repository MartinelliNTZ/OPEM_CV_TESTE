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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# =============================================================================
# CONFIGURAÇÕES GLOBAIS (AJUSTE CONFORME SEU AMBIENTE)
# =============================================================================

INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
TRAINING_TREE_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"
TRAINING_SOIL_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp"
BUFFER_SIZE_METERS = 1 
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\resultados_rf_watershed"
OUTPUT_TREES_FILENAME = "arvores_segmentadas_rf_watershed.geojson"
OUTPUT_SOIL_FILENAME = "solo_segmentado_rf_watershed.geojson"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def log_message(message):
    print(f"[{time.strftime("%H:%M:%S")}] {message}")

def calculate_exg(img_rgb):
    r, g, b = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    r_n, g_n, b_n = r/sum_rgb, g/sum_rgb, b/sum_rgb
    exg = 2*g_n - r_n - b_n
    return exg

def get_pixel_coords(src, lon, lat):
    return src.index(lon, lat)

def extract_features_from_buffers(tiff_path, tree_points_path, soil_points_path, buffer_size_m):
    log_message("Iniciando extração de features de buffers...")
    tree_points_path = Path(tree_points_path)
    soil_points_path = Path(soil_points_path)

    with rasterio.open(tiff_path) as src:
        transform = src.transform
        gsd = abs(transform[0])
        buffer_size_pixels = max(1, int(np.ceil(buffer_size_m / gsd)))
        log_message(f"GSD: {gsd:.2f}m/px. Buffer: {buffer_size_pixels}px.")

        img_data = src.read([1, 2, 3])
        img_rgb = np.moveaxis(img_data, 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)

        all_features = []
        all_labels = []

        def process_points(path, label_val):
            if path.is_file():
                log_message(f"Processando {path.name}...")
                gdf = gpd.read_file(path)
                # Otimização: usar apply para evitar iterrows, mas ainda pode ser lento para muitos pontos
                # Melhoria futura: rasterizar buffers diretamente para features
                for idx, row in gdf.iterrows(): # Mantido iterrows por enquanto para compatibilidade com a lógica de buffer
                    if row.geometry.geom_type == 'Point':
                        col_px, row_px = get_pixel_coords(src, row.geometry.x, row.geometry.y)
                        min_row, max_row = max(0, int(row_px - buffer_size_pixels)), min(src.height, int(row_px + buffer_size_pixels))
                        min_col, max_col = max(0, int(col_px - buffer_size_pixels)), min(src.width, int(col_px + buffer_size_pixels))
                        if min_row < max_row and min_col < max_col:
                            p_rgb = img_rgb[min_row:max_row, min_col:max_col].reshape(-1, 3)
                            p_hsv = img_hsv[min_row:max_row, min_col:max_col].reshape(-1, 3)
                            p_exg = img_exg[min_row:max_row, min_col:max_col].reshape(-1, 1)
                            p_feat = np.hstack([p_rgb, p_hsv, p_exg])
                            all_features.append(p_feat)
                            all_labels.append(np.full(p_feat.shape[0], label_val, dtype=np.uint8))

        process_points(tree_points_path, 1)
        process_points(soil_points_path, 0)

    if all_features:
        return np.vstack(all_features), np.concatenate(all_labels)
    return np.empty((0, 7)), np.empty((0,))

def train_random_forest(features, labels):
    log_message("Treinando Random Forest...")
    # Não reduzir o número de amostras conforme solicitado pelo usuário
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5)
    model.fit(X_train, y_train)
    log_message(f"Acurácia: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    return model

def classify_and_segment(tiff_path, rf_model, tree_points_path, soil_points_path, output_dir, chunk_size=2000):
    log_message("Iniciando classificação e segmentação...")
    output_dir = Path(output_dir)
    proc_dir = output_dir / "process"
    proc_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tiff_path) as src:
        h, w = src.shape
        meta = src.meta.copy()
        
        prob_map_trees = np.zeros((h, w), dtype=np.float32)
        prob_map_soil = np.zeros((h, w), dtype=np.float32)
        
        # Otimização: Criar marcadores de forma eficiente
        log_message("Criando marcadores de Watershed...")
        watershed_markers_trees = np.zeros((h, w), dtype=np.int32)
        watershed_markers_soil = np.zeros((h, w), dtype=np.int32)

        # Para árvores
        tree_gdf = gpd.read_file(tree_points_path)
        tree_coords_px = [src.index(p.x, p.y) for p in tree_gdf.geometry if p.geom_type == 'Point']
        for i, (r, c) in enumerate(tree_coords_px, 1):
            if 0 <= r < h and 0 <= c < w: watershed_markers_trees[int(r), int(c)] = i
        log_message(f"Total de marcadores de árvores: {len(tree_coords_px)}")

        # Para solo (opcional, mas útil para segmentar o solo também)
        soil_gdf = gpd.read_file(soil_points_path)
        soil_coords_px = [src.index(p.x, p.y) for p in soil_gdf.geometry if p.geom_type == 'Point']
        for i, (r, c) in enumerate(soil_coords_px, 1):
            if 0 <= r < h and 0 <= c < w: watershed_markers_soil[int(r), int(c)] = i
        log_message(f"Total de marcadores de solo: {len(soil_coords_px)}")

        # Classificação em chunks
        total_chunks = (len(range(0, h, chunk_size))) * (len(range(0, w, chunk_size)))
        current_chunk = 0
        for r_s in range(0, h, chunk_size):
            r_e = min(r_s + chunk_size, h)
            for c_s in range(0, w, chunk_size):
                c_e = min(c_s + chunk_size, w)
                current_chunk += 1
                if current_chunk % 10 == 0 or current_chunk == total_chunks:
                    log_message(f"Processando chunk {current_chunk}/{total_chunks}...")
                
                win = rasterio.windows.Window(c_s, r_s, c_e - c_s, r_e - r_s)
                data = src.read([1, 2, 3], window=win)
                rgb = np.moveaxis(data, 0, -1)
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                
                feat = np.hstack([
                    rgb_u8.reshape(-1, 3),
                    cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3),
                    calculate_exg(rgb_u8).reshape(-1, 1)
                ])
                
                if feat.size > 0:
                    # Prever probabilidades para a classe 'árvore' (1) e 'solo' (0)
                    predictions_proba = rf_model.predict_proba(feat)
                    prob_map_trees[r_s:r_e, c_s:c_e] = predictions_proba[:, 1].reshape(r_e-r_s, c_e-c_s)
                    prob_map_soil[r_s:r_e, c_s:c_e] = predictions_proba[:, 0].reshape(r_e-r_s, c_e-c_s)

        # Salvar Mapas de Probabilidade Intermediários
        meta.update(dtype='float32', count=1)
        with rasterio.open(proc_dir / "prob_map_trees.tif", 'w', **meta) as dst: dst.write(prob_map_trees, 1)
        log_message(f"Mapa de probabilidade de árvores salvo em: {proc_dir / 'prob_map_trees.tif'}")
        with rasterio.open(proc_dir / "prob_map_soil.tif", 'w', **meta) as dst: dst.write(prob_map_soil, 1)
        log_message(f"Mapa de probabilidade de solo salvo em: {proc_dir / 'prob_map_soil.tif'}")

        # Segmentação e Vetorização para Árvores
        log_message("Executando Watershed para Árvores...")
        # Usar a probabilidade de árvores como imagem de distância (inversa) e máscara
        distance_trees = 1.0 - prob_map_trees
        mask_trees = (prob_map_trees > 0.5).astype(np.uint8) # Limiar de 0.5 para árvores
        # OpenCV dilate não suporta uint32, usamos int32 ou float32
        dilated_markers_trees = cv2.dilate(watershed_markers_trees.astype(np.int32), np.ones((3,3), np.uint8))
        labels_trees = watershed(distance_trees, dilated_markers_trees, mask=mask_trees)
        
        meta.update(dtype='int32')
        with rasterio.open(proc_dir / "labels_trees.tif", 'w', **meta) as dst: dst.write(labels_trees, 1)
        log_message(f"Labels de árvores salvos em: {proc_dir / 'labels_trees.tif'}")

        polygons_trees = []
        for lb in np.unique(labels_trees):
            if lb == 0: continue
            m = (labels_trees == lb).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 20: continue # Área mínima para árvores
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.6: continue # Circularidade mínima para árvores
                pts = [rasterio.transform.xy(src.transform, p[0][1], p[0][0]) for p in cnt]
                if len(pts) >= 3: polygons_trees.append(Polygon(pts))

        if polygons_trees:
            gpd.GeoDataFrame(geometry=polygons_trees, crs=src.crs).to_file(output_dir / OUTPUT_TREES_FILENAME, driver='GeoJSON')
            log_message(f"Sucesso! {len(polygons_trees)} polígonos de árvores salvos em {output_dir / OUTPUT_TREES_FILENAME}")
        else:
            log_message("Nenhum polígono de árvore detectado com os parâmetros atuais.")

        # Segmentação e Vetorização para Solo
        log_message("Executando Watershed para Solo...")
        # Usar a probabilidade de solo como imagem de distância (inversa) e máscara
        distance_soil = 1.0 - prob_map_soil
        mask_soil = (prob_map_soil > 0.5).astype(np.uint8) # Limiar de 0.5 para solo
        # OpenCV dilate não suporta uint32, usamos int32 ou float32
        dilated_markers_soil = cv2.dilate(watershed_markers_soil.astype(np.int32), np.ones((3,3), np.uint8))
        labels_soil = watershed(distance_soil, dilated_markers_soil, mask=mask_soil)

        meta.update(dtype='int32')
        with rasterio.open(proc_dir / "labels_soil.tif", 'w', **meta) as dst: dst.write(labels_soil, 1)
        log_message(f"Labels de solo salvos em: {proc_dir / 'labels_soil.tif'}")

        polygons_soil = []
        for lb in np.unique(labels_soil):
            if lb == 0: continue
            m = (labels_soil == lb).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 50: continue # Área mínima para solo (pode ser ajustado)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0 if perimeter == 0 else 4 * np.pi * area / (perimeter * perimeter)
                # if circularity < 0.2: continue # Circularidade mínima para solo (pode ser ajustado)
                pts = [rasterio.transform.xy(src.transform, p[0][1], p[0][0]) for p in cnt]
                if len(pts) >= 3: polygons_soil.append(Polygon(pts))

        if polygons_soil:
            gpd.GeoDataFrame(geometry=polygons_soil, crs=src.crs).to_file(output_dir / OUTPUT_SOIL_FILENAME, driver='GeoJSON')
            log_message(f"Sucesso! {len(polygons_soil)} polígonos de solo salvos em {output_dir / OUTPUT_SOIL_FILENAME}")
        else:
            log_message("Nenhum polígono de solo detectado com os parâmetros atuais.")


if __name__ == "__main__":
    start = time.time()
    # Criar diretório de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    feats, lbs = extract_features_from_buffers(INPUT_IMAGE_PATH, TRAINING_TREE_POINTS_PATH, TRAINING_SOIL_POINTS_PATH, BUFFER_SIZE_METERS)
    if len(feats) > 0:
        model = train_random_forest(feats, lbs)
        classify_and_segment(INPUT_IMAGE_PATH, model, TRAINING_TREE_POINTS_PATH, TRAINING_SOIL_POINTS_PATH, OUTPUT_DIR)
    else:
        log_message("Erro: Nenhuma feature extraída para treinamento. Verifique os caminhos dos pontos de treino.")
    log_message(f"Tempo total: {time.time() - start:.2f}s")
