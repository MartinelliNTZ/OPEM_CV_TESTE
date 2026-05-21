import rasterio
import rasterio.features
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon
from pathlib import Path
import os
import sys
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
TRAINING_TREE_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"
TRAINING_SOIL_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp"
BUFFER_SIZE_METERS = 1 
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v10_final"
OUTPUT_TREES_FILENAME = "arvores_segmentadas.geojson"
OUTPUT_SOIL_FILENAME = "solo_segmentado.geojson"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def log_message(message, log_file=None):
    msg = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

def calculate_exg(img_rgb):
    r, g, b = img_rgb[:,:,0].astype(float), img_rgb[:,:,1].astype(float), img_rgb[:,:,2].astype(float)
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    r_n, g_n, b_n = r/sum_rgb, g/sum_rgb, b/sum_rgb
    exg = 2*g_n - r_n - b_n
    return exg

def extract_features_from_buffers(tiff_path, tree_points_path, soil_points_path, buffer_size_m, log_file):
    log_message("Iniciando extração de features...", log_file)
    with rasterio.open(tiff_path) as src:
        gsd = abs(src.transform[0])
        buffer_px = max(1, int(np.ceil(buffer_size_m / gsd)))
        
        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)
        
        all_features, all_labels = [], []

        # Função process definida fora para evitar erros de escopo
        def process_class(path, label_val, name, src_obj, rgb, hsv, exg, b_px):
            if Path(path).is_file():
                gdf = gpd.read_file(path)
                log_message(f"Extraindo {len(gdf)} pontos de {name}...", log_file)
                f_list, l_list = [], []
                for geom in gdf.geometry:
                    if geom.geom_type == 'Point':
                        c, r = src_obj.index(geom.x, geom.y)
                        r_s, r_e = max(0, r-b_px), min(src_obj.height, r+b_px)
                        c_s, c_e = max(0, c-b_px), min(src_obj.width, c+b_px)
                        if r_s < r_e and c_s < c_e:
                            feat = np.hstack([
                                rgb[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                hsv[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                exg[r_s:r_e, c_s:c_e].reshape(-1, 1)
                            ])
                            f_list.append(feat)
                            l_list.append(np.full(feat.shape[0], label_val))
                return f_list, l_list
            return [], []

        f_tree, l_tree = process_class(tree_points_path, 1, "Árvores", src, img_rgb, img_hsv, img_exg, buffer_px)
        f_soil, l_soil = process_class(soil_points_path, 0, "Solo", src, img_rgb, img_hsv, img_exg, buffer_px)
        
        all_features = f_tree + f_soil
        all_labels = l_tree + l_soil
        
    if not all_features:
        return np.empty((0, 7)), np.empty((0,))
        
    return np.vstack(all_features), np.concatenate(all_labels)

def train_model(features, labels, log_file):
    log_message("Treinando Random Forest...", log_file)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test))
    log_message(f"Acurácia: {acc:.4f}", log_file)
    log_message(f"Relatório:\n{report}", log_file)
    return model

def run_pipeline(tiff_path, rf_model, tree_pts, soil_pts, out_dir, chunk_size=1024, batch_size=200000):
    log_message("Iniciando Classificação e Watershed...")
    out_dir = Path(out_dir)
    proc_dir = out_dir / "process"
    proc_dir.mkdir(parents=True, exist_ok=True)
    log_f = proc_dir / "process_info.txt"
    
    with rasterio.open(tiff_path) as src:
        h, w = src.shape
        meta = src.meta.copy()
        
        prob_map_trees = np.zeros((h, w), dtype=np.float32)
        prob_map_soil = np.zeros((h, w), dtype=np.float32)
        markers = np.zeros((h, w), dtype=np.int32)
        
        # 1. Marcadores
        log_message("Criando marcadores...", log_f)
        tree_gdf = gpd.read_file(tree_pts)
        soil_gdf = gpd.read_file(soil_pts)
        
        for i, geom in enumerate(tree_gdf.geometry, 1):
            if geom.geom_type == 'Point':
                c, r = src.index(geom.x, geom.y)
                if 0 <= r < h and 0 <= c < w: markers[r, c] = i
        
        offset = len(tree_gdf) + 1
        for i, geom in enumerate(soil_gdf.geometry, offset):
            if geom.geom_type == 'Point':
                c, r = src.index(geom.x, geom.y)
                if 0 <= r < h and 0 <= c < w: markers[r, c] = i
        
        # 2. Classificação em Chunks
        log_message(f"Classificando imagem ({h}x{w})...", log_f)
        for r_s in range(0, h, chunk_size):
            r_e = min(r_s + chunk_size, h)
            for c_s in range(0, w, chunk_size):
                c_e = min(c_s + chunk_size, w)
                win = rasterio.windows.Window(c_s, r_s, c_e - c_s, r_e - r_s)
                rgb = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1)
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                feat = np.hstack([
                    rgb_u8.reshape(-1, 3),
                    cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3),
                    calculate_exg(rgb_u8).reshape(-1, 1)
                ])

                # Predict in batches to avoid large memory spikes
                n = feat.shape[0]
                probas = np.zeros((n, 2), dtype=np.float32)
                for i in range(0, n, batch_size):
                    j = min(i + batch_size, n)
                    probas[i:j] = rf_model.predict_proba(feat[i:j])

                prob_map_trees[r_s:r_e, c_s:c_e] = probas[:, 1].reshape(r_e-r_s, c_e-c_s)
                prob_map_soil[r_s:r_e, c_s:c_e] = probas[:, 0].reshape(r_e-r_s, c_e-c_s)

        # Salvar Mapas de Probabilidade Separados
        meta.update(dtype='float32', count=1)
        with rasterio.open(proc_dir / "prob_map_trees.tif", 'w', **meta) as dst: dst.write(prob_map_trees, 1)
        with rasterio.open(proc_dir / "prob_map_soil.tif", 'w', **meta) as dst: dst.write(prob_map_soil, 1)

        # 3. Watershed
        log_message("Executando Watershed...", log_f)
        dist = np.minimum(1.0 - prob_map_trees, 1.0 - prob_map_soil)
        mask = ((prob_map_trees > 0.3) | (prob_map_soil > 0.3)).astype(np.uint8)
        labels = watershed(dist, markers, mask=mask)
        
        meta.update(dtype='int32')
        with rasterio.open(proc_dir / "labels_final.tif", 'w', **meta) as dst: dst.write(labels, 1)

        # 4. Vetorização Rápida
        log_message("Vetorizando resultados...", log_f)
        shapes = rasterio.features.shapes(labels, mask=mask, transform=src.transform)
        
        tree_geoms, soil_geoms = [], []
        for geom, val in shapes:
            val = int(val)
            if val == 0: continue
            poly = shape(geom)
            if poly.area < 5: continue
            if val < offset: tree_geoms.append(poly)
            else: soil_geoms.append(poly)

        if tree_geoms:
            gpd.GeoDataFrame(geometry=tree_geoms, crs=src.crs).to_file(out_dir / OUTPUT_TREES_FILENAME, driver='GeoJSON')
            log_message(f"Árvores: {len(tree_geoms)} polígonos.", log_f)
        if soil_geoms:
            gpd.GeoDataFrame(geometry=soil_geoms, crs=src.crs).to_file(out_dir / OUTPUT_SOIL_FILENAME, driver='GeoJSON')
            log_message(f"Solo: {len(soil_geoms)} polígonos.", log_f)

if __name__ == "__main__":
    start = time.time()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    log_file = Path(OUTPUT_DIR) / "process" / "process_info.txt"
    os.makedirs(log_file.parent, exist_ok=True)
    
    try:
        f, l = extract_features_from_buffers(INPUT_IMAGE_PATH, TRAINING_TREE_POINTS_PATH, TRAINING_SOIL_POINTS_PATH, BUFFER_SIZE_METERS, log_file)
        if len(f) > 0:
            model = train_model(f, l, log_file)
            run_pipeline(INPUT_IMAGE_PATH, model, TRAINING_TREE_POINTS_PATH, TRAINING_SOIL_POINTS_PATH, OUTPUT_DIR)
        log_message(f"Concluído em {time.time()-start:.2f}s", log_file)
    except KeyboardInterrupt:
        log_message("Process interrupted by user.", log_file)
