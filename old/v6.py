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
from sklearn.metrics import accuracy_score, classification_report
import time

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
TRAINING_TREE_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"
TRAINING_SOIL_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp"
BUFFER_SIZE_METERS = 1 
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v7_final"
OUTPUT_TREES_FILENAME = "arvores_segmentadas.geojson"
OUTPUT_SOIL_FILENAME = "solo_segmentado.geojson"
MAX_POINTS_PER_CLASS = 500
MAX_PIXELS_PER_BUFFER = 400

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
    rng = np.random.RandomState(42)
    with rasterio.open(tiff_path) as src:
        gsd = abs(src.transform[0])
        buffer_px = max(1, int(np.ceil(buffer_size_m / gsd)))
        
        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)
        
        all_features, all_labels = [], []

        def process(path, label_val, name):
            if Path(path).is_file():
                gdf = gpd.read_file(path)
                total_points = len(gdf)
                if total_points > MAX_POINTS_PER_CLASS:
                    gdf = gdf.sample(n=MAX_POINTS_PER_CLASS, random_state=42)
                    log_message(f"Amostrando {MAX_POINTS_PER_CLASS}/{total_points} pontos de {name}...", log_file)
                else:
                    log_message(f"Extraindo {total_points} pontos de {name}...", log_file)

                for idx, geom in enumerate(gdf.geometry, 1):
                    if idx % 100 == 0:
                        log_message(f"  Processados {idx}/{len(gdf)} pontos de {name}...", log_file)
                    if geom.geom_type == 'Point':
                        c, r = src.index(geom.x, geom.y)
                        r_s, r_e = max(0, r-buffer_px), min(src.height, r+buffer_px)
                        c_s, c_e = max(0, c-buffer_px), min(src.width, c+buffer_px)
                        if r_s < r_e and c_s < c_e:
                            feat = np.hstack([
                                img_rgb[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                img_hsv[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                img_exg[r_s:r_e, c_s:c_e].reshape(-1, 1)
                            ])
                            if feat.shape[0] > MAX_PIXELS_PER_BUFFER:
                                indices = rng.choice(feat.shape[0], size=MAX_PIXELS_PER_BUFFER, replace=False)
                                feat = feat[indices]
                            all_features.append(feat)
                            all_labels.append(np.full(feat.shape[0], label_val))

        process(tree_points_path, 1, "Árvores")
        process(soil_points_path, 0, "Solo")
        
    if not all_features:
        return np.empty((0, 7)), np.empty((0,))
    return np.vstack(all_features), np.concatenate(all_labels)

def train_model(features, labels, log_file):
    log_message("Treinando Random Forest...", log_file)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, n_jobs=1, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test))
    log_message(f"Acurácia: {acc:.4f}", log_file)
    log_message(f"Relatório:\n{report}", log_file)
    return model

def run_pipeline(tiff_path, rf_model, tree_pts, soil_pts, out_dir, chunk_size=2000):
    log_message("Iniciando Classificação e Watershed Unificado...")
    out_dir = Path(out_dir)
    proc_dir = out_dir / "process"
    proc_dir.mkdir(parents=True, exist_ok=True)
    log_f = proc_dir / "process_info.txt"
    
    with rasterio.open(tiff_path) as src:
        h, w = src.shape
        meta = src.meta.copy()
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        markers = np.zeros((h, w), dtype=np.int32)
        
        # 1. Marcadores Unificados (Preservando IDs)
        log_message("Criando marcadores unificados...", log_f)
        tree_gdf = gpd.read_file(tree_pts)
        soil_gdf = gpd.read_file(soil_pts)
        
        # IDs de Árvores: 1 a N
        for i, geom in enumerate(tree_gdf.geometry, 1):
            if geom.geom_type == 'Point':
                c, r = src.index(geom.x, geom.y)
                if 0 <= r < h and 0 <= c < w: markers[r, c] = i
        
        # IDs de Solo: N+1 em diante (usaremos um ID negativo ou alto para solo)
        # Para simplificar e permitir múltiplos polígonos de solo, cada ponto de solo é um marcador único
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
                prob_map[r_s:r_e, c_s:c_e] = rf_model.predict_proba(feat)[:, 1].reshape(r_e-r_s, c_e-c_s)

        # Salvar Probabilidade (Branco = Árvore, Preto = Solo)
        meta.update(dtype='float32', count=1)
        with rasterio.open(proc_dir / "prob_map_final.tif", 'w', **meta) as dst: dst.write(prob_map, 1)

        # 3. Watershed Unificado (Competição)
        log_message("Executando Watershed Unificado (Competição Árvore vs Solo)...", log_f)
        # Imagem de distância: 1 - prob para árvores, prob para solo. 
        # Como é unificado, usamos 0.5 - prob_map para criar vales onde a probabilidade é alta para qualquer classe
        # Mas o mais fidedigno é usar o gradiente ou a própria probabilidade invertida
        dist = 1.0 - np.abs(prob_map - 0.0) # Vales em 0 (solo) e 1 (árvore)
        
        # Dilatar marcadores sem perder IDs (usando dilatação de labels)
        markers_dilated = cv2.dilate(markers.astype(np.int32), np.ones((3,3), np.uint8))
        
        # Watershed sem máscara para preencher todo o espaço ou com máscara de dados
        labels = watershed(dist, markers_dilated)
        
        meta.update(dtype='int32')
        with rasterio.open(proc_dir / "labels_unificados.tif", 'w', **meta) as dst: dst.write(labels, 1)

        # 4. Vetorização Separada
        log_message("Vetorizando resultados...", log_f)
        tree_polys, soil_polys = [], []
        unique_labels = np.unique(labels)
        
        for lb in unique_labels:
            if lb == 0: continue
            mask = (labels == lb).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 10: continue
                
                pts = [rasterio.transform.xy(src.transform, p[0][1], p[0][0]) for p in cnt]
                if len(pts) < 3: continue
                poly = Polygon(pts)
                
                # Decidir se é árvore ou solo baseado no ID original
                if lb < offset:
                    tree_polys.append(poly)
                else:
                    soil_polys.append(poly)

        # Salvar GeoJSONs
        if tree_polys:
            gpd.GeoDataFrame(geometry=tree_polys, crs=src.crs).to_file(out_dir / OUTPUT_TREES_FILENAME, driver='GeoJSON')
            log_message(f"Árvores: {len(tree_polys)} polígonos.", log_f)
        if soil_polys:
            gpd.GeoDataFrame(geometry=soil_polys, crs=src.crs).to_file(out_dir / OUTPUT_SOIL_FILENAME, driver='GeoJSON')
            log_message(f"Solo: {len(soil_polys)} polígonos.", log_f)

if __name__ == "__main__":
    start = time.time()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    log_file = Path(OUTPUT_DIR) / "process" / "process_info.txt"
    os.makedirs(log_file.parent, exist_ok=True)
    
    f, l = extract_features_from_buffers(INPUT_IMAGE_PATH, TRAINING_TREE_POINTS_PATH, TRAINING_SOIL_POINTS_PATH, BUFFER_SIZE_METERS, log_file)
    model = train_model(f, l, log_file)
    run_pipeline(INPUT_IMAGE_PATH, model, TRAINING_TREE_POINTS_PATH, TRAINING_SOIL_POINTS_PATH, OUTPUT_DIR)
    
    log_message(f"Concluído em {time.time()-start:.2f}s", log_file)
