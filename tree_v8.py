import rasterio
import cv2
import numpy as np
from pathlib import Path
import os
import time
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import geopandas as gpd

# =============================================================================
# CONFIGURACOES GLOBAIS
# =============================================================================
INPUT_IMAGE_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif"
TRAINING_TREE_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\floresta_pts.shp"
TRAINING_SOIL_POINTS_PATH = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\solo_pts.shp"
BUFFER_SIZE_METERS = 1
OUTPUT_DIR = r"D:\TESTES_PYTHON\OPEM_CV_TESTE\v11_final"

# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================


def log_message(message, log_file=None, also_print=True):
    msg = f"[{time.strftime('%H:%M:%S')}] {message}"
    if also_print:
        try:
            print(msg)
        except UnicodeEncodeError:
            sanitized = msg.encode("ascii", errors="replace").decode("ascii")
            print(sanitized)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def calculate_exg(img_rgb):
    """Excess Green Index: 2*G - R - B (normalized RGB)"""
    r, g, b = (
        img_rgb[:, :, 0].astype(float),
        img_rgb[:, :, 1].astype(float),
        img_rgb[:, :, 2].astype(float),
    )
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1
    r_n, g_n, b_n = r / sum_rgb, g / sum_rgb, b / sum_rgb
    return 2 * g_n - r_n - b_n


def extract_features_for_training(
    tiff_path, tree_points_path, soil_points_path, buffer_size_m, log_file
):
    log_message("=" * 60, log_file)
    log_message("ETAPA 1: EXTRACAO DE FEATURES PARA TREINAMENTO", log_file)
    log_message("=" * 60, log_file)
    log_message(f"Imagem de entrada: {tiff_path}", log_file)
    log_message(f"Arquivo de pontos de arvores: {tree_points_path}", log_file)
    log_message(f"Arquivo de pontos de solo: {soil_points_path}", log_file)

    with rasterio.open(tiff_path) as src:
        gsd = abs(src.transform[0])
        buffer_px = max(1, int(np.ceil(buffer_size_m / gsd)))

        h_img, w_img = src.shape
        n_bands = src.count
        crs_str = str(src.crs) if src.crs else "Nao definido"
        bounds = src.bounds
        log_message(
            f"Dimensoes da imagem: {w_img} x {h_img} pixels ({n_bands} bandas)",
            log_file,
        )
        log_message(f"Resolucao (GSD): {gsd:.6f} m/pixel", log_file)
        log_message(f"CRS: {crs_str}", log_file)
        log_message(
            f"Extent (bounds): left={bounds.left:.2f}, bottom={bounds.bottom:.2f}, right={bounds.right:.2f}, top={bounds.top:.2f}",
            log_file,
        )
        log_message(
            f"Buffer de treinamento: {buffer_size_m}m -> {buffer_px} pixels", log_file
        )
        log_message(f"Usando bandas 1(R), 2(G), 3(B) para features", log_file)

        img_rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_exg = calculate_exg(img_rgb)

        n_features = 7  # R, G, B, H, S, V, ExG

        all_features, all_labels = [], []

        def process_class(path, label_val, name, src_obj, rgb, hsv, exg, b_px):
            if Path(path).is_file():
                gdf = gpd.read_file(path)
                log_message(f"Extraindo {len(gdf)} pontos de {name}...", log_file)
                f_list, l_list = [], []
                valid_count = 0
                out_of_bounds = 0
                for geom in gdf.geometry:
                    if geom.geom_type == "Point":
                        c, r = src_obj.index(geom.x, geom.y)
                        r_s, r_e = max(0, r - b_px), min(src_obj.height, r + b_px)
                        c_s, c_e = max(0, c - b_px), min(src_obj.width, c + b_px)
                        if r_s < r_e and c_s < c_e:
                            feat = np.hstack(
                                [
                                    rgb[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                    hsv[r_s:r_e, c_s:c_e].reshape(-1, 3),
                                    exg[r_s:r_e, c_s:c_e].reshape(-1, 1),
                                ]
                            )
                            f_list.append(feat)
                            l_list.append(np.full(feat.shape[0], label_val))
                            valid_count += 1
                        else:
                            out_of_bounds += 1
                log_message(
                    f"  -> {name}: {valid_count} pontos dentro da imagem, {out_of_bounds} fora dos limites",
                    log_file,
                )
                return f_list, l_list
            else:
                log_message(f"  -> Arquivo de {name} nao encontrado: {path}", log_file)
                return [], []

        f_tree, l_tree = process_class(
            tree_points_path, 1, "Arvores", src, img_rgb, img_hsv, img_exg, buffer_px
        )
        f_soil, l_soil = process_class(
            soil_points_path, 0, "Solo", src, img_rgb, img_hsv, img_exg, buffer_px
        )

        all_features = f_tree + f_soil
        all_labels = l_tree + l_soil

    if not all_features:
        log_message(
            "ERRO: Nenhuma feature extraida! Verifique os arquivos de pontos.", log_file
        )
        return np.empty((0, n_features)), np.empty((0,))

    total_pixels = sum(f.shape[0] for f in all_features)
    tree_pixels = sum(f.shape[0] for f in f_tree) if f_tree else 0
    soil_pixels = sum(f.shape[0] for f in f_soil) if f_soil else 0

    log_message(f"\nResumo da extracao:", log_file)
    log_message(
        f"  -> Features por pixel: R(1) G(1) B(1) | H(1) S(1) V(1) | ExG(1) = {n_features} features",
        log_file,
    )
    log_message(
        f"  -> Total de pixels de arvore para treino: {tree_pixels:,}", log_file
    )
    log_message(f"  -> Total de pixels de solo para treino: {soil_pixels:,}", log_file)
    log_message(f"  -> Total de amostras (pixels): {total_pixels:,}", log_file)
    log_message(
        f"  -> Proporcao arvore/solo: {tree_pixels/(soil_pixels+1):.2f}", log_file
    )

    return np.vstack(all_features), np.concatenate(all_labels)


def train_model(features, labels, log_file):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 2: TREINAMENTO DO MODELO (Random Forest)", log_file)
    log_message("=" * 60, log_file)
    t0 = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    log_message(f"Tamanho do conjunto de treino: {X_train.shape[0]:,} pixels", log_file)
    log_message(f"Tamanho do conjunto de teste: {X_test.shape[0]:,} pixels", log_file)
    log_message(
        f"Distribuicao treino - Arvore(1): {np.sum(y_train==1)}, Solo(0): {np.sum(y_train==0)}",
        log_file,
    )
    log_message(
        f"Distribuicao teste  - Arvore(1): {np.sum(y_test==1)}, Solo(0): {np.sum(y_test==0)}",
        log_file,
    )

    log_message("Treinando Random Forest (n_estimators=100, max_depth=20)...", log_file)
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, max_depth=20, random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    log_message(f"\nTempo de treinamento: {format_time(train_time)}", log_file)
    log_message(f"Acuracia: {acc:.4f}", log_file)
    log_message(f"Relatorio de Classificacao:\n{report}", log_file)

    feature_names = ["R", "G", "B", "H", "S", "V", "ExG"]
    log_message(f"\nImportancia das Features (Random Forest):", log_file)
    importances = sorted(
        zip(feature_names, model.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        log_message(f"  {name}: {imp:.4f} ({imp*100:.1f}%)", log_file)

    log_message(
        f"\nClasses do modelo (ordem predict_proba): {model.classes_}", log_file
    )
    log_message(f"Numero de classes: {len(model.classes_)}", log_file)

    return model


def generate_probability_maps(
    tiff_path,
    rf_model,
    tree_points_path,
    soil_points_path,
    out_dir,
    log_file,
    chunk_size=1024,
    batch_size=200000,
):
    log_message("\n" + "=" * 60, log_file)
    log_message("ETAPA 3: GERACAO DE MAPAS DE PROBABILIDADE", log_file)
    log_message("=" * 60, log_file)

    input_path = Path(tiff_path)
    output_base_name = input_path.stem
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_message(f"\nConfiguracoes de Geracao de Mapas:", log_file)
    log_message(f"  -> Imagem de entrada: {tiff_path}", log_file)
    log_message(f"  -> Diretorio de saida: {output_dir}", log_file)
    log_message(f"  -> Nome base dos arquivos de saida: {output_base_name}", log_file)
    log_message(f"  -> Tamanho do chunk: {chunk_size}x{chunk_size} pixels", log_file)
    log_message(f"  -> Tamanho do batch de predicao: {batch_size:,} pixels", log_file)
    log_message(f"  -> Features: R, G, B, H, S, V, ExG (7 features)", log_file)

    t_pipeline_start = time.time()

    tree_class_idx = np.where(rf_model.classes_ == 1)[0][0]
    soil_class_idx = np.where(rf_model.classes_ == 0)[0][0]

    log_message(
        f"\n[INFO] Mapeamento de classes: rf_model.classes_ = {rf_model.classes_}",
        log_file,
    )
    log_message(
        f"[INFO] tree_class_idx = {tree_class_idx} (probabilidade de arvore, classe 1)",
        log_file,
    )
    log_message(
        f"[INFO] soil_class_idx = {soil_class_idx} (probabilidade de solo, classe 0)",
        log_file,
    )

    with rasterio.open(tiff_path) as src:
        h, w = src.shape
        meta = src.meta.copy()

        prob_map_trees = np.zeros((h, w), dtype=np.float32)
        prob_map_soil = np.zeros((h, w), dtype=np.float32)

        t_class = time.time()
        log_message(f"\nClassificando a imagem inteira com Random Forest...", log_file)
        log_message(f"  -> Dimensoes: {w}x{h} pixels = {w*h:,} pixels", log_file)

        n_chunks = 0
        for r_s in range(0, h, chunk_size):
            r_e = min(r_s + chunk_size, h)
            for c_s in range(0, w, chunk_size):
                c_e = min(c_s + chunk_size, w)
                n_chunks += 1
                win = rasterio.windows.Window(c_s, r_s, c_e - c_s, r_e - r_s)
                rgb = np.moveaxis(src.read([1, 2, 3], window=win), 0, -1)
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                feat = np.hstack(
                    [
                        rgb_u8.reshape(-1, 3),
                        cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3),
                        calculate_exg(rgb_u8).reshape(-1, 1),
                    ]
                )

                n = feat.shape[0]
                probas = np.zeros((n, 2), dtype=np.float32)
                for i in range(0, n, batch_size):
                    j = min(i + batch_size, n)
                    probas[i:j] = rf_model.predict_proba(feat[i:j])

                prob_map_trees[r_s:r_e, c_s:c_e] = probas[:, tree_class_idx].reshape(
                    r_e - r_s, c_e - c_s
                )
                prob_map_soil[r_s:r_e, c_s:c_e] = probas[:, soil_class_idx].reshape(
                    r_e - r_s, c_e - c_s
                )

        class_time = time.time() - t_class
        log_message(f"  -> Total de chunks processados: {n_chunks}", log_file)
        log_message(f"  -> Tempo de classificacao: {format_time(class_time)}", log_file)
        log_message(
            f"  -> Velocidade media: {(w*h)/class_time/1e6:.2f}M pixels/s", log_file
        )

        log_message(f"\n  -> Estatisticas prob_map_trees:", log_file)
        log_message(
            f"      min={prob_map_trees.min():.4f}, max={prob_map_trees.max():.4f}, mean={prob_map_trees.mean():.4f}, std={prob_map_trees.std():.4f}",
            log_file,
        )
        log_message(f"  -> Estatisticas prob_map_soil:", log_file)
        log_message(
            f"      min={prob_map_soil.min():.4f}, max={prob_map_soil.max():.4f}, mean={prob_map_soil.mean():.4f}, std={prob_map_soil.std():.4f}",
            log_file,
        )

        # FIX: DETECCAO E CORRECAO DE INVERSION DAS PROBABILIDADES
        invertido = False
        tree_gdf = gpd.read_file(tree_points_path)
        soil_gdf = gpd.read_file(soil_points_path)

        tree_probs_at_tree_pts = []
        for geom in tree_gdf.geometry:
            if geom.geom_type == "Point":
                c, r = src.index(geom.x, geom.y)
                if 0 <= r < h and 0 <= c < w:
                    tree_probs_at_tree_pts.append(prob_map_trees[r, c])

        soil_probs_at_soil_pts = []
        for geom in soil_gdf.geometry:
            if geom.geom_type == "Point":
                c, r = src.index(geom.x, geom.y)
                if 0 <= r < h and 0 <= c < w:
                    soil_probs_at_soil_pts.append(prob_map_soil[r, c])

        mean_tree_prob_at_tree = (
            np.mean(tree_probs_at_tree_pts) if tree_probs_at_tree_pts else 0
        )
        mean_soil_prob_at_soil = (
            np.mean(soil_probs_at_soil_pts) if soil_probs_at_soil_pts else 0
        )

        log_message(
            f"\n  -> FIX INVERSION (verificacao nos pontos de treino):", log_file
        )
        log_message(
            f"      P(arvore medio) nos PONTOS DE ARVORE: {mean_tree_prob_at_tree:.4f}",
            log_file,
        )
        log_message(
            f"      P(solo medio) nos PONTOS DE SOLO: {mean_soil_prob_at_soil:.4f}",
            log_file,
        )

        meta.update(dtype="float32", count=1)

        output_tree_path = output_dir / f"{output_base_name}_prob_trees.tif"
        output_soil_path = output_dir / f"{output_base_name}_prob_soil.tif"

        with rasterio.open(output_tree_path, "w", **meta) as dst:
            dst.write(prob_map_trees, 1)
        with rasterio.open(output_soil_path, "w", **meta) as dst:
            dst.write(prob_map_soil, 1)
        log_message(f"\n  -> Mapas de probabilidade salvos:", log_file)
        log_message(f"     - {output_tree_path} (P de ser arvore)", log_file)
        log_message(f"     - {output_soil_path} (P de ser solo)", log_file)
        if invertido:
            log_message(
                f"     (SWAP APLICADO - mapas foram corrigidos automaticamente)",
                log_file,
            )

    t_pipeline = time.time() - t_pipeline_start
    log_message(
        f"\n  -> Tempo total de geracao de mapas: {format_time(t_pipeline)}", log_file
    )

    return output_tree_path, output_soil_path


if __name__ == "__main__":
    start = time.time()

    # Define o arquivo de log no diretorio de saida
    log_dir = Path(OUTPUT_DIR)
    log_file_path = log_dir / f"{Path(INPUT_IMAGE_PATH).stem}_process_info.txt"
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'#'*80}\n")
        f.write(f"# NOVA EXECUCAO: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"# SCRIPT: generate_probability_maps.py\n")
        f.write(f"{'#'*80}\n\n")

    log_message(f"Arquivo de log: {log_file_path}", log_file_path)
    log_message(f"Imagem de entrada: {INPUT_IMAGE_PATH}", log_file_path)
    log_message(
        f"Pontos de treino de arvores: {TRAINING_TREE_POINTS_PATH}", log_file_path
    )
    log_message(f"Pontos de treino de solo: {TRAINING_SOIL_POINTS_PATH}", log_file_path)
    log_message(f"Features: R, G, B, H, S, V, ExG (7 features)", log_file_path)

    try:
        features, labels = extract_features_for_training(
            INPUT_IMAGE_PATH,
            TRAINING_TREE_POINTS_PATH,
            TRAINING_SOIL_POINTS_PATH,
            BUFFER_SIZE_METERS,
            log_file_path,
        )
        if len(features) > 0:
            model = train_model(features, labels, log_file_path)
            generate_probability_maps(
                INPUT_IMAGE_PATH,
                model,
                TRAINING_TREE_POINTS_PATH,
                TRAINING_SOIL_POINTS_PATH,
                OUTPUT_DIR,
                log_file_path,
            )
        else:
            log_message(
                "Nao foi possivel extrair features para treinamento. Encerrando.",
                log_file_path,
            )

        total_time = time.time() - start
        log_message("\n" + "=" * 60, log_file_path)
        log_message(f"PROCESSO CONCLUIDO em {format_time(total_time)}", log_file_path)
        log_message(f"Termino: {time.strftime('%d/%m/%Y %H:%M:%S')}", log_file_path)
        log_message("=" * 60, log_file_path)
    except Exception as e:
        log_message(f"\nERRO: {str(e)}", log_file_path)
        log_message(
            "Ocorreu um erro. Verifique o log para mais detalhes.", log_file_path
        )
        raise
