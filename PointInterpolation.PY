"""
point_interpolation.py
======================
Classe utilitária genérica para interpolação de nuvens de pontos 2.5D (x, y, z).

Suporta entrada via:
  - Vetores  : Shapefile, GeoPackage, GeoJSON, FlatGeobuf, CSV (qualquer formato lido por Fiona/GeoPandas)
  - LiDAR    : LAS / LAZ  (via laspy)
  - NumPy    : arrays xs, ys, zs diretamente

Métodos de interpolação disponíveis (enum InterpolationMethod):
  - IDW              : Inverse Distance Weighting  (scipy KDTree)
  - KRIGING_ORDINARY : Ordinary Kriging automático (pykrige)
  - KRIGING_UNIVERSAL: Universal Kriging automático (pykrige)
  - RBF              : Radial Basis Function        (scipy)
  - LINEAR           : Triangulação Linear          (scipy)
  - NEAREST          : Nearest Neighbor             (scipy)
  - SPLINE           : Smoothing Bivariate Spline   (scipy)

Saída: GeoTIFF Float32 georeferenciado.

Dependências obrigatórias:
    pip install numpy scipy rasterio tqdm

Dependências opcionais (instaladas conforme o método / fonte usado):
    pip install geopandas laspy pykrige

Exemplo mínimo:
    from point_interpolation import PointInterpolation, InterpolationMethod, VariogramModel

    pi = PointInterpolation(
        method=InterpolationMethod.KRIGING_ORDINARY,
        resolution=1.0,
        epsg=31982,
    )
    pi.load_vector("meus_pontos.gpkg", z_field="cota")
    pi.interpolate(output_path="mdt.tif")
"""

from __future__ import annotations

import os
import time
import warnings
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Enums públicos
# ---------------------------------------------------------------------------

class InterpolationMethod(Enum):
    """Algoritmo de interpolação a utilizar."""
    IDW               = "idw"
    KRIGING_ORDINARY  = "kriging_ordinary"
    KRIGING_UNIVERSAL = "kriging_universal"
    RBF               = "rbf"
    LINEAR            = "linear"
    NEAREST           = "nearest"
    SPLINE            = "spline"


class VariogramModel(Enum):
    """Modelos de variograma para Kriging (pykrige)."""
    LINEAR      = "linear"
    POWER       = "power"
    GAUSSIAN    = "gaussian"    # mais suave, bom para superfícies contínuas
    SPHERICAL   = "spherical"   # clássico para MDT
    EXPONENTIAL = "exponential" # decaimento mais rápido
    HOLE_EFFECT = "hole_effect"


class RBFKernel(Enum):
    """Funções de base radial (scipy.interpolate.RBFInterpolator)."""
    LINEAR          = "linear"
    THIN_PLATE      = "thin_plate_spline"   # ótimo para MDT — suave e sem oscilações
    CUBIC           = "cubic"
    QUINTIC         = "quintic"
    MULTIQUADRIC    = "multiquadric"
    INV_MULTIQUADRIC= "inverse_multiquadric"
    GAUSSIAN        = "gaussian"


class LidarClassFilter(Enum):
    """Filtros de classe LAS/LAZ padrão ASPRS."""
    ALL             = None   # sem filtro
    GROUND          = 2
    LOW_VEGETATION  = 3
    MED_VEGETATION  = 4
    HIGH_VEGETATION = 5
    BUILDING        = 6
    WATER           = 9
    GROUND_AND_WATER = -1    # tratamento especial: classes 2 + 9


class NoDataStrategy(Enum):
    """Como preencher pixels sem dados após interpolação."""
    LEAVE_NAN   = "leave_nan"    # mantém NaN (padrão)
    NEAREST     = "nearest"      # propaga valor do vizinho mais próximo
    MEAN        = "mean"         # preenche com média global


class OutlierMethod(Enum):
    """Método de remoção de outliers em Z."""
    NONE    = "none"
    ZSCORE  = "zscore"           # remove pontos fora de ±n desvios padrão
    IQR     = "iqr"              # remove pontos fora de [Q1-k*IQR, Q3+k*IQR]


# ---------------------------------------------------------------------------
# Dataclass de configuração (todos os parâmetros com defaults sensatos)
# ---------------------------------------------------------------------------

@dataclass
class InterpolationConfig:
    # ---- Geral ------------------------------------------------------------
    method: InterpolationMethod         = InterpolationMethod.KRIGING_ORDINARY
    resolution: float                   = 1.0          # metros por pixel
    epsg: int                           = 4326         # CRS de saída

    # ---- IDW --------------------------------------------------------------
    idw_k_neighbors: int                = 16           # vizinhos usados
    idw_power: float                    = 2.0          # expoente de distância
    idw_smooth_sigma: float             = 1.0          # suavização Gaussian pós-IDW (0=desliga)

    # ---- Kriging ----------------------------------------------------------
    kriging_variogram: VariogramModel   = VariogramModel.SPHERICAL
    kriging_nlags: int                  = 20           # lags para ajuste do variograma
    kriging_weight: bool                = True         # pondera lags por nº de pares
    kriging_max_points: int             = 60_000       # subsample acima deste limite
    kriging_backend: str                = "vectorized" # 'vectorized' ou 'loop' (menos RAM)

    # ---- RBF --------------------------------------------------------------
    rbf_kernel: RBFKernel               = RBFKernel.THIN_PLATE
    rbf_smoothing: float                = 0.0          # 0 = interpolação exata; >0 = suavizado
    rbf_max_points: int                 = 30_000       # subsample (RBF é O(n³))
    rbf_neighbors: Optional[int]        = 50           # None = global (mais preciso, mais lento)
    rbf_epsilon: float                  = 1.0          # escala para kernels que usam epsilon

    # ---- Spline -----------------------------------------------------------
    spline_kx: int                      = 3            # grau spline em X (1–5)
    spline_ky: int                      = 3            # grau spline em Y
    spline_smoothing: Optional[float]   = None         # None = automático

    # ---- Chunks / memória -------------------------------------------------
    chunk_size: int                     = 512          # pixels por chunk (quadrado)

    # ---- Outliers ---------------------------------------------------------
    outlier_method: OutlierMethod       = OutlierMethod.ZSCORE
    outlier_zscore_thresh: float        = 3.5          # threshold para ZSCORE
    outlier_iqr_k: float                = 2.5          # multiplicador para IQR

    # ---- NoData -----------------------------------------------------------
    nodata_strategy: NoDataStrategy     = NoDataStrategy.LEAVE_NAN

    # ---- LiDAR ------------------------------------------------------------
    lidar_class_filter: LidarClassFilter = LidarClassFilter.GROUND
    lidar_z_field: str                  = "z"          # campo de elevação no LAS

    # ---- Saída ------------------------------------------------------------
    compress: str                       = "lzw"
    tiled: bool                         = True
    verbose: bool                       = True


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class PointInterpolation:
    """
    Interpolação de nuvem de pontos para raster GeoTIFF.

    Parâmetros
    ----------
    method : InterpolationMethod
        Algoritmo de interpolação.
    resolution : float
        Tamanho do pixel em unidades do CRS (metros se projetado).
    epsg : int
        EPSG do CRS de saída / entrada.
    **kwargs
        Qualquer parâmetro de InterpolationConfig pode ser sobrescrito.

    Exemplos
    --------
    # Kriging com variograma gaussiano
    pi = PointInterpolation(
        method=InterpolationMethod.KRIGING_ORDINARY,
        resolution=0.5,
        epsg=31982,
        kriging_variogram=VariogramModel.GAUSSIAN,
    )
    pi.load_vector("pontos.gpkg", z_field="elevation")
    pi.interpolate("mdt_kriging.tif")

    # IDW a partir de LAS
    pi = PointInterpolation(
        method=InterpolationMethod.IDW,
        resolution=1.0,
        epsg=32722,
        idw_k_neighbors=24,
        lidar_class_filter=LidarClassFilter.GROUND,
    )
    pi.load_lidar("nuvem.laz")
    pi.interpolate("mdt_idw.tif")

    # RBF a partir de arrays NumPy
    pi = PointInterpolation(method=InterpolationMethod.RBF, resolution=2.0, epsg=4326)
    pi.load_arrays(xs, ys, zs)
    pi.interpolate("mdt_rbf.tif")
    """

    def __init__(
        self,
        method: InterpolationMethod = InterpolationMethod.KRIGING_ORDINARY,
        resolution: float = 1.0,
        epsg: int = 4326,
        **kwargs,
    ):
        cfg_kwargs = {k: v for k, v in kwargs.items()
                      if k in InterpolationConfig.__dataclass_fields__}
        self.config = InterpolationConfig(
            method=method,
            resolution=resolution,
            epsg=epsg,
            **cfg_kwargs,
        )
        self._xs: Optional[np.ndarray] = None
        self._ys: Optional[np.ndarray] = None
        self._zs: Optional[np.ndarray] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_vector(
        self,
        path: str,
        z_field: str = "z",
        layer: Optional[str] = None,
        crs_override: Optional[int] = None,
    ) -> "PointInterpolation":
        """
        Carrega pontos de arquivo vetorial (GPKG, SHP, GeoJSON, FlatGeobuf, CSV…).

        Parâmetros
        ----------
        path : str
            Caminho para o arquivo vetorial.
        z_field : str
            Nome do atributo com o valor Z (elevação/cota).
        layer : str, opcional
            Camada dentro do arquivo (útil para GPKG multi-layer).
        crs_override : int, opcional
            Força um EPSG de entrada (útil quando o arquivo não tem CRS definido).
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas é necessário para load_vector. "
                              "Instale com: pip install geopandas")

        self._log(f"Carregando vetor: {path}")
        kw = {"layer": layer} if layer else {}
        gdf = gpd.read_file(path, **kw)

        if crs_override:
            gdf = gdf.set_crs(epsg=crs_override, allow_override=True)

        target_epsg = crs_override or self.config.epsg
        if gdf.crs and gdf.crs.to_epsg() != target_epsg:
            self._log(f"  Reprojetando de EPSG:{gdf.crs.to_epsg()} → EPSG:{target_epsg}")
            gdf = gdf.to_crs(epsg=target_epsg)

        if z_field not in gdf.columns:
            # tenta extrair Z da geometria
            try:
                gdf[z_field] = gdf.geometry.z
                self._log(f"  Campo Z extraído da geometria 3D.")
            except Exception:
                raise ValueError(
                    f"Campo '{z_field}' não encontrado e geometria não possui Z. "
                    f"Colunas disponíveis: {list(gdf.columns)}"
                )

        self._xs = gdf.geometry.x.to_numpy(dtype=np.float64)
        self._ys = gdf.geometry.y.to_numpy(dtype=np.float64)
        self._zs = gdf[z_field].to_numpy(dtype=np.float64)
        self._log(f"  {len(self._xs):,} pontos carregados.")
        return self

    def load_lidar(
        self,
        path: str,
        class_filter: Optional[LidarClassFilter] = None,
    ) -> "PointInterpolation":
        """
        Carrega nuvem de pontos LAS / LAZ.

        Parâmetros
        ----------
        path : str
            Caminho para o arquivo .las ou .laz.
        class_filter : LidarClassFilter, opcional
            Sobrescreve config.lidar_class_filter para este carregamento.
        """
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy é necessário para load_lidar. "
                              "Instale com: pip install laspy[lazrs]")

        cf = class_filter or self.config.lidar_class_filter
        self._log(f"Carregando LiDAR: {path}  (filtro={cf.name})")

        las = laspy.read(path)
        classifications = np.array(las.classification)

        if cf == LidarClassFilter.ALL or cf.value is None:
            mask = np.ones(len(las.x), dtype=bool)
        elif cf == LidarClassFilter.GROUND_AND_WATER:
            mask = np.isin(classifications, [2, 9])
        else:
            mask = classifications == cf.value

        scale = las.header.scales
        offset = las.header.offsets

        self._xs = (np.array(las.X)[mask] * scale[0] + offset[0]).astype(np.float64)
        self._ys = (np.array(las.Y)[mask] * scale[1] + offset[1]).astype(np.float64)
        self._zs = (np.array(las.Z)[mask] * scale[2] + offset[2]).astype(np.float64)

        self._log(f"  {mask.sum():,} / {len(mask):,} pontos após filtro de classe.")
        return self

    def load_arrays(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
    ) -> "PointInterpolation":
        """
        Carrega pontos diretamente de arrays NumPy.

        Parâmetros
        ----------
        xs, ys, zs : np.ndarray
            Arrays 1D com coordenadas X, Y e valor Z.
        """
        self._xs = np.asarray(xs, dtype=np.float64)
        self._ys = np.asarray(ys, dtype=np.float64)
        self._zs = np.asarray(zs, dtype=np.float64)
        self._log(f"{len(self._xs):,} pontos carregados via arrays.")
        return self

    # ------------------------------------------------------------------
    # Pré-processamento
    # ------------------------------------------------------------------

    def _remove_outliers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, zs = self._xs.copy(), self._ys.copy(), self._zs.copy()
        method = self.config.outlier_method

        if method == OutlierMethod.NONE:
            return xs, ys, zs

        if method == OutlierMethod.ZSCORE:
            mean, std = np.nanmean(zs), np.nanstd(zs)
            thresh = self.config.outlier_zscore_thresh
            mask = np.abs(zs - mean) <= thresh * std

        elif method == OutlierMethod.IQR:
            q1, q3 = np.nanpercentile(zs, 25), np.nanpercentile(zs, 75)
            iqr = q3 - q1
            k = self.config.outlier_iqr_k
            mask = (zs >= q1 - k * iqr) & (zs <= q3 + k * iqr)

        removed = (~mask).sum()
        if removed:
            self._log(f"  Outliers removidos ({method.name}): {removed:,} pontos")
        return xs[mask], ys[mask], zs[mask]

    def _compute_bounds(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> Tuple[float, float, float, float]:
        if self._bounds:
            return self._bounds
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def _fill_nodata(self, raster: np.ndarray) -> np.ndarray:
        strategy = self.config.nodata_strategy
        if strategy == NoDataStrategy.LEAVE_NAN:
            return raster

        nan_mask = np.isnan(raster)
        if not nan_mask.any():
            return raster

        if strategy == NoDataStrategy.MEAN:
            raster[nan_mask] = np.nanmean(raster)

        elif strategy == NoDataStrategy.NEAREST:
            from scipy.ndimage import distance_transform_edt
            _, idx = distance_transform_edt(nan_mask, return_indices=True)
            raster[nan_mask] = raster[tuple(idx[:, nan_mask])]

        return raster

    # ------------------------------------------------------------------
    # Interpoladores
    # ------------------------------------------------------------------

    def _interpolate_idw(
        self, xs, ys, zs, grid_x, grid_y
    ) -> np.ndarray:
        from scipy.spatial import KDTree
        from scipy.ndimage import gaussian_filter

        cfg = self.config
        tree = KDTree(np.column_stack([xs, ys]))
        query = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        k = min(cfg.idw_k_neighbors, len(xs))
        distances, indices = tree.query(query, k=k, workers=-1)

        if distances.ndim == 1:
            distances, indices = distances[:, None], indices[:, None]

        weights = 1.0 / (distances ** cfg.idw_power + 1e-12)
        exact = distances[:, 0] == 0
        weights[exact] = 0.0
        weights[exact, 0] = 1.0

        z_nn = zs[indices]
        w_sum = weights.sum(axis=1)
        result = np.full(grid_x.size, np.nan, dtype=np.float32)
        valid = w_sum > 0
        result[valid] = (z_nn * weights).sum(axis=1)[valid] / w_sum[valid]
        grid = result.reshape(grid_x.shape).astype(np.float32)

        if cfg.idw_smooth_sigma > 0:
            nan_mask = np.isnan(grid)
            tmp = grid.copy()
            tmp[nan_mask] = np.nanmean(grid)
            sm = gaussian_filter(tmp, sigma=cfg.idw_smooth_sigma, mode="nearest")
            sm_w = gaussian_filter(
                (~nan_mask).astype(np.float32),
                sigma=cfg.idw_smooth_sigma, mode="nearest"
            )
            with np.errstate(invalid="ignore"):
                grid = np.where(sm_w > 0.01, sm / sm_w, np.nan).astype(np.float32)

        return grid

    def _interpolate_kriging(
        self, xs, ys, zs, pixel_x, pixel_y, method: str
    ) -> np.ndarray:
        try:
            from pykrige.ok import OrdinaryKriging
            from pykrige.uk import UniversalKriging
        except ImportError:
            raise ImportError("pykrige é necessário para Kriging. "
                              "Instale com: pip install pykrige")

        cfg = self.config
        max_pts = cfg.kriging_max_points

        if len(xs) > max_pts:
            idx = np.random.default_rng(42).choice(len(xs), max_pts, replace=False)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
            self._log(f"  Kriging: subsample → {max_pts:,} pontos")

        variogram = cfg.kriging_variogram.value
        kw = dict(
            variogram_model=variogram,
            nlags=cfg.kriging_nlags,
            weight=cfg.kriging_weight,
            enable_plotting=False,
            verbose=False,
        )

        KrigClass = OrdinaryKriging if method == "ordinary" else UniversalKriging
        ok = KrigClass(xs, ys, zs, **kw)
        self._log(f"  Variograma ajustado — modelo={variogram}  "
                  f"range={ok.variogram_model_parameters[1]:.2f}  "
                  f"sill={ok.variogram_model_parameters[0]:.4f}  "
                  f"nugget={ok.variogram_model_parameters[2]:.4f}")

        z_pred, _ = ok.execute(
            "grid", pixel_x, pixel_y, backend=cfg.kriging_backend
        )
        return z_pred.astype(np.float32)

    def _interpolate_rbf(
        self, xs, ys, zs, grid_x, grid_y
    ) -> np.ndarray:
        from scipy.interpolate import RBFInterpolator

        cfg = self.config
        max_pts = cfg.rbf_max_points

        if len(xs) > max_pts:
            idx = np.random.default_rng(42).choice(len(xs), max_pts, replace=False)
            xs, ys, zs = xs[idx], ys[idx], zs[idx]
            self._log(f"  RBF: subsample → {max_pts:,} pontos")

        rbf = RBFInterpolator(
            np.column_stack([xs, ys]),
            zs,
            kernel=cfg.rbf_kernel.value,
            smoothing=cfg.rbf_smoothing,
            neighbors=cfg.rbf_neighbors,
            epsilon=cfg.rbf_epsilon,
        )
        pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        result = rbf(pts).reshape(grid_x.shape).astype(np.float32)
        return result

    def _interpolate_scipy_griddata(
        self, xs, ys, zs, grid_x, grid_y, method_str: str
    ) -> np.ndarray:
        from scipy.interpolate import griddata

        pts = np.column_stack([xs, ys])
        result = griddata(
            pts, zs,
            (grid_x, grid_y),
            method=method_str,
            fill_value=np.nan,
        )
        return result.astype(np.float32)

    def _interpolate_spline(
        self, xs, ys, zs, pixel_x, pixel_y
    ) -> np.ndarray:
        from scipy.interpolate import SmoothBivariateSpline

        cfg = self.config
        kw = {}
        if cfg.spline_smoothing is not None:
            kw["s"] = cfg.spline_smoothing

        spl = SmoothBivariateSpline(
            xs, ys, zs,
            kx=cfg.spline_kx,
            ky=cfg.spline_ky,
            **kw,
        )
        z = spl(pixel_x, pixel_y)
        return z.astype(np.float32)   # shape: (len(pixel_x), len(pixel_y))

    # ------------------------------------------------------------------
    # Engine principal
    # ------------------------------------------------------------------

    def interpolate(
        self,
        output_path: str,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> str:
        """
        Executa a interpolação e grava o GeoTIFF.

        Parâmetros
        ----------
        output_path : str
            Caminho do arquivo de saída (.tif).
        bounds : tuple (left, bottom, right, top), opcional
            Limites da área de interesse. Se None, usa a extensão dos pontos.

        Retorna
        -------
        str
            Caminho do arquivo gerado.
        """
        import rasterio
        from rasterio.transform import from_origin

        if self._xs is None:
            raise RuntimeError("Nenhum dado carregado. Use load_vector(), load_lidar() ou load_arrays().")

        t0 = time.time()
        cfg = self.config

        self._log(f"\n{'='*60}")
        self._log(f"INTERPOLAÇÃO: {cfg.method.name}  |  resolução={cfg.resolution}m  |  EPSG:{cfg.epsg}")
        self._log(f"{'='*60}")

        # 1. Pré-processamento
        xs, ys, zs = self._remove_outliers()
        self._log(f"Pontos válidos: {len(xs):,}")

        if bounds:
            self._bounds = bounds
        left, bottom, right, top = self._compute_bounds(xs, ys)

        width  = int(np.ceil((right - left) / cfg.resolution))
        height = int(np.ceil((top - bottom) / cfg.resolution))
        self._log(f"Grid: {width} x {height} px  ({width*height:,} pixels)")

        transform = from_origin(left, top, cfg.resolution, cfg.resolution)
        meta = {
            "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
            "width": width, "height": height, "count": 1,
            "crs": f"EPSG:{cfg.epsg}", "transform": transform,
            "compress": cfg.compress, "tiled": cfg.tiled,
            "blockxsize": cfg.chunk_size, "blockysize": cfg.chunk_size,
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        method = cfg.method

        # 2. Métodos que operam em grade completa (sem chunking interno)
        if method in (
            InterpolationMethod.KRIGING_ORDINARY,
            InterpolationMethod.KRIGING_UNIVERSAL,
            InterpolationMethod.SPLINE,
        ):
            pixel_x = left + (np.arange(width)  + 0.5) * cfg.resolution
            pixel_y = top  - (np.arange(height) + 0.5) * cfg.resolution

            if method == InterpolationMethod.KRIGING_ORDINARY:
                grid = self._interpolate_kriging(xs, ys, zs, pixel_x, pixel_y, "ordinary")
            elif method == InterpolationMethod.KRIGING_UNIVERSAL:
                grid = self._interpolate_kriging(xs, ys, zs, pixel_x, pixel_y, "universal")
            elif method == InterpolationMethod.SPLINE:
                # SmoothBivariateSpline retorna (ny, nx) — transpor
                grid = self._interpolate_spline(xs, ys, zs, pixel_x, pixel_y).T

            grid = self._fill_nodata(grid)

            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(grid, 1)

        # 3. Métodos que operam por chunk
        else:
            from tqdm import tqdm

            num_ch = int(np.ceil(height / cfg.chunk_size))
            num_cw = int(np.ceil(width  / cfg.chunk_size))
            total  = num_ch * num_cw
            full_raster = np.full((height, width), np.nan, dtype=np.float32)

            with tqdm(total=total, desc=method.name, unit="chunk") as pbar:
                for r0 in range(0, height, cfg.chunk_size):
                    r1 = min(r0 + cfg.chunk_size, height)
                    for c0 in range(0, width, cfg.chunk_size):
                        c1 = min(c0 + cfg.chunk_size, width)

                        px = left + (c0 + np.arange(c1 - c0) + 0.5) * cfg.resolution
                        py = top  - (r0 + np.arange(r1 - r0) + 0.5) * cfg.resolution
                        gx, gy = np.meshgrid(px, py)

                        if method == InterpolationMethod.IDW:
                            chunk = self._interpolate_idw(xs, ys, zs, gx, gy)

                        elif method == InterpolationMethod.RBF:
                            chunk = self._interpolate_rbf(xs, ys, zs, gx, gy)

                        elif method in (InterpolationMethod.LINEAR,
                                        InterpolationMethod.NEAREST):
                            chunk = self._interpolate_scipy_griddata(
                                xs, ys, zs, gx, gy, method.value
                            )

                        full_raster[r0:r1, c0:c1] = chunk
                        pbar.update(1)

            full_raster = self._fill_nodata(full_raster)

            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(full_raster, 1)

        # 4. Estatísticas finais
        with rasterio.open(output_path) as src:
            data = src.read(1)
        valid = data[~np.isnan(data)]
        if len(valid):
            self._log(f"\nEstatísticas do raster gerado:")
            self._log(f"  Z mín   : {valid.min():.3f}")
            self._log(f"  Z máx   : {valid.max():.3f}")
            self._log(f"  Z médio : {valid.mean():.3f}")
            self._log(f"  Z std   : {valid.std():.3f}")
            pct = 100 * len(valid) / data.size
            self._log(f"  Cobertura: {len(valid):,} / {data.size:,} ({pct:.1f}%)")

        elapsed = time.time() - t0
        self._log(f"\nTempo total: {self._fmt_time(elapsed)}")
        self._log(f"Arquivo gerado: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_bounds(self, left: float, bottom: float, right: float, top: float) -> "PointInterpolation":
        """Define manualmente os limites da área de interpolação."""
        self._bounds = (left, bottom, right, top)
        return self

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"

    def __repr__(self) -> str:
        n = len(self._xs) if self._xs is not None else 0
        return (
            f"PointInterpolation("
            f"method={self.config.method.name}, "
            f"resolution={self.config.resolution}, "
            f"epsg={self.config.epsg}, "
            f"points={n:,})"
        )


# ---------------------------------------------------------------------------
# Exemplos de uso (executar como script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Exemplo 1: Kriging Ordinário a partir de GeoPackage ──────────────
    pi = PointInterpolation(
        method=InterpolationMethod.KRIGING_ORDINARY,
        resolution=1.0,
        epsg=31982,
        kriging_variogram=VariogramModel.SPHERICAL,
        kriging_max_points=50_000,
        outlier_method=OutlierMethod.ZSCORE,
        nodata_strategy=NoDataStrategy.NEAREST,
    )
    # pi.load_vector("pontos.gpkg", z_field="cota")
    # pi.interpolate("mdt_kriging.tif")

    # ── Exemplo 2: RBF (Thin Plate Spline) a partir de LAS ───────────────
    pi2 = PointInterpolation(
        method=InterpolationMethod.RBF,
        resolution=0.5,
        epsg=32722,
        rbf_kernel=RBFKernel.THIN_PLATE,
        rbf_smoothing=0.1,
        lidar_class_filter=LidarClassFilter.GROUND,
        outlier_method=OutlierMethod.IQR,
    )
    # pi2.load_lidar("nuvem.laz")
    # pi2.interpolate("mdt_rbf.tif")

    # ── Exemplo 3: IDW suavizado a partir de arrays ───────────────────────
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1000, 5000)
    ys = rng.uniform(0, 1000, 5000)
    zs = np.sin(xs / 200) * np.cos(ys / 200) * 100

    pi3 = PointInterpolation(
        method=InterpolationMethod.IDW,
        resolution=5.0,
        epsg=32722,
        idw_k_neighbors=24,
        idw_power=2.0,
        idw_smooth_sigma=1.5,
        nodata_strategy=NoDataStrategy.NEAREST,
    )
    pi3.load_arrays(xs, ys, zs)
    pi3.interpolate("/tmp/mdt_idw_demo.tif")

    print("\nClasse PointInterpolation pronta para uso.")