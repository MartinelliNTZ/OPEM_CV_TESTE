import rasterio
import numpy as np

print("=== prob_map_trees.tif ===")
with rasterio.open(r'd:\TESTES_PYTHON\OPEM_CV_TESTE\v10_final\process\prob_map_trees.tif') as src:
    data = src.read(1)
    print(f"  min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    print(f"  Pixels > 0.5: {np.sum(data > 0.5):,} / {data.size:,} ({100*np.sum(data>0.5)/data.size:.1f}%)")

print("\n=== prob_map_soil.tif ===")
with rasterio.open(r'd:\TESTES_PYTHON\OPEM_CV_TESTE\v10_final\process\prob_map_soil.tif') as src:
    data = src.read(1)
    print(f"  min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    print(f"  Pixels > 0.5: {np.sum(data > 0.5):,} / {data.size:,} ({100*np.sum(data>0.5)/data.size:.1f}%)")

# Verify complement
with rasterio.open(r'd:\TESTES_PYTHON\OPEM_CV_TESTE\v10_final\process\prob_map_trees.tif') as t:
    d_t = t.read(1)
with rasterio.open(r'd:\TESTES_PYTHON\OPEM_CV_TESTE\v10_final\process\prob_map_soil.tif') as s:
    d_s = s.read(1)

print("\n=== Sum verification ===")
sum_data = d_t + d_s
print(f"  min sum: {sum_data.min():.4f}, max sum: {sum_data.max():.4f}, mean sum: {sum_data.mean():.4f}")

# Check if they might be swapped relative to intuitive expectation
# The images show 80% of pixels as trees - is that realistic?
print(f"\n=== Realistic check ===")
print(f"  prob_map_trees > 0.5: {np.sum(d_t > 0.5)/d_t.size*100:.1f}% of image has P(tree)>0.5")
print(f"  prob_map_soil > 0.5: {np.sum(d_s > 0.5)/d_s.size*100:.1f}% of image has P(soil)>0.5")

# Check the initial TIF to understand land cover visually
print(f"\n=== Quick look at input image ===")
with rasterio.open(r'D:\TESTES_PYTHON\OPEM_CV_TESTE\imaru\Imaru2.tif') as src:
    b1 = src.read(1)
    b2 = src.read(2)
    b3 = src.read(3)
    ndvi = (b3.astype(float) - b1.astype(float)) / (b3.astype(float) + b1.astype(float) + 1)
    print(f"  NIR-R index (proxy vegetation): mean={ndvi.mean():.4f}")
    # If mean NDVI-like > 0.3, the area is mostly forest/vegetation
    veg_pct = np.sum(ndvi > 0.2) / ndvi.size * 100
    print(f"  Pixels with NIR-R > 0.2 (likely vegetation): {veg_pct:.1f}%")