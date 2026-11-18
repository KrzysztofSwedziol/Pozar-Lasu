"""Przygotowanie danych wejsciowych symulacji pozaru (Las Wolski, Krakow).

Wejscie : data/osm.json (OSM), data/dem.npy (wysokosci), data/bbox.json
Wyjscie : data/fuel.txt    - mapa paliwa (0..1) na siatce NxN (wezly, [0,100]^2)
          data/fields.txt  - pola: odpowiedz wiatru na wiatr jednostkowy E i N (ux,uy) + nachylenie terenu (sx,sy)
          data/fuel_preview.png - kontrola wizualna klasyfikacji
Konwencja: x -> wschod, y -> polnoc, indeks [ix, iy]; jednostka modelu: 100 j. = 3000 m (1 j. = 30 m).
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
NGRID = 201                         # liczba wezlow (200 elementow)
PX = 1200                           # rozdzielczosc rastra pomocniczego
bb = json.load(open(f'{D}/bbox.json'))
SIZE_M = bb['size_m']
M_PER_PX = SIZE_M / PX

# wartosci paliwa wg klas (0 = brak palnego materialu; model pali tylko przy paliwie > 0.2)
FUEL = dict(open=0.45, built=0.05, scrub=0.80, forest=1.00, water=0.0, building=0.0, road=0.0)

osm = json.load(open(f'{D}/osm.json'))['elements']


def to_px(pt):
    return ((pt['lon'] - bb['W']) / (bb['E'] - bb['W']) * PX, (bb['N'] - pt['lat']) / (bb['N'] - bb['S']) * PX)


def poly_px(geom):
    return [to_px(p) for p in geom]


def stitch(ways):
    """Laczy odcinki (listy punktow) w zamkniete pierscienie."""
    segs = [list(w) for w in ways if len(w) > 1]
    rings = []
    key = lambda p: (round(p['lat'], 7), round(p['lon'], 7))
    while segs:
        ring = segs.pop(0)
        changed = True
        while changed and key(ring[0]) != key(ring[-1]):
            changed = False
            for i, s in enumerate(segs):
                if key(s[0]) == key(ring[-1]):
                    ring += s[1:]
                elif key(s[-1]) == key(ring[-1]):
                    ring += s[::-1][1:]
                elif key(s[-1]) == key(ring[0]):
                    ring = s[:-1] + ring
                elif key(s[0]) == key(ring[0]):
                    ring = s[::-1][:-1] + ring
                else:
                    continue
                segs.pop(i); changed = True; break
        rings.append(ring)
    return rings


def classify(tags):
    n, l, lu, le = tags.get('natural'), tags.get('landuse'), tags.get('landuse'), tags.get('leisure')
    if n == 'water': return 'water'
    if n == 'wood' or lu == 'forest': return 'forest'
    if n in ('scrub', 'heath'): return 'scrub'
    if lu in ('residential', 'industrial', 'retail', 'commercial'): return 'built'
    if n == 'grassland' or lu in ('meadow', 'grass', 'farmland', 'orchard', 'cemetery', 'recreation_ground') \
            or le in ('park', 'pitch', 'garden', 'golf_course'):
        return 'open'
    return None


masks = {k: Image.new('L', (PX, PX), 0) for k in ['open', 'built', 'scrub', 'forest', 'water', 'building', 'road']}
road_w = {'motorway': 25, 'trunk': 25, 'primary': 18, 'secondary': 14, 'tertiary': 12, 'residential': 8, 'unclassified': 8}

for e in osm:
    t = e.get('tags', {})
    if e['type'] == 'way' and 'geometry' in e:
        if 'building' in t:
            ImageDraw.Draw(masks['building']).polygon(poly_px(e['geometry']), fill=255)
        elif 'highway' in t and t['highway'] in road_w:
            w = max(1, int(round(road_w[t['highway']] / M_PER_PX)))
            ImageDraw.Draw(masks['road']).line(poly_px(e['geometry']), fill=255, width=w)
        elif t.get('waterway') == 'river':
            ImageDraw.Draw(masks['water']).line(poly_px(e['geometry']), fill=255, width=int(30 / M_PER_PX))
        else:
            c = classify(t)
            if c and len(e['geometry']) > 2:
                ImageDraw.Draw(masks[c]).polygon(poly_px(e['geometry']), fill=255)
    elif e['type'] == 'relation':
        c = classify(t)
        if not c:
            continue
        outer = [[p for p in m['geometry']] for m in e['members'] if m['type'] == 'way' and m['role'] == 'outer' and 'geometry' in m]
        inner = [[p for p in m['geometry']] for m in e['members'] if m['type'] == 'way' and m['role'] == 'inner' and 'geometry' in m]
        layer = Image.new('L', (PX, PX), 0)
        dr = ImageDraw.Draw(layer)
        for r in stitch(outer):
            if len(r) > 2: dr.polygon(poly_px(r), fill=255)
        for r in stitch(inner):
            if len(r) > 2: dr.polygon(poly_px(r), fill=0)
        masks[c] = Image.fromarray(np.maximum(np.asarray(masks[c]), np.asarray(layer)))

M = {k: np.asarray(v) > 0 for k, v in masks.items()}
fuel_fine = np.full((PX, PX), FUEL['open'])
cls = np.zeros((PX, PX), np.uint8)   # 0 open,1 built,2 scrub,3 forest,4 water,5 building,6 road
for i, k in enumerate(['open', 'built', 'scrub', 'forest', 'water', 'building', 'road']):
    fuel_fine[M[k]] = FUEL[k]
    cls[M[k]] = i

# ------------------------------------------------------------------ probkowanie na siatke modelu
def to_grid(a_img, sigma_px):
    """a_img: raster [row=polnoc..poludnie, col=zachod..wschod] -> siatka [ix, iy] NGRID x NGRID."""
    a = ndimage.gaussian_filter(a_img.astype(float), sigma_px)
    idx = np.linspace(0, a.shape[0] - 1, NGRID)
    X, Y = np.meshgrid(idx, idx, indexing='ij')          # X -> kolumna (x), Y -> wiersz od polnocy
    # ix -> kolumna, iy -> wiersz liczony od POLUDNIA
    vals = ndimage.map_coordinates(a, [a.shape[0] - 1 - Y, X], order=1)
    return vals


cell_px = PX / (NGRID - 1)
fuel = np.clip(to_grid(fuel_fine, cell_px * 0.6), 0, 1)
forestness = to_grid((cls == 3).astype(float) + 0.5 * (cls == 2), cell_px * 1.5)

# ------------------------------------------------------------------ teren
dem = np.load(f'{D}/dem.npy')
dem = ndimage.gaussian_filter(dem, 3)
idx = np.linspace(0, dem.shape[0] - 1, NGRID)
X, Y = np.meshgrid(idx, idx, indexing='ij')
h = ndimage.map_coordinates(dem, [dem.shape[0] - 1 - Y, X], order=1)   # [ix, iy] w metrach n.p.m.
dx = SIZE_M / (NGRID - 1)
sx, sy = np.gradient(h, dx)                                            # nachylenie (m/m); wskazuje "pod gore"

# ------------------------------------------------------------------ pole wiatru: przeplyw w warstwie o grubosci H = Ztop - h
# (zachowanie strumienia div(H u)=0: przyspieszenie nad grzbietami, omijanie wzgorz)
Ztop = h.max() + 250.0
H = Ztop - h
n = NGRID


def lap2d():
    I = sparse.identity(n - 2)
    T = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n - 2, n - 2))
    return (sparse.kron(T, I) + sparse.kron(I, T)) / dx ** 2


L = lap2d().tocsc()


def wind_response(ux0, uy0):
    qx, qy = H * ux0, H * uy0
    div = np.gradient(qx, dx, axis=0) + np.gradient(qy, dx, axis=1)
    psi = np.zeros((n, n))
    psi[1:-1, 1:-1] = spsolve(L, div[1:-1, 1:-1].ravel()).reshape(n - 2, n - 2)
    gx, gy = np.gradient(psi, dx)
    ux, uy = (qx - gx) / H, (qy - gy) / H
    shelter = 1.0 - 0.45 * np.clip(forestness, 0, 1)                    # oslabienie wiatru przez korony drzew
    return ux * shelter, uy * shelter


wEx, wEy = wind_response(1.0, 0.0)
wNx, wNy = wind_response(0.0, 1.0)
print('fuel range', fuel.min(), fuel.max(), 'h', h.min(), h.max(), 'slope max', np.hypot(sx, sy).max())
print('wind E: |u| min/max', np.hypot(wEx, wEy).min(), np.hypot(wEx, wEy).max())

# ------------------------------------------------------------------ zapis
def save_grid(path, arrays):
    with open(path, 'w') as f:
        f.write(f'{n} {n} {len(arrays)}\n')
        for a in arrays:
            np.savetxt(f, a.reshape(1, -1), fmt='%.6g')


save_grid(f'{D}/fuel.txt', [fuel])
save_grid(f'{D}/fields.txt', [wEx, wEy, wNx, wNy, sx, sy])
np.savez(f'{D}/grids.npz', fuel=fuel, h=h, sx=sx, sy=sy, wEx=wEx, wEy=wEy, wNx=wNx, wNy=wNy, forestness=forestness)

# ------------------------------------------------------------------ podglad kontrolny
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sat = np.asarray(Image.open(f'{D}/satellite.png'))
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].imshow(sat); ax[0].set_title('satelita (Esri)')
ax[1].imshow(sat); ax[1].imshow(fuel.T[::-1], extent=[0, PX, PX, 0], cmap='RdYlGn', alpha=0.55, vmin=0, vmax=1); ax[1].set_title('paliwo (siatka modelu)')
c = ax[2].imshow(h.T[::-1], extent=[0, PX, PX, 0], cmap='terrain'); ax[2].set_title('DEM [m n.p.m.]'); plt.colorbar(c, ax=ax[2], fraction=0.04)
ax[2].contour(np.linspace(0, PX, n), np.linspace(PX, 0, n), h.T, 12, colors='k', linewidths=0.4)
plt.tight_layout(); plt.savefig(f'{D}/fuel_preview.png', dpi=70)
