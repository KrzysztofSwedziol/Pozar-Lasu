"""Pobiera dane dla Lasu Wolskiego (Krakow): podklad satelitarny (Esri), DEM (AWS Terrarium), OSM (Overpass)."""
import io, json, math, os, sys, time, urllib.request, urllib.parse
import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT, exist_ok=True)
LAT0, LON0 = 50.0535, 19.8470     # srodek obszaru (Las Wolski, Krakow)
SIZE_M = 3000.0                   # bok obszaru [m]
UA = 'student-sim-agh/1.0'

def get(url, data=None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={'User-Agent': UA})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:
            print('retry', url[:60], e); time.sleep(3 * (i + 1))
    raise RuntimeError(url)

def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y

# bbox
dlat = SIZE_M / 2 / 111320.0
dlon = SIZE_M / 2 / (111320.0 * math.cos(math.radians(LAT0)))
S, N, W, E = LAT0 - dlat, LAT0 + dlat, LON0 - dlon, LON0 + dlon
json.dump(dict(S=S, N=N, W=W, E=E, lat0=LAT0, lon0=LON0, size_m=SIZE_M), open(f'{OUT}/bbox.json', 'w'))
print('bbox', S, N, W, E)

def mosaic(url_fmt, z):
    x0, y1 = lonlat_to_tile(W, N, z); x1, y0 = lonlat_to_tile(E, S, z)
    # (x0,y1)=gorny lewy, (x1,y0)=dolny prawy w ukladzie kafelkow
    tx0, tx1 = int(x0), int(x1); ty0, ty1 = int(y1), int(y0)
    big = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            t = Image.open(io.BytesIO(get(url_fmt.format(z=z, x=tx, y=ty)))).convert('RGB')
            big.paste(t, ((tx - tx0) * 256, (ty - ty0) * 256))
    box = ((x0 - tx0) * 256, (y1 - ty0) * 256, (x1 - tx0) * 256, (y0 - ty0) * 256)
    return big.crop(tuple(int(round(b)) for b in box))

sat = mosaic('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 17)
sat = sat.resize((1200, 1200), Image.LANCZOS); sat.save(f'{OUT}/satellite.png'); print('sat', sat.size)

dem_rgb = mosaic('https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png', 14)
a = np.asarray(dem_rgb).astype(float)
h = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
h = np.asarray(Image.fromarray(h.astype(np.float32)).resize((400, 400), Image.BICUBIC))
np.save(f'{OUT}/dem.npy', h.astype(np.float32)); print('dem', h.shape, h.min(), h.max())

bb = f'{S},{W},{N},{E}'
q = f'''[out:json][timeout:90];
(
 way["natural"~"wood|scrub|grassland|water|heath"]({bb});
 way["landuse"~"forest|meadow|grass|farmland|residential|industrial|retail|commercial|orchard|cemetery|recreation_ground"]({bb});
 way["leisure"~"park|pitch|garden|golf_course|nature_reserve"]({bb});
 way["waterway"~"river|stream|canal"]({bb});
 way["building"]({bb});
 way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|track|path|footway"]({bb});
 relation["natural"~"wood|water|scrub|grassland"]({bb});
 relation["landuse"~"forest|meadow|grass|residential|farmland"]({bb});
);
out geom;'''
import subprocess
raw = b''
for i in range(8):
    for ep in ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter']:
        raw = subprocess.run(['curl','-s','-m','120','-A',UA,'--data-urlencode','data='+q,ep],capture_output=True).stdout
        if raw.startswith(b'{'): break
    if raw.startswith(b'{'): break
    print('overpass busy, retry', i); time.sleep(10*(i+1))
open(f'{OUT}/osm.json', 'wb').write(raw)
print('osm elements', len(json.loads(raw)['elements']))
