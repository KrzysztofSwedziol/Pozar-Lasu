"""Generowanie zsynchronizowanych animacji GIF oraz rycin do sprawozdania LaTeX.
Projekt: Symulacja rozprzestrzeniania sie pozaru lasu (Las Wolski, Krakow)
Metoda: IGA-ADS (Isogeometric Analysis FEM + Alternating Direction Solver)
"""
import glob
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WORK_RESULTS = os.path.join(BASE_DIR, 'work', 'results')
GIFS_DIR = os.path.join(BASE_DIR, 'gifs')
REPORT_DIR = os.path.join(BASE_DIR, 'report')

os.makedirs(GIFS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 1. Wczytanie danych bazowych
grids = np.load(os.path.join(DATA_DIR, 'grids.npz'))
fuel_base = grids['fuel']
h_dem = grids['h']
wEx = grids['wEx']
wEy = grids['wEy']
wNx = grids['wNx']
wNy = grids['wNy']
sx = grids['sx']
sy = grids['sy']
forestness = grids['forestness']

sat_img = Image.open(os.path.join(DATA_DIR, 'satellite.png'))
sat_arr = np.asarray(sat_img)

bbox = json.load(open(os.path.join(DATA_DIR, 'bbox.json')))
SIZE_M = bbox['size_m']  # 3000 m

# 2. Wyszukanie wszystkich zarejestrowanych krokow czasowych
u_files = glob.glob(os.path.join(WORK_RESULTS, 'out_*.data'))
step_numbers = sorted([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in u_files])
print(f'Znaleziono {len(step_numbers)} plikow danych (kroki od {step_numbers[0]} do {step_numbers[-1]})')

DT = 0.0003
W_E_BASE = 38.0
W_N_BASE = 22.0
GUST_AMP = 8.0
GUST_FREQ = 1.8

# Wyznaczenie wymiaru siatki z pliku
sample = np.loadtxt(os.path.join(WORK_RESULTS, 'out_0.data'))
N_AXIS = int(np.round(np.sqrt(sample.shape[0])))
print(f'Rozmiar siatki wynikowej: {N_AXIS}x{N_AXIS}')

def load_scalar_field(path):
    d = np.loadtxt(path)
    return d[:, 2].reshape((N_AXIS, N_AXIS))

print('Ladowanie wszystkich pol temperatury i paliwa do pamieci...')
all_steps = step_numbers
all_times = [s * DT for s in all_steps]
all_T = []
all_F = []

for s in all_steps:
    u_f = os.path.join(WORK_RESULTS, f'out_{s}.data')
    f_f = os.path.join(WORK_RESULTS, f'fuel_{s}.data')
    all_T.append(load_scalar_field(u_f))
    all_F.append(load_scalar_field(f_f))

all_T = np.array(all_T)
all_F = np.array(all_F)
print(f'Zaladowano {len(all_steps)} klatek (T shape: {all_T.shape}, Fuel shape: {all_F.shape})')

# Siatka wektorowa wiatru
SKIP = 7
x_quiv = np.linspace(0, 100, grids['fuel'].shape[0])[::SKIP]
y_quiv = np.linspace(0, 100, grids['fuel'].shape[1])[::SKIP]
XQ, YQ = np.meshgrid(x_quiv, y_quiv, indexing='ij')
wEx_q = wEx[::SKIP, ::SKIP]
wEy_q = wEy[::SKIP, ::SKIP]
wNx_q = wNx[::SKIP, ::SKIP]
wNy_q = wNy[::SKIP, ::SKIP]
h_q = h_dem[::SKIP, ::SKIP]

# Metryki czasowe
T_max_series = []
T_mean_series = []
burned_area_ha = []
active_fire_area_ha = []
wind_speed_series = []

CELL_AREA_HA = (SIZE_M / (N_AXIS - 1)) ** 2 / 10000.0  # ha na komorke
F_0 = all_F[0]

for idx, t in enumerate(all_times):
    T_curr = all_T[idx]
    F_curr = all_F[idx]
    T_max_series.append(np.max(T_curr))
    T_mean_series.append(np.mean(T_curr))
    
    active_cells = np.sum(T_curr > 600.0)
    active_fire_area_ha.append(active_cells * CELL_AREA_HA)
    
    burned_cells = np.sum((F_0 > 0.3) & (F_curr < 0.2))
    burned_area_ha.append(burned_cells * CELL_AREA_HA)
    
    WE_t = W_E_BASE + GUST_AMP * np.cos(GUST_FREQ * t)
    WN_t = W_N_BASE + GUST_AMP * np.sin(GUST_FREQ * t)
    wind_speed_series.append(np.hypot(WE_t, WN_t))

# Kolorystyka
fire_colors = [
    (0.0, 0.0, 0.0, 0.0),      # < 350K niewidoczny
    (0.8, 0.1, 0.0, 0.45),     # 500K ciemnoczerwony
    (1.0, 0.3, 0.0, 0.75),     # 750K czerwono-pomaranczowy
    (1.0, 0.7, 0.0, 0.90),     # 1000K jasnopomaranczowy/zolty
    (1.0, 1.0, 0.8, 1.0)       # 1300K bialo-zolty zar
]
fire_cmap = mcolors.LinearSegmentedColormap.from_list('fire_alpha', fire_colors)
fire_norm = mcolors.Normalize(vmin=320, vmax=1320)

fuel_colors = [
    '#2b2b2b',   # 0.0 spalone / brak
    '#5c4033',   # 0.1 zweglone resztki
    '#d4b26f',   # 0.3 trawy / zabudowa
    '#8cb04e',   # 0.5 zarosla
    '#2e7d32',   # 0.8 gesty las
    '#1b5e20'    # 1.0 pelny drzewostan
]
fuel_cmap = mcolors.LinearSegmentedColormap.from_list('fuel_state', fuel_colors)

landmarks = [
    ('Kopiec Piłsudskiego', 49.0, 51.5),
    ('Ogród Zoologiczny', 58.0, 48.0),
    ('Bielany', 18.0, 32.0),
    ('Wola Justowska', 75.0, 78.0),
    ('Polana Lea', 38.0, 54.0)
]

def save_animated_gif(frames, path, duration_ms=60):
    print(f'Zapis {len(frames)} klatek animacji do {os.path.basename(path)}...')
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True)
    print(f'Pomyslnie zapisano: {path} (Rozmiar: {os.path.getsize(path)/1024:.1f} KB)')

# -------------------------------------------------------------
# GIF 1: Fala pozaru na podkladzie satelitarnym
# -------------------------------------------------------------
print('\n[1/4] Renderowanie klatek GIF 1: Fala pozaru na satelicie...')
frames_gif1 = []
fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=95)

for idx, (step, t) in enumerate(zip(all_steps, all_times)):
    ax.clear()
    ax.imshow(sat_arr, extent=[0, 100, 0, 100])
    
    T_curr = all_T[idx]
    ax.imshow(T_curr.T, extent=[0, 100, 0, 100], origin='lower',
              cmap=fire_cmap, norm=fire_norm, interpolation='bicubic')
    
    if np.max(T_curr) > 700:
        ax.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T,
                   levels=[800, 1100], colors=['#ffea00', '#ffffff'], linewidths=[1.2, 1.8])
    
    for name, lx, ly in landmarks:
        ax.plot(lx, ly, marker='o', markersize=4, color='#ffffff', markeredgecolor='#000000')
        ax.text(lx + 1.2, ly - 0.5, name, color='#ffffff', fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#000000', alpha=0.55, edgecolor='none'))
    
    ax.plot([8, 8 + 33.33], [6, 6], color='white', linewidth=4)
    ax.text(8 + 16.66, 8, '1 km (1000 m)', color='white', fontsize=9, weight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6, edgecolor='none'))
    
    txt_info = (f"Czas: {t:4.2f} s  (krok {step})\n"
                f"T_max: {T_max_series[idx]:.0f} K ({T_max_series[idx]-273.15:.0f}°C)\n"
                f"Obszar plomieni: {active_fire_area_ha[idx]:.1f} ha\n"
                f"Obszar spalony: {burned_area_ha[idx]:.1f} ha")
    ax.text(0.03, 0.97, txt_info, transform=ax.transAxes, color='#ffffff', fontsize=9,
            va='top', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.75, edgecolor='#ff5722', linewidth=1.5))
    
    ax.set_title(f"Symulacja pożaru - Las Wolski (Kraków) | Fala ognia", fontsize=10.5, weight='bold', pad=7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Współrzędna X [30 m / jedn.] (Zachód $\\to$ Wschód)", fontsize=8.5)
    ax.set_ylabel("Współrzędna Y [30 m / jedn.] (Południe $\\to$ Północ)", fontsize=8.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames_gif1.append(Image.fromarray(rgba).convert('RGB'))

plt.close(fig)
save_animated_gif(frames_gif1, os.path.join(GIFS_DIR, '1_fire_spread_satellite.gif'))


# -------------------------------------------------------------
# GIF 2: Pole wektorowe wiatru i orografia
# -------------------------------------------------------------
print('\n[2/4] Renderowanie klatek GIF 2: Pole wektorowe wiatru...')
frames_gif2 = []
fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=95)

for idx, (step, t) in enumerate(zip(all_steps, all_times)):
    ax.clear()
    
    ax.imshow(h_dem.T, extent=[0, 100, 0, 100], origin='lower',
              cmap='terrain', alpha=0.82, vmin=190, vmax=380)
    
    ax.contour(np.linspace(0, 100, h_dem.shape[0]), np.linspace(0, 100, h_dem.shape[1]),
               h_dem.T, levels=np.arange(200, 375, 15), colors='#263238', linewidths=0.6, alpha=0.7)
    
    WE_t = W_E_BASE + GUST_AMP * np.cos(GUST_FREQ * t)
    WN_t = W_N_BASE + GUST_AMP * np.sin(GUST_FREQ * t)
    
    UX = WE_t * wEx_q + WN_t * wNx_q
    UY = WE_t * wEy_q + WN_t * wNy_q
    U_mag = np.hypot(UX, UY)
    
    ax.quiver(XQ, YQ, UX, UY, U_mag, cmap='plasma', scale=650, width=0.0045,
              headwidth=3.8, headlength=4.5, clim=[15, 55])
    
    T_curr = all_T[idx]
    if np.max(T_curr) > 700:
        ax.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T,
                   levels=[800], colors=['#d50000'], linewidths=[2.2])
    
    for name, lx, ly in landmarks:
        ax.plot(lx, ly, marker='^', markersize=4.5, color='#ffffff', markeredgecolor='#000000')
        ax.text(lx + 1.2, ly - 0.5, name, color='#111111', fontsize=7.5, weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#ffffff', alpha=0.75, edgecolor='none'))
    
    wind_angle_deg = np.degrees(np.arctan2(WN_t, WE_t))
    txt_wind = (f"Wiatr atmosferyczny:\n"
                f"Kierunek: {wind_angle_deg:.1f}° (SW $\\to$ NE)\n"
                f"Prędkość: {np.hypot(WE_t, WN_t):.1f} m/s\n"
                f"Porywy: $\\pm${GUST_AMP} m/s ($\\omega$={GUST_FREQ})")
    ax.text(0.03, 0.97, txt_wind, transform=ax.transAxes, color='#ffffff', fontsize=9,
            va='top', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d47a1', alpha=0.8, edgecolor='#00e5ff', linewidth=1.5))
    
    ax.set_title(f"Pole wiatru & ukształtowanie terenu | t = {t:4.2f} s", fontsize=10.5, weight='bold', pad=7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Współrzędna X [jedn. modelu]", fontsize=8.5)
    ax.set_ylabel("Współrzędna Y [jedn. modelu]", fontsize=8.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames_gif2.append(Image.fromarray(rgba).convert('RGB'))

plt.close(fig)
save_animated_gif(frames_gif2, os.path.join(GIFS_DIR, '2_wind_field_vectors.gif'))


# -------------------------------------------------------------
# GIF 3: Zuzycie biomasy paliwowej
# -------------------------------------------------------------
print('\n[3/4] Renderowanie klatek GIF 3: Zuzycie biomasy...')
frames_gif3 = []
fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=95)

for idx, (step, t) in enumerate(zip(all_steps, all_times)):
    ax.clear()
    
    F_curr = np.clip(all_F[idx], 0.0, 1.0)
    ax.imshow(F_curr.T, extent=[0, 100, 0, 100], origin='lower',
              cmap=fuel_cmap, vmin=0.0, vmax=1.0, interpolation='bicubic')
    
    T_curr = all_T[idx]
    if np.max(T_curr) > 700:
        ax.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T,
                   levels=[800], colors=['#ff3d00'], linewidths=[1.8])
    
    for name, lx, ly in landmarks:
        ax.plot(lx, ly, marker='s', markersize=4, color='#ffffff', markeredgecolor='#000000')
        ax.text(lx + 1.2, ly - 0.5, name, color='#ffffff', fontsize=7.5, weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#000000', alpha=0.6, edgecolor='none'))
    
    txt_fuel = (f"Stan biomasy leśnej:\n"
                f"Czas: {t:4.2f} s\n"
                f"Pozostałe paliwo: {np.mean(F_curr)/np.mean(F_0)*100:.1f}%\n"
                f"Powierzchnia zniszczona: {burned_area_ha[idx]:.1f} ha")
    ax.text(0.03, 0.97, txt_fuel, transform=ax.transAxes, color='#ffffff', fontsize=9,
            va='top', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1b5e20', alpha=0.85, edgecolor='#76ff03', linewidth=1.5))
    
    ax.set_title(f"Wypalanie biomasy leśnej (Paliwo $X_1$) | t = {t:4.2f} s", fontsize=10.5, weight='bold', pad=7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Współrzędna X [jedn. modelu]", fontsize=8.5)
    ax.set_ylabel("Współrzędna Y [jedn. modelu]", fontsize=8.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames_gif3.append(Image.fromarray(rgba).convert('RGB'))

plt.close(fig)
save_animated_gif(frames_gif3, os.path.join(GIFS_DIR, '3_fuel_consumption.gif'))


# -------------------------------------------------------------
# GIF 4: Dashboard 2x2
# -------------------------------------------------------------
print('\n[4/4] Renderowanie klatek GIF 4: Dashboard 2x2...')
frames_gif4 = []
fig, axs = plt.subplots(2, 2, figsize=(12, 11), dpi=85)

for idx, (step, t) in enumerate(zip(all_steps, all_times)):
    T_curr = all_T[idx]
    F_curr = np.clip(all_F[idx], 0.0, 1.0)
    
    # Panel (0, 0)
    ax = axs[0, 0]; ax.clear()
    ax.imshow(sat_arr, extent=[0, 100, 0, 100])
    ax.imshow(T_curr.T, extent=[0, 100, 0, 100], origin='lower',
              cmap=fire_cmap, norm=fire_norm, interpolation='bicubic')
    if np.max(T_curr) > 700:
        ax.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T,
                   levels=[800], colors=['#ffea00'], linewidths=[1.5])
    for name, lx, ly in landmarks[:3]:
        ax.plot(lx, ly, marker='o', markersize=3, color='#ffffff')
        ax.text(lx + 1.2, ly - 0.5, name, color='#ffffff', fontsize=7, weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#000000', alpha=0.55))
    ax.set_title(f"(a) Front ognia na ortofotomapie [K]", fontsize=10, weight='bold')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
    
    # Panel (0, 1)
    ax = axs[0, 1]; ax.clear()
    ax.imshow(h_dem.T, extent=[0, 100, 0, 100], origin='lower', cmap='terrain', alpha=0.8, vmin=190, vmax=380)
    ax.contour(np.linspace(0, 100, h_dem.shape[0]), np.linspace(0, 100, h_dem.shape[1]),
               h_dem.T, levels=np.arange(200, 375, 20), colors='#263238', linewidths=0.5, alpha=0.6)
    WE_t = W_E_BASE + GUST_AMP * np.cos(GUST_FREQ * t)
    WN_t = W_N_BASE + GUST_AMP * np.sin(GUST_FREQ * t)
    UX = WE_t * wEx_q + WN_t * wNx_q
    UY = WE_t * wEy_q + WN_t * wNy_q
    U_mag = np.hypot(UX, UY)
    ax.quiver(XQ, YQ, UX, UY, U_mag, cmap='plasma', scale=650, width=0.0045, headwidth=3.5, clim=[15, 55])
    if np.max(T_curr) > 700:
        ax.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T,
                   levels=[800], colors=['#d50000'], linewidths=[1.8])
    ax.set_title(f"(b) Pole wiatru i orografia ($|u| \\in [15, 55]$)", fontsize=10, weight='bold')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
    
    # Panel (1, 0)
    ax = axs[1, 0]; ax.clear()
    ax.imshow(F_curr.T, extent=[0, 100, 0, 100], origin='lower', cmap=fuel_cmap, vmin=0.0, vmax=1.0)
    if np.max(T_curr) > 700:
        ax.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T,
                   levels=[800], colors=['#ff3d00'], linewidths=[1.5])
    ax.set_title(f"(c) Rozkład i zużycie biomasy ($X_1$)", fontsize=10, weight='bold')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
    
    # Panel (1, 1)
    ax = axs[1, 1]; ax.clear()
    ax.plot(all_times[:idx+1], T_max_series[:idx+1], color='#d50000', linewidth=2.0, label='$T_{max}$ [K]')
    ax.set_ylabel("Temperatura [K]", color='#d50000', fontsize=9, weight='bold')
    ax.set_xlabel("Czas symulacji $t$ [s]", fontsize=9)
    ax.set_xlim(0, all_times[-1])
    ax.set_ylim(250, 1600)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax.twinx()
    ax2.plot(all_times[:idx+1], burned_area_ha[:idx+1], color='#2e7d32', linewidth=2.0, linestyle='--', label='Spalony [ha]')
    ax2.plot(all_times[:idx+1], active_fire_area_ha[:idx+1], color='#ff9100', linewidth=1.6, label='Front [ha]')
    ax2.set_ylabel("Powierzchnia [ha]", color='#2e7d32', fontsize=9, weight='bold')
    ax2.set_ylim(0, max(max(burned_area_ha) * 1.2, 10))
    
    ax.axvline(t, color='blue', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.set_title(f"(d) Dynamika czasowa parametrów (t = {t:4.2f} s)", fontsize=10, weight='bold')
    
    fig.suptitle(f"Zintegrowana symulacja IGA-ADS pożaru Lasu Wolskiego | t = {t:4.2f} s", fontsize=12, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames_gif4.append(Image.fromarray(rgba).convert('RGB'))

plt.close(fig)
save_animated_gif(frames_gif4, os.path.join(GIFS_DIR, '4_fire_dashboard_synchronized.gif'))

# -------------------------------------------------------------
# Ryciny statyczne dla sprawozdania LaTeX
# -------------------------------------------------------------
print('\n[5/5] Generowanie rycin statycznych dla LaTeX...')

# Rycina 1
fig, axs = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
axs[0].imshow(sat_arr, extent=[0, 100, 0, 100])
for name, lx, ly in landmarks:
    axs[0].plot(lx, ly, 'wo', markersize=3.5, markeredgecolor='k')
    axs[0].text(lx+1, ly-0.5, name, color='white', fontsize=7, weight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
axs[0].set_title("a) Ortofotomapa (Las Wolski)", fontsize=10, weight='bold')

im1 = axs[1].imshow(h_dem.T, extent=[0, 100, 0, 100], origin='lower', cmap='terrain')
axs[1].contour(np.linspace(0, 100, h_dem.shape[0]), np.linspace(0, 100, h_dem.shape[1]), h_dem.T, levels=12, colors='k', linewidths=0.4)
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='Wysokość [m n.p.m.]')
axs[1].set_title("b) Model terenu DEM (AWS)", fontsize=10, weight='bold')

im2 = axs[2].imshow(fuel_base.T, extent=[0, 100, 0, 100], origin='lower', cmap=fuel_cmap, vmin=0, vmax=1)
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label='Ułamek biomasy $X_1$')
axs[2].set_title("c) Klasyfikacja paliwa (OSM)", fontsize=10, weight='bold')

UX_0 = W_E_BASE * wEx_q + W_N_BASE * wNx_q
UY_0 = W_E_BASE * wEy_q + W_N_BASE * wNy_q
U_mag_0 = np.hypot(UX_0, UY_0)
im3 = axs[3].imshow(U_mag_0.T, extent=[0, 100, 0, 100], origin='lower', cmap='plasma')
axs[3].quiver(XQ, YQ, UX_0, UY_0, color='white', scale=700, width=0.0035, headwidth=3.5)
plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04, label='Prędkość lokalna $|u|$')
axs[3].set_title("d) Pole przepływu wiatru", fontsize=10, weight='bold')

for a in axs:
    a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_aspect('equal')
    a.set_xlabel("X [jedn. modelu]"); a.set_ylabel("Y [jedn. modelu]")

plt.tight_layout()
fig1_path = os.path.join(REPORT_DIR, 'fig_domain_overview.png')
plt.savefig(fig1_path, dpi=300)
plt.close(fig)
print(f'Zapisano: {fig1_path}')

# Rycina 2
fig, axs = plt.subplots(2, 4, figsize=(20, 9.5), dpi=300)
key_indices = [0, len(all_steps)//3, 2*len(all_steps)//3, len(all_steps)-1]

for col, idx in enumerate(key_indices):
    t_val = all_times[idx]
    T_curr = all_T[idx]
    F_curr = np.clip(all_F[idx], 0.0, 1.0)
    
    ax_t = axs[0, col]
    ax_t.imshow(sat_arr, extent=[0, 100, 0, 100])
    ax_t.imshow(T_curr.T, extent=[0, 100, 0, 100], origin='lower', cmap=fire_cmap, norm=fire_norm)
    if np.max(T_curr) > 700:
        ax_t.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T, levels=[800], colors=['#ffea00'], linewidths=[1.5])
    ax_t.set_title(f"Temperatura $T(x,y)$ (t = {t_val:4.2f} s)", fontsize=10, weight='bold')
    ax_t.set_xlim(0, 100); ax_t.set_ylim(0, 100); ax_t.set_aspect('equal')
    
    ax_f = axs[1, col]
    ax_f.imshow(F_curr.T, extent=[0, 100, 0, 100], origin='lower', cmap=fuel_cmap, vmin=0, vmax=1)
    if np.max(T_curr) > 700:
        ax_f.contour(np.linspace(0, 100, N_AXIS), np.linspace(0, 100, N_AXIS), T_curr.T, levels=[800], colors=['#ff3d00'], linewidths=[1.5])
    ax_f.set_title(f"Biomasa $X_1(x,y)$ (t = {t_val:4.2f} s)", fontsize=10, weight='bold')
    ax_f.set_xlim(0, 100); ax_f.set_ylim(0, 100); ax_f.set_aspect('equal')

for a in axs.flat:
    a.set_xlabel("X [jedn.]"); a.set_ylabel("Y [jedn.]")

plt.tight_layout()
fig2_path = os.path.join(REPORT_DIR, 'fig_fire_progression.png')
plt.savefig(fig2_path, dpi=300)
plt.close(fig)
print(f'Zapisano: {fig2_path}')

# Rycina 3
fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=300)
ax1.plot(all_times, T_max_series, color='#d50000', linewidth=2.4, label='Maks. temperatura $T_{max}$ [K]')
ax1.plot(all_times, T_mean_series, color='#ff6d00', linewidth=1.8, linestyle=':', label='Średnia temperatura $T_{mean}$ [K]')
ax1.set_xlabel("Czas symulacji $t$ [s]", fontsize=10.5, weight='bold')
ax1.set_ylabel("Temperatura [K]", color='#d50000', fontsize=10.5, weight='bold')
ax1.set_ylim(250, 1600)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(all_times, burned_area_ha, color='#1b5e20', linewidth=2.4, label='Obszar spalony [ha]')
ax2.plot(all_times, active_fire_area_ha, color='#ffd600', linewidth=2.0, linestyle='--', label='Aktywny front płomieni [ha]')
ax2.set_ylabel("Powierzchnia [ha]", color='#1b5e20', fontsize=10.5, weight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

plt.title("Dynamika rozwoju pożaru i zużycia biomasy w czasie", fontsize=11.5, weight='bold', pad=10)
plt.tight_layout()
fig3_path = os.path.join(REPORT_DIR, 'fig_burned_area_metrics.png')
plt.savefig(fig3_path, dpi=300)
plt.close(fig)
print(f'Zapisano: {fig3_path}')

print('\nZakonczono pomyslnie rendering GIFow i rycin!')
