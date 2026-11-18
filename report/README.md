# Sprawozdanie – Symulacja pożaru lasu (Las Wolski, Kraków)

## Instrukcja uruchomienia w Overleaf
Aby skompilować sprawozdanie w [Overleaf](https://www.overleaf.com):
1. Utwórz nowy projekt (Blank Project).
2. Przeciągnij i upuść (Drag & Drop) całą zawartość tego folderu `report/` do projektu na Overleafie:
   - `main.tex` (główny plik sprawozdania)
   - `fig_domain_overview.png` (Rysunek 1 – mapa domeny)
   - `fig_fire_progression.png` (Rysunek 2 – ewolucja pożaru)
   - `fig_burned_area_metrics.png` (Rysunek 3 – wykresy dynamiki czasowej)
   - `1_fire_spread_satellite.gif` (GIF 1)
   - `2_wind_field_vectors.gif` (GIF 2)
   - `3_fuel_consumption.gif` (GIF 3)
   - `4_fire_dashboard_synchronized.gif` (GIF 4)
3. Upewnij się, że kompilator to **pdfLaTeX** (domyślny) lub **LuaLaTeX / XeLaTeX** i kliknij **Recompile**.

## Zawartość folderu
- `main.tex` / `sprawozdanie.tex` – Kompletny kod źródłowy sprawozdania w LaTeX ze wszystkimi sekcjami, równaniami, opisem metody IGA-ADS, rycinami oraz listingami kodu źródłowego.
- `1_fire_spread_satellite.gif` – Zsynchronizowana animacja fali ognia i temperatury na rzeczywistej ortofotomapie satelitarnej Lasu Wolskiego (Esri).
- `2_wind_field_vectors.gif` – Zsynchronizowana animacja pola wektorowego wiatru (gęste strzałki) z ukształtowaniem terenu (DEM) i obrysem czoła ognia.
- `3_fuel_consumption.gif` – Zsynchronizowana animacja ubytku biomasy leśnej.
- `4_fire_dashboard_synchronized.gif` – Zsynchronizowany pulpit nawigacyjny (2x2) łączący wszystkie widoki i dynamiczne wykresy metryk.
