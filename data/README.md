# data/

Input tables are **not** version-controlled (see `.gitignore`).

Generate the synthetic examples:

    python make_demo_data.py --layout subbasins --out data/subbasin_data.xlsx --years 12
    python make_future_scenarios.py --out data/future_scenarios.xlsx \
        --scenarios rcp26 rcp45 rcp85 ssp370 --subbasins 3 --start 2026 --end 2060

For a real basin, put your own Excel file here (or anywhere) and set `data.path`
in the config. The required columns are described in README section 3.
