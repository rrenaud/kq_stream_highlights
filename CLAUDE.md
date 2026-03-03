# Project Conventions

## Build
- `npm run build` — type-checks with `tsc` then builds with Vite
- `npx vite` — dev server with HMR
- `npx vite preview` — serves the production build from `dist/`

## Data
- Vite dev server serves static files from `public/`. Data files in `chapters/` at the project root are NOT served by Vite — they must be copied to `public/chapters/` to be visible in dev mode.
- `run_kquity.py` auto-copies output to `public/chapters/` after writing, but if you manually create or update data files in `chapters/`, remember to copy them to `public/chapters/` too.

## UI Conventions
- All percentage values in grid/table cells must be centered (text-align: center, vertical-align: middle)
- Egg and berry counterfactual grids should match in total width (egg cells are wider to compensate for fewer columns)
