# Publishing this repository

## 1. Edit three placeholders first

| File | Where | Replace |
|---|---|---|
| `LICENSE` | line 3 | `<YEAR> <YOUR NAME>` |
| `CITATION.cff` | authors block | `<YOUR GIVEN NAME>`, `<YOUR FAMILY NAME>`; optionally uncomment `orcid` / `affiliation` |
| `CITATION.cff` | `date-released`, `repository-code` | the release date and your repository URL |

Nothing else needs changing. Check none are left:

    findstr /S "<YOUR <YEAR> <YYYY" *.md *.cff LICENSE      (Windows)
    grep -rn "<YOUR\|<YEAR\|<YYYY" .                        (macOS / Linux)

## 2. First push

From this folder, with your empty GitHub repository already created:

    git init
    git branch -M main
    git config user.name  "Your Name"
    git config user.email "you@example.org"
    git add -A
    git commit -m "hydroml v1.3: configuration-driven ML hydrological modelling template"
    git remote add origin https://github.com/<YOUR USER>/<YOUR REPO>.git
    git push -u origin main

## 3. What is and is not tracked

`.gitignore` keeps the repository small and reproducible from source:

* **tracked** - all code, the three configurations, `requirements.txt`, the
  driver notebook, `README.md`, `LICENSE`, `CITATION.cff` (~0.25 MB total);
* **not tracked** - `results/`, `projections/` and `data/*.xlsx`. These are
  generated: `make_demo_data.py` and `make_future_scenarios.py` recreate the
  example inputs, and a pipeline run recreates the outputs. Keeping them out
  avoids a multi-megabyte repository whose committed files drift out of step
  with the code that produced them.

If you do want an example run visible on GitHub, commit one figure and its
`metrics_wide.csv` into a `docs/` folder and reference them from the README,
rather than un-ignoring the whole `results/` tree.

Before pushing, confirm no real basin data is staged:

    git status --short
    git ls-files | findstr /I ".xlsx .xls .csv"

## 4. After the first release

* GitHub reads `CITATION.cff` and shows a "Cite this repository" button.
* For a citable DOI, link the repository at zenodo.org (Settings - GitHub) and
  publish a release; Zenodo archives that tag and mints a version DOI.
* Add the DOI badge to the top of `README.md` once it is minted.

## 5. Authorship

You are the sole author and contributor. The refactor was done with AI
assistance, which `README.md` section 16 acknowledges; under ICMJE and COPE
guidance an AI system cannot be an author or contributor, because it can
neither take responsibility for the work nor consent to its publication. Do
not add it to `CITATION.cff` or the contributor list.
