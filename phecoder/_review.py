"""Interactive ICD review and OHDSI ATLAS concept-set export."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    from importlib.metadata import version as _pkg_version

    _PHECODER_VERSION = _pkg_version("phecoder")
except Exception:
    _PHECODER_VERSION = "unknown"


def _require_ipywidgets():
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError as e:
        raise ImportError(
            "Interactive review requires ipywidgets. "
            "Install with: pip install 'phecoder[review]'"
        ) from e
    return w, display


def _safe_filename(s: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return cleaned or "phecode"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewSession:
    """
    Interactive selection of top-K ICD codes per phecode in a Jupyter notebook.

    Typical use:

        session = pc.review(top_k=30, score_threshold=0.5)
        session                       # renders the widget in a notebook cell
        session.save("picks.parquet") # or .csv / .json
    """

    def __init__(
        self,
        results: pd.DataFrame,
        top_k: int = 50,
        score_threshold: Optional[float] = None,
    ):
        if results is None or results.empty:
            raise ValueError(
                "No results to review. The results DataFrame is empty — "
                "check the `models`, `phecodes`, and `include_ensembles` "
                "arguments you passed to pc.review() (or call pc.load_results() "
                "directly to see what is available)."
            )
        required = {"phecode", "phecode_string", "icd_code", "icd_string", "score", "rank"}
        missing = required - set(results.columns)
        if missing:
            raise ValueError(f"results missing required columns: {missing}")

        self._results = _collapse_by_icd(results, top_k=top_k)
        self._score_threshold = score_threshold
        self._checkboxes: dict[int, object] = {}
        self._widget = None

    _ROW_CSS = """
    <style>
      .phecoder-row { padding: 2px 4px; align-items: center; }
      .phecoder-row-odd  { background-color: #f5f5f5; }
      .phecoder-row-even { background-color: #ffffff; }
      .phecoder-header   { background-color: #e8e8e8; font-weight: 600; padding: 4px; }
      .phecoder-cell     { font-family: -apple-system, system-ui, sans-serif; font-size: 12px; padding: 0 6px; }
      .phecoder-cell-mono{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; padding: 0 6px; }
    </style>
    """

    _COL_WIDTHS = {
        "check": "34px",
        "icd_code": "110px",
        "icd_string": "auto",  # flex
        "score": "80px",
        "rank": "60px",
        "model": "240px",
    }

    @classmethod
    def _cell(cls, html_text: str, key: str, mono: bool = False, align: str = "left"):
        import ipywidgets as w  # local
        cls_name = "phecoder-cell-mono" if mono else "phecoder-cell"
        styled = f"<div style='text-align:{align};'>{html_text}</div>"
        widget = w.HTML(value=styled, layout=w.Layout(width=cls._COL_WIDTHS[key], flex="1 1 auto" if cls._COL_WIDTHS[key] == "auto" else "0 0 auto"))
        widget.add_class(cls_name)
        return widget

    def _build_widget(self):
        w, _ = _require_ipywidgets()

        shared_threshold = w.FloatText(
            value=float(self._score_threshold) if self._score_threshold is not None else 0.5,
            description="Score threshold",
            step=0.01,
            layout=w.Layout(width="280px"),
            style={"description_width": "110px"},
        )

        tab_children = []
        titles = []
        for phecode, sub in self._results.groupby("phecode", sort=False):
            phe_str = sub["phecode_string"].iloc[0]
            header = w.HTML(
                f"<b>{phecode}</b> — {phe_str} "
                f"<span style='color:gray'>({len(sub)} unique ICDs)</span>"
            )

            col_header = w.HBox(
                [
                    w.HTML("", layout=w.Layout(width=self._COL_WIDTHS["check"], flex="0 0 auto")),
                    self._cell("ICD code", "icd_code"),
                    self._cell("Description", "icd_string"),
                    self._cell("Score", "score", align="right"),
                    self._cell("Rank", "rank", align="right"),
                    self._cell("Model(s)", "model"),
                ]
            )
            col_header.add_class("phecoder-header")

            row_checkboxes: list[object] = []
            row_boxes: list[object] = []
            for i_row, (idx, row) in enumerate(sub.iterrows()):
                pre_checked = (
                    self._score_threshold is not None
                    and float(row["score"]) >= self._score_threshold
                )
                models_list = list(row["models"]) if "models" in row.index else []
                best_model = row.get("model", "") if "model" in row.index else ""
                if len(models_list) > 1:
                    others = [m for m in models_list if m != best_model]
                    model_html = (
                        f"<b>{best_model}</b> "
                        f"<span style='color:gray'>+{len(others)} other"
                        f"{'s' if len(others) > 1 else ''}</span>"
                    )
                elif best_model:
                    model_html = f"<b>{best_model}</b>"
                else:
                    model_html = ""

                cb = w.Checkbox(
                    value=bool(pre_checked),
                    indent=False,
                    layout=w.Layout(width=self._COL_WIDTHS["check"], flex="0 0 auto"),
                )
                self._checkboxes[idx] = cb
                row_checkboxes.append(cb)

                box = w.HBox(
                    [
                        cb,
                        self._cell(str(row["icd_code"]), "icd_code", mono=True),
                        self._cell(str(row["icd_string"]), "icd_string"),
                        self._cell(f"{float(row['score']):.3f}", "score", mono=True, align="right"),
                        self._cell(str(int(row["rank"])), "rank", mono=True, align="right"),
                        self._cell(model_html, "model"),
                    ]
                )
                box.add_class("phecoder-row")
                box.add_class("phecoder-row-odd" if i_row % 2 else "phecoder-row-even")
                row_boxes.append(box)

            select_all_btn = w.Button(description="Select all", layout=w.Layout(width="auto"))
            select_none_btn = w.Button(description="Select none", layout=w.Layout(width="auto"))
            apply_thresh_btn = w.Button(
                description="Apply score (this tab only)", layout=w.Layout(width="auto")
            )

            def _mk_set_all(cbs, value):
                def _handler(_):
                    for cb in cbs:
                        cb.value = value
                return _handler

            def _mk_apply_threshold(cbs, sub_local, thresh_widget):
                def _handler(_):
                    cutoff = float(thresh_widget.value)
                    scores = sub_local["score"].astype(float).tolist()
                    for cb, s in zip(cbs, scores):
                        cb.value = s >= cutoff
                return _handler

            select_all_btn.on_click(_mk_set_all(row_checkboxes, True))
            select_none_btn.on_click(_mk_set_all(row_checkboxes, False))
            apply_thresh_btn.on_click(
                _mk_apply_threshold(row_checkboxes, sub.reset_index(drop=True), shared_threshold)
            )

            actions = w.HBox([select_all_btn, select_none_btn, apply_thresh_btn])
            tab_children.append(w.VBox([header, actions, col_header] + row_boxes))
            titles.append(str(phecode))

        tab = w.Tab(children=tab_children)
        for i, t in enumerate(titles):
            tab.set_title(i, t)

        path_input = w.Text(
            value="selection.parquet",
            description="Save to:",
            layout=w.Layout(width="60%"),
        )
        save_button = w.Button(description="Save selection", button_style="primary")
        status = w.HTML("")

        def _on_save(_):
            try:
                out = self.save(path_input.value)
                status.value = (
                    f"<span style='color:green'>Saved "
                    f"{len(self.selection)} rows to {out}</span>"
                )
            except Exception as exc:
                status.value = f"<span style='color:red'>Error: {exc}</span>"

        save_button.on_click(_on_save)

        apply_all_btn = w.Button(
            description="Apply score to all", layout=w.Layout(width="auto")
        )

        def _apply_all(_):
            cutoff = float(shared_threshold.value)
            scores = self._results["score"].astype(float).tolist()
            for (idx, cb), s in zip(self._checkboxes.items(), scores):
                cb.value = s >= cutoff

        apply_all_btn.on_click(_apply_all)

        top_controls = w.HBox([shared_threshold, apply_all_btn])
        save_controls = w.HBox([path_input, save_button])
        css = w.HTML(self._ROW_CSS)
        self._widget = w.VBox([css, top_controls, tab, save_controls, status])
        return self._widget

    def display(self):
        _, display = _require_ipywidgets()
        if self._widget is None:
            self._build_widget()
        display(self._widget)

    def _ipython_display_(self):
        self.display()

    @property
    def selection(self) -> pd.DataFrame:
        """Currently-selected rows as a DataFrame with provenance columns."""
        if not self._checkboxes:
            if self._score_threshold is None:
                sel = self._results.iloc[0:0].copy()
            else:
                mask = self._results["score"].astype(float) >= self._score_threshold
                sel = self._results[mask].copy()
        else:
            keep = [idx for idx, cb in self._checkboxes.items() if cb.value]
            sel = self._results.loc[keep].copy()
        sel["selected_at"] = _now_iso()
        sel["phecoder_version"] = _PHECODER_VERSION
        return sel.reset_index(drop=True)

    def save(self, path: Union[str, Path]) -> Path:
        """Save current selection. Format inferred from suffix (.parquet/.csv/.json)."""
        path = Path(path)
        sel = self.selection
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            sel.to_parquet(path, index=False)
        elif suffix == ".csv":
            sel.to_csv(path, index=False)
        elif suffix == ".json":
            path.write_text(json.dumps(_to_concept_set_json(sel), indent=2))
        else:
            raise ValueError(
                f"Unsupported extension {path.suffix!r}. Use .parquet, .csv, or .json"
            )
        return path


def _collapse_by_icd(results: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Collapse rows by (phecode, icd_code) keeping the best-scoring row, attach
    a `models` list of all models that retrieved this ICD (sorted by score
    desc), then keep the top_k unique ICDs per phecode by best rank.
    """
    ordered = results.sort_values(["phecode", "icd_code", "score"], ascending=[True, True, False])
    best = ordered.drop_duplicates(subset=["phecode", "icd_code"], keep="first").copy()

    if "model" in results.columns:
        models_per_icd = (
            ordered.groupby(["phecode", "icd_code"])["model"]
            .apply(lambda s: [m for m in s.tolist() if pd.notna(m)])
            .reset_index()
            .rename(columns={"model": "models"})
        )
        best = best.merge(models_per_icd, on=["phecode", "icd_code"], how="left")
    else:
        best["models"] = [[] for _ in range(len(best))]

    best = best.sort_values(["phecode", "rank"], kind="stable")
    best = best.groupby("phecode", group_keys=False).head(top_k)
    return best.reset_index(drop=True)


def _to_concept_set_json(sel: pd.DataFrame) -> dict:
    """Group a flat selection DataFrame into a nested concept-set structure."""
    phecodes = []
    for phecode, sub in sel.groupby("phecode", sort=False):
        codes = []
        for _, row in sub.iterrows():
            entry = {
                "icd_code": row["icd_code"],
                "icd_string": row["icd_string"],
                "score": float(row["score"]),
                "rank": int(row["rank"]),
            }
            for opt in ("model", "concept_id", "vocabulary_id"):
                if opt in row.index and pd.notna(row[opt]):
                    val = row[opt]
                    if opt == "concept_id":
                        val = int(val)
                    entry[opt] = val
            if "models" in row.index and isinstance(row["models"], (list, tuple)):
                entry["models"] = list(row["models"])
            codes.append(entry)
        phecodes.append(
            {
                "phecode": str(phecode),
                "phecode_string": str(sub["phecode_string"].iloc[0]),
                "codes": codes,
            }
        )
    return {
        "created_at": _now_iso(),
        "phecoder_version": _PHECODER_VERSION,
        "phecodes": phecodes,
    }


def _load_selection(source) -> pd.DataFrame:
    if isinstance(source, ReviewSession):
        return source.selection
    if isinstance(source, pd.DataFrame):
        return source.copy()
    p = Path(source)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".json":
        data = json.loads(p.read_text())
        rows = []
        for ph in data.get("phecodes", []):
            for c in ph.get("codes", []):
                rows.append(
                    {
                        "phecode": ph["phecode"],
                        "phecode_string": ph.get("phecode_string", ""),
                        **c,
                    }
                )
        return pd.DataFrame(rows)
    raise ValueError(f"Cannot load selection from {source!r}")


_ATLAS_REQUIRED_COL = "concept_id"


def export_atlas(
    selection,
    out_path: Union[str, Path],
    include_descendants: bool = True,
    include_mapped: bool = True,
    concept_set_name_template: str = "{phecode_string}",
) -> list[Path]:
    """
    Export a selection as OHDSI ATLAS concept-set JSON.

    Requires a 'concept_id' column (OMOP concept_id). Optional companion
    columns: vocabulary_id, icd_code, icd_string.

    If out_path ends in '.json' a single bundle file is written (a JSON
    list of concept sets). Otherwise out_path is treated as a directory
    and one '<phecode>.json' file is written per phecode.
    """
    sel = _load_selection(selection)
    if _ATLAS_REQUIRED_COL not in sel.columns:
        raise ValueError(
            "ATLAS export requires a 'concept_id' column. Add OMOP concept_id "
            "values to your icd_df before running Phecoder (any extra columns "
            "on icd_df are preserved end-to-end). See README section "
            "'Interactive review & ATLAS export'."
        )

    sel = sel.dropna(subset=[_ATLAS_REQUIRED_COL]).copy()
    if sel.empty:
        raise ValueError("No rows with a non-null concept_id to export.")

    out_path = Path(out_path)
    bundle_mode = out_path.suffix.lower() == ".json"
    if bundle_mode:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    bundle: list[dict] = []
    written: list[Path] = []
    for phecode, sub in sel.groupby("phecode", sort=False):
        phe_str = (
            str(sub["phecode_string"].iloc[0])
            if "phecode_string" in sub.columns
            else str(phecode)
        )
        items = []
        for _, row in sub.iterrows():
            concept = {"CONCEPT_ID": int(row["concept_id"])}
            if "icd_code" in row.index and pd.notna(row["icd_code"]):
                concept["CONCEPT_CODE"] = str(row["icd_code"])
            if "icd_string" in row.index and pd.notna(row["icd_string"]):
                concept["CONCEPT_NAME"] = str(row["icd_string"])
            if "vocabulary_id" in row.index and pd.notna(row["vocabulary_id"]):
                concept["VOCABULARY_ID"] = str(row["vocabulary_id"])
            items.append(
                {
                    "concept": concept,
                    "isExcluded": False,
                    "includeDescendants": bool(include_descendants),
                    "includeMapped": bool(include_mapped),
                }
            )
        concept_set = {
            "name": concept_set_name_template.format(
                phecode=phecode, phecode_string=phe_str
            ),
            "expression": {"items": items},
        }
        if bundle_mode:
            bundle.append(concept_set)
        else:
            fp = out_path / f"{_safe_filename(phecode)}.json"
            fp.write_text(json.dumps(concept_set, indent=2))
            written.append(fp)

    if bundle_mode:
        out_path.write_text(json.dumps(bundle, indent=2))
        written.append(out_path)
    return written
