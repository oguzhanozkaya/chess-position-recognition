"""Project paths and shared configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAX_TEXT_TOKENS = 256

@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by pipeline stages."""

    root: Path
    data: Path
    raw_data: Path
    interim_data: Path
    processed_data: Path
    output: Path
    figures: Path
    models: Path
    predictions: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        data = root / "data"
        output = root / "output"
        return cls(
            root=root,
            data=data,
            raw_data=data / "raw",
            interim_data=data / "interim",
            processed_data=data / "processed",
            output=output,
            figures=output / "figures",
            models=output / "models",
            predictions=output / "predictions",
            reports=output / "reports",
        )

    def generated_directories(self) -> tuple[Path, ...]:
        return (
            self.raw_data,
            self.interim_data,
            self.processed_data,
            self.figures,
            self.models,
            self.predictions,
            self.reports,
        )


def build_paths(root: Path | None = None) -> ProjectPaths:
    """Build project paths for the repository or a test root."""

    return ProjectPaths.from_root(PROJECT_ROOT if root is None else root)


DEFAULT_PATHS = build_paths()


def ensure_generated_directories(paths: ProjectPaths = DEFAULT_PATHS) -> tuple[Path, ...]:
    """Create generated data and output directories if they are missing."""

    directories = paths.generated_directories()
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


@dataclass(frozen=True)
class SourceDefinition:
    """A source that can be downloaded into `data/raw/`."""

    source_id: str
    title: str
    category: str
    source_type: str
    url: str
    raw_path: Path
    notes: str


CBRT_CONSUMER_PRICES = SourceDefinition(
    source_id="cbrt_consumer_prices",
    title="CBRT Consumer Prices",
    category="numeric",
    source_type="official_html",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Statistics/Inflation+Data/Consumer+Prices",
    raw_path=Path("numeric/cbrt_consumer_prices.html"),
    notes="Official CBRT page listing TURKSTAT CPI year-to-year and month-to-month rates.",
)

CBRT_FX_MONTH_END = SourceDefinition(
    source_id="cbrt_fx_month_end",
    title="CBRT Indicative Exchange Rates Month End Archive",
    category="numeric",
    source_type="official_xml_month_end_archive",
    url="https://www.tcmb.gov.tr/kurlar/",
    raw_path=Path("numeric/cbrt_fx_month_end"),
    notes="Official CBRT public exchange-rate XML archive sampled at each month end with business-day fallback.",
)

FRED_BRENT_OIL = SourceDefinition(
    source_id="fred_brent_oil",
    title="FRED Brent Crude Oil Price",
    category="numeric",
    source_type="fred_csv",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
    raw_path=Path("numeric/fred/dcoilbrenteu.csv"),
    notes="Public FRED CSV for Europe Brent crude oil spot price in USD per barrel.",
)

FRED_TURKEY_INDUSTRIAL_PRODUCTION = SourceDefinition(
    source_id="fred_turkey_industrial_production",
    title="FRED Turkey Industrial Production Growth",
    category="numeric",
    source_type="fred_csv",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=TURPRINTO01GYSAM",
    raw_path=Path("numeric/fred/turprinto01gysam.csv"),
    notes="Public FRED/OECD monthly industrial production year-over-year growth for Turkiye.",
)

FRED_TURKEY_UNEMPLOYMENT_RATE = SourceDefinition(
    source_id="fred_turkey_unemployment_rate",
    title="FRED Turkey Monthly Unemployment Rate",
    category="numeric",
    source_type="fred_csv",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=LRHUTTTTTRM156S",
    raw_path=Path("numeric/fred/lrhutttttrm156s.csv"),
    notes="Public FRED/OECD monthly seasonally adjusted unemployment rate for Turkiye.",
)

CBRT_MPC_DECISIONS = SourceDefinition(
    source_id="cbrt_mpc_decisions",
    title="CBRT MPC Meeting Decisions",
    category="text",
    source_type="official_html_listing",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/MPC/MPC%2BMeeting%2BDecisions",
    raw_path=Path("text/cbrt_mpc_decisions.html"),
    notes="Official listing page for interest-rate press releases and MPC decisions.",
)

CBRT_MPC_SUMMARIES = SourceDefinition(
    source_id="cbrt_mpc_summaries",
    title="CBRT MPC Meeting Summaries",
    category="text",
    source_type="official_html_listing",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/MPC/MPC%2BMeeting%2BSummaries",
    raw_path=Path("text/cbrt_mpc_summaries.html"),
    notes="Official listing page for MPC meeting summaries.",
)

SOURCE_REGISTRY: tuple[SourceDefinition, ...] = (
    CBRT_CONSUMER_PRICES,
    CBRT_FX_MONTH_END,
    FRED_BRENT_OIL,
    FRED_TURKEY_INDUSTRIAL_PRODUCTION,
    FRED_TURKEY_UNEMPLOYMENT_RATE,
    CBRT_MPC_DECISIONS,
    CBRT_MPC_SUMMARIES,
)


FRED_SERIES = {
    FRED_BRENT_OIL.source_id: ("DCOILBRENTEU", "brent_oil_usd", "daily"),
    FRED_TURKEY_INDUSTRIAL_PRODUCTION.source_id: (
        "TURPRINTO01GYSAM",
        "turkey_industrial_production_yoy_sa",
        "monthly",
    ),
    FRED_TURKEY_UNEMPLOYMENT_RATE.source_id: ("LRHUTTTTTRM156S", "turkey_unemployment_rate_sa", "monthly"),
}
