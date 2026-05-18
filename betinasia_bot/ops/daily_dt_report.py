from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from loguru import logger

from .daily_full_report import DailyReportCfg, _load_env_file, run_daily_full


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Relatório diário DT (espelha daily_full_report com defaults de DT/down)."
    )
    ap.add_argument("--out-dir", default=os.getenv("DAILY_DT_REPORT_OUT_DIR", "logs/daily_reports_dt"))
    ap.add_argument("--hypothesis-type", default=os.getenv("DAILY_DT_OOS_HYPOTHESIS_TYPE", "DT"))
    ap.add_argument("--direction", default=os.getenv("DAILY_DT_OOS_DIRECTION", "down"))
    ap.add_argument("--versions", default=os.getenv("DAILY_DT_OOS_VERSIONS", "v5.3-ws-gate-back"))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    # Defaults operacionais do daily DT sem afetar o daily H3B.
    os.environ["DAILY_OOS_HYPOTHESIS_TYPE"] = str(args.hypothesis_type)
    os.environ["DAILY_OOS_DIRECTION"] = str(args.direction)
    os.environ["DAILY_OOS_VERSIONS"] = str(args.versions)
    os.environ["DAILY_REPORT_OUT_DIR"] = str(args.out_dir)
    # Por padrão, o daily DT não publica policy_current global.
    os.environ["DAILY_WF_PUBLISH_CURRENT"] = str(os.getenv("DAILY_DT_WF_PUBLISH_CURRENT", "0"))

    # Permite canal de Telegram separado para o report DT.
    if "DAILY_DT_REPORT_TELEGRAM" in os.environ:
        os.environ["DAILY_REPORT_TELEGRAM"] = str(os.getenv("DAILY_DT_REPORT_TELEGRAM", "1"))

    cfg = DailyReportCfg(out_dir=Path(str(args.out_dir)))
    out = asyncio.run(run_daily_full(cfg))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
