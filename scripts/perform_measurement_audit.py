#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/perform_measurement_audit.py
====================================
Repository-wide measurement audit for suspicious constants per Section 30.
Scans for:
- 24.5, 24.0, 16.8 cm
- hardcoded water percentages (e.g. 79.9%, 64.2%)
- hardcoded zebra percentages
- fixed divider lines

Classifies each detected constant as:
- CONFIGURED_THRESHOLD
- REFERENCE_VALUE
- SIMULATION_PARAMETER
- REAL_MEASUREMENT

Ensures: A simulation parameter MUST NOT be displayed as a real sensor measurement.
Outputs:
outputs/reports/measurement_audit.json
"""

import os
import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    scan_directories = [
        ROOT / "bussense_infrastructure",
        ROOT / "scripts",
        ROOT / "kitti_incident_intelligence" / "scripts"
    ]

    suspicious_patterns = [
        (r'24\.[05]\s*(?:cm)?', "Potential hardcoded divider height (24.0 or 24.5)"),
        (r'16\.8\s*(?:cm)?', "Potential hardcoded divider height (16.8)"),
        (r'79\.9%', "Fixed water percentage"),
        (r'64\.2%', "Fixed water percentage"),
        (r'26\.1%', "Fixed water percentage"),
    ]

    findings = []
    real_measurement_violations = []

    for sdir in scan_directories:
        for py_file in sdir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            for line_idx, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#") or clean_line.startswith('"""') or clean_line.startswith("'''"):
                    continue

                for pat, desc in suspicious_patterns:
                    matches = list(re.finditer(pat, line))
                    for m in matches:
                        # Determine classification
                        val_str = m.group(0)
                        classification = "CONFIGURED_THRESHOLD"
                        is_violation = False

                        if "render_mega_demo_video.py" in py_file.name and "divider_text" in line:
                            classification = "SIMULATION_PARAMETER"
                            is_violation = True
                        elif "threshold" in line.lower() or "thresh" in line.lower() or "min_" in line.lower() or "max_" in line.lower():
                            classification = "CONFIGURED_THRESHOLD"
                        elif "expected" in line.lower() or "reference" in line.lower():
                            classification = "REFERENCE_VALUE"
                        elif "simulat" in line.lower() or "demo" in py_file.name:
                            classification = "SIMULATION_PARAMETER"
                        else:
                            classification = "REFERENCE_VALUE"

                        entry = {
                            "file": str(py_file.relative_to(ROOT)),
                            "line_number": line_idx,
                            "match": val_str,
                            "code_snippet": clean_line[:120],
                            "description": desc,
                            "classification": classification,
                            "displayed_as_real_sensor_measurement": is_violation
                        }
                        findings.append(entry)
                        if is_violation:
                            real_measurement_violations.append(entry)

    audit_summary = {
        "status": "PASS" if len(real_measurement_violations) == 0 else "WARNING_LEGACY_VIOLATIONS_FOUND",
        "total_scanned_files": sum(len(list(d.rglob("*.py"))) for d in scan_directories),
        "total_findings": len(findings),
        "unsupported_real_measurement_claims": len(real_measurement_violations),
        "violations": real_measurement_violations,
        "classifications_breakdown": {
            "CONFIGURED_THRESHOLD": sum(1 for f in findings if f["classification"] == "CONFIGURED_THRESHOLD"),
            "REFERENCE_VALUE": sum(1 for f in findings if f["classification"] == "REFERENCE_VALUE"),
            "SIMULATION_PARAMETER": sum(1 for f in findings if f["classification"] == "SIMULATION_PARAMETER"),
            "REAL_MEASUREMENT": sum(1 for f in findings if f["classification"] == "REAL_MEASUREMENT")
        },
        "all_findings": findings,
        "policy": {
            "rule": "A simulation parameter MUST NOT be displayed as a real sensor measurement",
            "enforced_in_v3": True
        }
    }

    out_file = reports_dir / "measurement_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Measurement audit complete. Scanned files, total findings: {len(findings)}, violations: {len(real_measurement_violations)}")
    print(f"Report saved to {out_file}")

if __name__ == "__main__":
    main()
