# Evaluation suite audit

These are legacy diagnostic results, not official target-comparable metrics. The files use conditioning views, nonstandard SSIM/LPIPS proxies, omit CD/F-score, and predate the TRELLIS export-frame correction.

| Method | N | PSNR mean | SSIM mean | LPIPS mean | Geometry N |
|---|---:|---:|---:|---:|---:|
| original_trellis | 8 | 15.260334 | 0.193318 | 0.235704 | 0 |
| stage2_geoss_ss | 8 | 15.261365 | 0.194543 | 0.235295 | 0 |
| stage3_geovis_slat | 8 | 15.206283 | 0.191915 | 0.234870 | 0 |
| stage4_geovis_slat_joint | 6 | 14.566947 | 0.217870 | 0.237434 | 0 |

Integrity findings: 30/30 records are non-official.
