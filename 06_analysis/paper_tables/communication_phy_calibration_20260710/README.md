# Communication PHY calibration tables

Generated with:

```powershell
python 05_simulation/run_communication_phy_calibration.py --samples 20000 --seed 20260710 --protocol-validation-episodes 5 --protocol-validation-slots 300
```

See `06_analysis/communication_phy_sensitivity_calibration_20260710.md` for assumptions, interpretation, and limitations. The protocol validation files are an auditable negative check: five 300-slot episodes produced only three aligned handshake attempts per profile and are not statistically discriminating.
