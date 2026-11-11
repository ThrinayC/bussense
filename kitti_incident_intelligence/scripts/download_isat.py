import gdown
import os

os.makedirs("data/evttc", exist_ok=True)
# isat.yaml ID: 14xFu70Mt89OuqRaKwiCp_8NsnPSHOFcr
gdown.download(id="14xFu70Mt89OuqRaKwiCp_8NsnPSHOFcr", output="data/evttc/isat.yaml", quiet=False)

if os.path.exists("data/evttc/isat.yaml"):
    with open("data/evttc/isat.yaml", "r", encoding="utf-8") as f:
        print(f.read()[:500])
