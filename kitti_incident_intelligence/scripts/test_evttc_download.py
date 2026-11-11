import gdown
import os

url = "https://drive.google.com/file/d/1l5_doDb30GsscNAtJquthr96RYZUXGBy/view?usp=sharing"
output = "data/evttc_test_gt_ttc.txt"
os.makedirs("data", exist_ok=True)

try:
    res = gdown.download(id="1l5_doDb30GsscNAtJquthr96RYZUXGBy", output=output, quiet=False)
    print("Download result:", res)
    if os.path.exists(output):
        print("File size:", os.path.getsize(output))
        with open(output, "r", encoding="utf-8", errors="ignore") as f:
            print("Content preview:")
            for i in range(10):
                line = f.readline()
                if not line:
                    break
                print(" ", line.strip())
except Exception as e:
    print("Download failed:", e)
