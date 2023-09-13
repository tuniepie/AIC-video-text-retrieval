import os
from pathlib import Path
FILE = Path(__file__).resolve()
# print(FILE)
ROOT = FILE.parents[1]
# print(ROOT)
WORK_DIR = os.path.dirname(ROOT)
# print(WORK_DIR)
model_path = os.path.join(WORK_DIR, 'models/model_base_retrieval_coco.pth')
# print(model_path)

# print(os.getcwd())
if os.getcwd() != WORK_DIR:
    print("Changing to proper working directory...")
    os.chdir(WORK_DIR)
    print(f"Done, working directory: {os.getcwd()}")
