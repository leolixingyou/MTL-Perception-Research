import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))+os.sep
print(ROOT_PATH)

def organize_files(output_dir, source_dirs, dest_dirs):
   for dest_dir in dest_dirs.values():
       Path(dest_dir).mkdir(parents=True, exist_ok=True)
  
   for split in ['train', 'val']:
       json_dir = Path(output_dir) / split
       
       for json_file in tqdm(list(json_dir.glob('*.json'))):
           basename = json_file.stem
           
           # 复制并转换图片文件
           src_img = Path(source_dirs['images']) / split / f"{basename}.jpg"
           dst_img = Path(dest_dirs['images']) / split / f"{basename}.jpg"
           if src_img.exists():
               dst_img.parent.mkdir(parents=True, exist_ok=True)
               img = cv2.imread(str(src_img))
               cv2.imwrite(str(dst_img), img)
           
           # 复制并转换drivable mask
           src_drive = Path(source_dirs['drivable']) / split / f"{basename}.png"
           dst_drive = Path(dest_dirs['drivable']) / split / f"{basename}.png"
           if src_drive.exists():
               dst_drive.parent.mkdir(parents=True, exist_ok=True)
               mask = cv2.imread(str(src_drive))
               cv2.imwrite(str(dst_drive), mask)
           
           # 复制并转换lane mask
           src_lane = Path(source_dirs['lane']) / split / f"{basename}.png"
           dst_lane = Path(dest_dirs['lane']) / split / f"{basename}.png"
           if src_lane.exists():
               dst_lane.parent.mkdir(parents=True, exist_ok=True)
               mask = cv2.imread(str(src_lane))
               cv2.imwrite(str(dst_lane), mask)
           
           # 复制detection json
           dst_det = Path(dest_dirs['det']) / split / f"{basename}.json"
           dst_det.parent.mkdir(parents=True, exist_ok=True)
           shutil.copy2(json_file, dst_det)

# 源目录和目标目录配置
source_dirs = {
   'images': '/workspace/bdd100k/images',
   'drivable': '/workspace/bdd100k/labels/drivable/masks',
   'lane': '/workspace/bdd100k/labels/lane/masks'
}

dest_dirs = {
   'images': f'{ROOT_PATH}/sample_dataset/images',
   'drivable': f'{ROOT_PATH}/sample_dataset/drivable',
   'lane': f'{ROOT_PATH}/sample_dataset/lane',
   'det': f'{ROOT_PATH}/sample_dataset/det'
}




organize_files(f'{ROOT_PATH}/output', source_dirs, dest_dirs)