import os
import json
import os
import json
from tqdm import tqdm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
print(ROOT_PATH)

def convert_format(input_json):
   output = {
       'name': input_json['name'].split('.')[0],
       'frames': [{
           'timestamp': input_json['timestamp'],
           'objects': []
       }],
       'attributes': input_json['attributes']
   }
   
   try:
    for label in input_json['labels']:
        obj = {
            'category': label['category'],
            'id': int(label['id']),
            'attributes': {
                'occluded': label['attributes']['occluded'],
                'truncated': label['attributes']['truncated'],
                'trafficLightColor': 'green' if label['attributes']['trafficLightColor'] == 'G' else 'none'
            }
        }
        if 'box2d' in label:
            obj['box2d'] = label['box2d']
        output['frames'][0]['objects'].append(obj)
   except:
       pass
   return output

def process_json_files(input_dir, output_dir):
   # 创建输出目录
   os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
   os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
   
   # 处理训练集
   with open(os.path.join(input_dir, 'det_train.json')) as f:
       train_data = json.load(f)
   
#    for item in train_data[:100]:
   for item in tqdm(train_data):
       output_data = convert_format(item)
       output_file = os.path.join(output_dir, 'train', f"{output_data['name']}.json")
       with open(output_file, 'w') as f:
           json.dump(output_data, f, indent=2)
           
   # 处理验证集
   with open(os.path.join(input_dir, 'det_val.json')) as f:
       val_data = json.load(f)
   
#    for item in val_data[:100]:
   for item in tqdm(val_data):
       output_data = convert_format(item)
       output_file = os.path.join(output_dir, 'val', f"{output_data['name']}.json")
       with open(output_file, 'w') as f:
           json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    # 使用示例
    input_dir = ROOT_PATH  # 包含det_train.json和det_val.json的目录
    output_dir = f'{ROOT_PATH}/resampled_detect' # 输出目录
    process_json_files(input_dir, output_dir)