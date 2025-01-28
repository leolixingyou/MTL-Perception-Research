import os
import json
import os
import json

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
   return output

def process_json_files(input_dir, output_dir):
    # 创建集合来存储所有不重复的category
    categories = set()
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # 处理训练集
    with open(os.path.join(input_dir, 'det_train.json')) as f:
        train_data = json.load(f)
        for item in train_data[:100]:
            for label in item['labels']:
                categories.add(label['category'])
            output_data = convert_format(item)

                
    # 处理验证集
    with open(os.path.join(input_dir, 'det_val.json')) as f:
        val_data = json.load(f)
        for item in val_data[:100]:
            for label in item['labels']:
                categories.add(label['category'])
            output_data = convert_format(item)

    
    print("所有类别:", sorted(list(categories)))
    print("类别总数:", len(categories))

if __name__ == "__main__":
    # 使用示例
    input_dir = ROOT_PATH  # 包含det_train.json和det_val.json的目录
    output_dir = f'{ROOT_PATH}/output' # 输出目录
    process_json_files(input_dir, output_dir)