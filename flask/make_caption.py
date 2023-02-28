import os
import json, pandas as pd


outpath = '../etri-vlm-web/flask/eval/coco_captions/captions_val2014.json' 

with open('../etri-vlm-web/flask/annotations/captions_val2014.json', 'r') as f:
  data=json.load(f)


data = data['annotations']
data = pd.DataFrame(data)
#print(data)
data = data.drop_duplicates(['image_id'])
data = data.sort_values('image_id')
data = data.reset_index(drop=True)
print(data)
#data.to_csv("image-caption.csv")

grid_count = 0
caption_file = dict()
for i in range(len(data['caption'])):
              text = data['caption'][grid_count]
              index = f'grid-{grid_count:04}'
              caption_file[index] = text
              grid_count += 1

              #if grid_count >= 2100 :
              #  break
                   
#print(caption_file)             
with open(outpath, 'w') as outfile:
              json.dump(caption_file, outfile)















"""

import requests
import json
import json, pandas as pd


#with open('/home/user/namkb/flask/eval/coco_captions/glide_captions.json', 'r') as f:
#  data=json.load(f)
#data = data['annotations']

#print(data)



with open('/home/user/namkb/flask/annotations/captions_val2014.json', 'r') as f:
  data=json.load(f)

#keys = [key for key in data]
#print(keys)

data = data['annotations']
data = pd.DataFrame(data)
data = data.drop_duplicates(['image_id'])
data = data.sort_values('image_id')
data = data.reset_index(drop=True)


grid_count = 0
for i in range(len(data['caption'])):
  prompt_all = data['caption'][grid_count]
  resp2 = requests.get("http://127.0.0.1:5000/inference2",  files={"file": prompt_all})
  save_file_glide  = resp2.json()
  save_file_glide  = save_file_glide['inference_result']
  
  # 인퍼런스 결과 출력
  print(save_file_glide)
  grid_count += 1
"""