import requests
import json
import json, pandas as pd


with open('/home/user/namkb/etri-vlm-web/flask/annotations/captions_val2014.json', 'r') as f:
  data=json.load(f)


data = data['annotations']
data = pd.DataFrame(data)
#print(data)
data = data.drop_duplicates(['image_id']) #(default : 중복되는 것 중 처음 값을 남김)
data = data.sort_values('image_id')
data = data.reset_index(drop=True)
print(data)


grid_count = 0
for i in range(len(data['caption'])):
  prompt_all = data['caption'][grid_count]
  resp1 = requests.get("http://127.0.0.1:5000/inference1",  files={"file": prompt_all})
  resp2 = requests.get("http://127.0.0.1:5000/inference2",  files={"file": prompt_all})
  resp3 = requests.get("http://127.0.0.1:5000/inference3",  files={"file": prompt_all})
  resp4 = requests.get("http://127.0.0.1:5000/inference4",  files={"file": prompt_all})
  resp5 = requests.get("http://127.0.0.1:5000/inference5",  files={"file": prompt_all})
  #save_file_glide  = resp2.json()
  #save_file_glide  = save_file_glide['inference_result']
  # 인퍼런스 결과 출력
  #print(save_file_glide)
  grid_count += 1


