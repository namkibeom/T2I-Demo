import io
import os
#import ray
from multiprocessing import Process

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from flask import Flask, jsonify, request
#from model.model import PestClassifier

from inference.inference_for_demo  import minDALLE, GLIDE, StableDiffusion, StableDiffusion2, Karlo
import sys
sys.path.append('../flask/karlo/karlo/sampler')
from template import BaseSampler

from flask import Flask
app = Flask(__name__)




@app.route('/inference1', methods=['GET']) # url 패턴 정의
#@ray.remote(num_gpus=0.25)
def inference1():
    if request.method == 'GET':
        #####data read
        file = request.files['file']
        prompt_all = file.read()
        prompt_all = prompt_all.decode()
        #print(type(prompt_all))

        #####inference
        save_file_mindalle = mindalle.txt2img(prompt_all)
        save_file_mindalle = jsonify({'inference_result': save_file_mindalle})

        return save_file_mindalle

#@ray.remote
@app.route('/inference2', methods=['GET']) 
#@ray.remote(num_gpus=0.25)
def inference2():
    if request.method == 'GET':
        #####data read
        file = request.files['file']
        prompt_all = file.read()
        prompt_all = prompt_all.decode()
        #print(type(prompt_all))

        #####inference
        save_file_glide = glide.txt2img(prompt_all)
        save_file_glide = jsonify({'inference_result': save_file_glide})
        
        return save_file_glide

#@ray.remote
@app.route('/inference3', methods=['GET']) 
#@ray.remote(num_gpus=0.25)
def inference3():
    if request.method == 'GET':
        #####data read
        file = request.files['file']
        prompt_all = file.read()
        prompt_all = prompt_all.decode()
        #print(type(prompt_all))

        #####inference
        save_file_sd = sd.txt2img(prompt_all, opt1, model, sampler, outpath, wm_encoder, sample_path, base_count, grid_count, start_code, precision_scope) 
        save_file_sd = jsonify({'inference_result': save_file_sd})
        
        return save_file_sd

#@ray.remote
@app.route('/inference4', methods=['GET']) 
#@ray.remote(num_gpus=0.25)
def inference4():
    if request.method == 'GET':
        #####data read
        file = request.files['file']
        prompt_all = file.read()
        prompt_all = prompt_all.decode()

        #####inference
        save_file_sd2 = sd2.txt2img(prompt_all, opt2, model2, sampler2, outpath2, wm_encoder2, sample_path2, sample_count2, base_count2, grid_count2, start_code2, precision_scope2) 
        save_file_sd2 = jsonify({'inference_result': save_file_sd2})

        return save_file_sd2

#@ray.remote
@app.route('/inference5', methods=['GET']) 
#@ray.remote(num_gpus=1)
def inference5():
    if request.method == 'GET':
        #####data read
        file = request.files['file']
        prompt_all = file.read()
        prompt_all = prompt_all.decode()

        #####inference
        karlo_generator = karlo.__call__(prompt_all, 3)
        save_file_karlo = karlo._sample(karlo_generator)

        save_file_karlo = jsonify({'inference_result': save_file_karlo})
        
        return save_file_karlo


if __name__ == '__main__':
    #ray.init()
    mindalle = minDALLE()
    glide = GLIDE()
    sd = StableDiffusion()
    sd2 = StableDiffusion2()
    karlo = Karlo(BaseSampler)
    
    mindalle.load()
    glide.load()
    opt1 = sd.parse()
    model, sampler, outpath, wm_encoder,sample_path, base_count, grid_count, start_code, precision_scope = sd.load(opt1)
    opt2 = sd2.parse_args()
    model2, sampler2, outpath2, wm_encoder2, sample_path2, sample_count2, base_count2, grid_count2, start_code2, precision_scope2 = sd2.load(opt2)

    app.run(host='127.0.0.1', port=5000, threaded=True)