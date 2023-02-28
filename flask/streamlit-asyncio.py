
import streamlit as st   
import googletrans     
import requests
import asyncio
from PIL import Image
import datetime
from multiprocessing import Process
#import requests_async as requests
import functools
from functools import partial

async def run(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, call)                   

async def api_1(prompt_all): 
    #st.subheader("minDALL-E ver.")
    before_inference_time = datetime.datetime.now()
    print(f'running api Number : 1')
    url = "http://127.0.0.1:5000/inference1"
    resp1= run(requests.get, url, files={"file": prompt_all})
    resp1 = await resp1
    save_file_mindalle = resp1.json()
    save_file_mindalle = save_file_mindalle['inference_result']
    image = Image.open(save_file_mindalle)
    st.image(image, caption=prompt_all+','+save_file_mindalle)
    st.subheader("minDALL-E ver.")
    print(f'end api Number : 1')
    after_inference_time = datetime.datetime.now()
    st.write(' - 소요 시간 : ')
    st.write(after_inference_time - before_inference_time)# 인퍼런스 소요 시간 출력
    st.write('')
    #return image, save_file_mindalle

async def api_2(prompt_all):
    #st.subheader("GLIDE ver.")
    before_inference_time2 = datetime.datetime.now()
    print(f'running api Number : 2')
    url = "http://127.0.0.1:5000/inference2"
    resp2= run(requests.get, url, files={"file": prompt_all})
    resp2 = await resp2
    save_file_glide = resp2.json()
    save_file_glide = save_file_glide['inference_result']
    image_glide = Image.open(save_file_glide)
    st.image(image_glide, caption=prompt_all+','+save_file_glide)
    st.subheader("GLIDE ver.")
    print(f'end api Number : 2')
    after_inference_time2 = datetime.datetime.now()
    st.write(' - 소요 시간 : ')
    st.write(after_inference_time2 - before_inference_time2)# 인퍼런스 소요 시간 출력
    st.write('')
    #return image_glide, save_file_glide

async def api_3(prompt_all):
    #st.subheader("Stable Diffusion ver.")
    before_inference_time3 = datetime.datetime.now()
    print(f'running api Number : 3')
    url = "http://127.0.0.1:5000/inference3"
    resp3= run(requests.get, url, files={"file": prompt_all})
    resp3 = await resp3
    save_file_sd = resp3.json()
    save_file_sd = save_file_sd['inference_result']
    image_sd = Image.open(save_file_sd)
    st.image(image_sd, caption=prompt_all+','+save_file_sd)
    st.subheader("Stable Diffusion ver.")
    print(f'end api Number : 3')
    after_inference_time3 = datetime.datetime.now()
    st.write(' - 소요 시간 : ')
    st.write(after_inference_time3 - before_inference_time3) # 인퍼런스 소요 시간 출력
    st.write('')
    #return image_sd, save_file_sd

async def api_4(prompt_all):
    #st.subheader("Stable Diffusion ver 2.")
    before_inference_time4 = datetime.datetime.now()
    print(f'running api Number : 4')
    url = "http://127.0.0.1:5000/inference4"
    resp4= run(requests.get, url, files={"file": prompt_all})
    resp4 = await resp4
    save_file_sd2 = resp4.json()
    save_file_sd2 = save_file_sd2['inference_result']
    image_sd2 = Image.open(save_file_sd2)
    st.image(image_sd2, caption=prompt_all+','+save_file_sd2)
    st.subheader("Stable Diffusion ver 2.")
    print(f'end api Number : 4')
    after_inference_time4 = datetime.datetime.now()
    st.write(' - 소요 시간 : ')
    st.write(after_inference_time4 - before_inference_time4)# 인퍼런스 소요 시간 출력
    st.write('')
    #return image_sd2, save_file_sd2

async def api_5(prompt_all):
    #st.subheader("karlo ver.")
    before_inference_time5 = datetime.datetime.now()
    print(f'running api Number : 5')
    url = "http://127.0.0.1:5000/inference5"
    resp5= run(requests.get, url, files={"file": prompt_all})
    resp5 = await resp5
    save_file_karlo = resp5.json()
    save_file_karlo = save_file_karlo['inference_result']
    image_karlo = Image.open(save_file_karlo)
    st.image(image_karlo, caption=prompt_all+','+save_file_karlo)
    st.subheader("karlo ver.")
    print(f'end api Number : 5')
    after_inference_time5 = datetime.datetime.now()
    st.write(' - 소요 시간 : ')
    st.write(after_inference_time5 - before_inference_time5)# 인퍼런스 소요 시간 출력
    st.write('')
    #return image_karlo, save_file_karlo

async def main2(prompt_all):
    await asyncio.gather(
        api_1(prompt_all),
        api_2(prompt_all),
        api_3(prompt_all),
        api_4(prompt_all),
        api_5(prompt_all)
    )    


def main():


    st.title("Text-to-Image Generation")

    st.info("입력예:\n\n" \
            "말을 타고 우주 비행사의 사진, a photograph of an astronaut riding a horse \n\n" \
            "별이 빛나는 밤 스타일의 여우 그림, a painting of a fox in the style of starry night")

    input_str = st.text_input(label="Text Prompt(한글 또는 영어)", key="input_sd", value="말을 타고  우주 비행사의 사진")

    if st.button("Submit"):

        #####################################################################
        # minDALL-E version
        #####################################################################
        #st.subheader("minDALL-E ver.")
        
        print('=======> translate(IN) done.')
        translator = googletrans.Translator()
        if translator.detect(input_str).lang == 'ko':
            result = translator.translate(input_str, src='ko',dest='en')
            prompt_all = result.text
        else:
            prompt_all = input_str
        
        print('=======> translate(OUT) done.')
        #prompt_all = input_str
        print('=======> txt2imge(IN) done.')
        
        #loop = asyncio.new_event_loop()
        #task = loop.create_task(main())
        #loop.run_until_complete(task)

        asyncio.run(main2(prompt_all))
        #loop = asyncio.new_event_loop()
        #asyncio.set_event_loop(loop)
        #task = loop.create_task(main2(prompt_all))
        #loop.run_until_complete(task)
        #loop.close()


    
    
        print('=======> txt2imge(OUT) done.')

        #st.image(image, caption=prompt_all+','+save_file_mindalle



        print('======> done...')


        #####################################################################
        # GLIDE version
        #####################################################################
        #st.subheader("GLIDE ver.")

        #glide.load()
        
        #prompt_glide = input_str

        print('=======> txt2imge(IN) done.')

        #resp2 = requests.get("http://127.0.0.1:5000/inference2",  files={"file": prompt_all})
        #task2 = asyncio.create_task(api_2(prompt_all))
        #image_glide, save_file_glide = await task2
        
        #save_file_glide = glide.txt2img(prompt_glide)
        print('=======> txt2imge(OUT) done.')

        #st.image(image_glide, caption=prompt_all+','+save_file_glide)
        




        print('======> done...')

 
        #####################################################################
        # Stable Diffusion version
        #####################################################################
        #st.subheader("Stable Diffusion ver.")

        print('=======> parse(IN) done.')
        #opt = sd.parse()
        print('=======> parse(OUT) done.')

        print('=======> load(IN) done.')
        #model, sampler, outpath, wm_encoder,sample_path, base_count, grid_count, start_code, precision_scope = sd.load(opt)
        print('=======> load(OUT) done.')

     
        #prompt_sd = input_str
    
        print('=======> txt2imge(IN) done.')
        #before_inference_time3 = datetime.datetime.now()
        #resp3 = requests.get("http://127.0.0.1:5000/inference3",  files={"file": prompt_all})
        #task3 = asyncio.create_task(api_3(prompt_all))
        #image_sd, save_file_sd = await task3
       
        
        print('=======> txt2imge(OUT) done.')

        #st.image(image_sd, caption=prompt_all+','+save_file_sd)

        #after_inference_time3 = datetime.datetime.now()

       # st.write(' - 소요 시간 : ')
        #st.write(after_inference_time3 - before_inference_time3)# 인퍼런스 소요 시간 출력


        print('======> done...')

        #####################################################################
        # Stable Diffusion version 2
        #####################################################################
        #st.subheader("Stable Diffusion ver 2.")

        print('=======> parse(IN) done.')
        #opt = sd2.parse_args()
        print('=======> parse(OUT) done.')

        print('=======> load(IN) done.')
        #model, sampler, outpath, wm_encoder, sample_path, sample_count, base_count, grid_count, start_code, precision_scope = sd2.load(opt)
        print('=======> load(OUT) done.')
        
        
        print('=======> txt2imge(IN) done.')

    
        #resp4 = requests.get("http://127.0.0.1:5000/inference4",  files={"file": prompt_all})
        #task4 = asyncio.create_task(api_4(prompt_all))
        #image_sd2, save_file_sd2 = await task4
       
        print('=======> txt2imge(OUT) done.')
        #st.image(image_sd2, caption=prompt_all+','+save_file_sd2)



        print('======> done...')
    
        #####################################################################
        # karlo version
        #####################################################################
        #st.subheader("karlo ver.")
        

        print('=======> txt2imge(IN) done.')

        #resp5 = requests.get("http://127.0.0.1:5000/inference5",  files={"file": prompt_all})
        #task5 = asyncio.create_task(api_5(prompt_all))
        #image_karlo, save_file_karlo = await task5
        #karlo_geneator = karlo.__call__(prompt_karlo, 3)
   
        print('=======> txt2imge(OUT) done.')

        #save_file_karlo = karlo._sample(karlo_geneator)
        #st.image(image_karlo, caption=prompt_all+','+save_file_karlo)



        #st.write(' - 총 소요 시간 : ')
        #st.write(after_inference_time5 - before_inference_time)# 인퍼런스 총 소요 시간 출력
        

        print('======> done...')





    st.title("Text-to-Video Generation")
    st.subheader("TBD")

if __name__ == "__main__":
    main()
    #asyncio.run(main())
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    # loop.close()