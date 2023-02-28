
import streamlit as st   
import googletrans     
import requests
from PIL import Image
import datetime


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
        st.subheader("minDALL-E ver.")
        
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
        #print(type(prompt_))
        #save_file_mindalle, save_file_glide, save_file_sd, save_file_sd2, save_file_karlo = 
        before_inference_time = datetime.datetime.now()
        resp1 = requests.get("http://127.0.0.1:5000/inference1",  files={"file": prompt_all})
        save_file_mindalle = resp1.json()
        save_file_mindalle = save_file_mindalle['inference_result']
    
        print('=======> txt2imge(OUT) done.')

        image = Image.open(save_file_mindalle)
        st.image(image, caption=prompt_all+','+save_file_mindalle)
    
        after_inference_time = datetime.datetime.now()

        st.write(' - 소요 시간 : ')
        st.write(after_inference_time - before_inference_time)# 인퍼런스 소요 시간 출력

        print('======> done...')


        #####################################################################
        # GLIDE version
        #####################################################################
        st.subheader("GLIDE ver.")

        #glide.load()
        
        #prompt_glide = input_str

        print('=======> txt2imge(IN) done.')
        before_inference_time2 = datetime.datetime.now()
        resp2 = requests.get("http://127.0.0.1:5000/inference2",  files={"file": prompt_all})
    
        save_file_glide = resp2.json()
        save_file_glide = save_file_glide['inference_result']
        #save_file_glide = glide.txt2img(prompt_glide)
        print('=======> txt2imge(OUT) done.')

        image_glide = Image.open(save_file_glide)
        st.image(image_glide, caption=prompt_all+','+save_file_glide)
        
        after_inference_time2 = datetime.datetime.now()

        st.write(' - 소요 시간 : ')
        st.write(after_inference_time2 - before_inference_time2)# 인퍼런스 소요 시간 출력

        print('======> done...')

 
        #####################################################################
        # Stable Diffusion version
        #####################################################################
        st.subheader("Stable Diffusion ver.")

        print('=======> parse(IN) done.')
        #opt = sd.parse()
        print('=======> parse(OUT) done.')

        print('=======> load(IN) done.')
        #model, sampler, outpath, wm_encoder,sample_path, base_count, grid_count, start_code, precision_scope = sd.load(opt)
        print('=======> load(OUT) done.')

     
        #prompt_sd = input_str
    
        print('=======> txt2imge(IN) done.')
        before_inference_time3 = datetime.datetime.now()
        resp3 = requests.get("http://127.0.0.1:5000/inference3",  files={"file": prompt_all})
    
        save_file_sd = resp3.json()
        save_file_sd = save_file_sd['inference_result']
        
        print('=======> txt2imge(OUT) done.')

        image = Image.open(save_file_sd)
        st.image(image, caption=prompt_all+','+save_file_sd)

        after_inference_time3 = datetime.datetime.now()

        st.write(' - 소요 시간 : ')
        st.write(after_inference_time3 - before_inference_time3)# 인퍼런스 소요 시간 출력

        print('======> done...')

        #####################################################################
        # Stable Diffusion version 2
        #####################################################################
        st.subheader("Stable Diffusion ver 2.")

        print('=======> parse(IN) done.')
        #opt = sd2.parse_args()
        print('=======> parse(OUT) done.')

        print('=======> load(IN) done.')
        #model, sampler, outpath, wm_encoder, sample_path, sample_count, base_count, grid_count, start_code, precision_scope = sd2.load(opt)
        print('=======> load(OUT) done.')
        
        
        print('=======> txt2imge(IN) done.')

        before_inference_time4 = datetime.datetime.now()
        resp4 = requests.get("http://127.0.0.1:5000/inference4",  files={"file": prompt_all})
    
        save_file_sd2 = resp4.json()
        save_file_sd2 = save_file_sd2['inference_result']
       
        print('=======> txt2imge(OUT) done.')

        image = Image.open(save_file_sd2)
        st.image(image, caption=prompt_all+','+save_file_sd2)

        after_inference_time4 = datetime.datetime.now()

        st.write(' - 소요 시간 : ')
        st.write(after_inference_time4 - before_inference_time4)# 인퍼런스 소요 시간 출력

        print('======> done...')
    
        #####################################################################
        # karlo version
        #####################################################################
        st.subheader("karlo ver.")
        

        print('=======> txt2imge(IN) done.')
        before_inference_time5 = datetime.datetime.now()
        resp5 = requests.get("http://127.0.0.1:5000/inference5",  files={"file": prompt_all})
    
        save_file_karlo = resp5.json()
        save_file_karlo = save_file_karlo['inference_result']
        #karlo_geneator = karlo.__call__(prompt_karlo, 3)
   
        print('=======> txt2imge(OUT) done.')

        #save_file_karlo = karlo._sample(karlo_geneator)
        image = Image.open(save_file_karlo)
        st.image(image, caption=prompt_all+','+save_file_karlo)

        after_inference_time5 = datetime.datetime.now()

        st.write(' - 소요 시간 : ')
        st.write(after_inference_time5 - before_inference_time5)# 인퍼런스 소요 시간 출력


        st.write(' - 총 소요 시간 : ')
        st.write(after_inference_time5 - before_inference_time)# 인퍼런스 총 소요 시간 출력
        

        print('======> done...')





    st.title("Text-to-Video Generation")
    st.subheader("TBD")

if __name__ == "__main__":
    main()