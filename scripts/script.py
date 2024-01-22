import gradio as gr
import gc,os,safetensors.torch,torch
from modules import scripts, devices, processing,shared,sd_hijack,sd_models,script_loading,paths, sd_models_config
from modules.modelloader import load_file_from_url
from tqdm import tqdm
from copy import deepcopy

networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

EXTENSION_NAME = 'Inpaint Model Converter'
EXTENSION_ROOT = scripts.basedir()
ext2abs = lambda *x: os.path.join(EXTENSION_ROOT,*x)

INPAINT_PANTS_URL = 'https://huggingface.co/groinge/inpaintdifferencemerge/resolve/main/V1_5inpaintdifference.safetensors'
INPAINT_PANTS = ext2abs('scripts','V1_5inpaintdifference.safetensors')

rejected_model_id = None

class script(scripts.Script):
        def title(self):
            return EXTENSION_NAME

        def show(self,is_img2img):
            return scripts.AlwaysVisible
        
        def ui(self,is_img2img):
            with gr.Accordion(label=EXTENSION_NAME,visible=is_img2img,open=False):
                checkbox = gr.Checkbox(label='Enable')
                if not os.path.exists(INPAINT_PANTS):
                    checkbox.input(fn=download_model,inputs=checkbox,outputs=checkbox)

            return [checkbox]

        def setup(self,p,enabled):
            if shared.sd_model:
                if isinstance(p, processing.StableDiffusionProcessingImg2Img) and p.image_mask and enabled:
                    if shared.sd_model.used_config != sd_models_config.config_inpainting:
                        convert()
                elif shared.sd_model.used_config == sd_models_config.config_inpainting:
                    convert(reverse=True)


def convert(reverse=False):
    if not shared.sd_model.is_sd1 or not (shared.sd_model.used_config == sd_models_config.config_default or shared.sd_model.used_config == sd_models_config.config_inpainting):
        #Save rejected model to prevent spam
        if shared.sd_model.sd_checkpoint_info.ids != rejected_model_id:
            gr.Warning(EXTENSION_NAME+': unsupported model.')
        rejected_model_id = shared.sd_model.sd_checkpoint_info.ids
        return

    with torch.no_grad():
        for module in shared.sd_model.modules():
            networks.network_restore_weights_from_backup(module)

    device = devices.get_optimal_device_name()
    shared.sd_model.to(device)

    sd_hijack.model_hijack.undo_hijack(shared.sd_model)

    checkpoint_info = deepcopy(shared.sd_model.sd_checkpoint_info)
    model_dict = shared.sd_model.state_dict()
    pants_dict = safetensors.torch.load_file(INPAINT_PANTS, device=device)

    if not reverse:
        checkpoint_info.name_for_extra = f"temp-inpainting-{checkpoint_info.name_for_extra}"
        for key in tqdm(list(pants_dict.keys()),desc='Merging models'):

            t0 = model_dict[key]
            t1 = pants_dict[key]

            a = list(t0.shape)
            b = list(t1.shape)

            #https://github.com/hako-mikan/sd-webui-supermerger 🙏
            if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
                t1[:, 0:4, :, :] = t1[:, 0:4, :, :] + t0
                model_dict[key] = t1
            else:
                model_dict[key] = t1 + t0
    else:
        checkpoint_info.name_for_extra = checkpoint_info.name_for_extra.replace('temp-inpainting-','')
        for key in tqdm(list(pants_dict.keys()),desc='Merging models'):

            t0 = model_dict[key]
            t1 = pants_dict[key]

            shape = list(t0.shape)
            
            if len(shape) == 4 and shape[1] == 9:
                model_dict[key] =  t0[:, 0:4, :, :] - t1[:, 0:4, :, :]
            else:
                model_dict[key] = t0 - t1
        

    del pants_dict
    sd_models.model_data.loaded_sd_models.remove(sd_models.model_data.sd_model)
    sd_models.model_data.sd_model = None
    sd_models.load_model(checkpoint_info=checkpoint_info, already_loaded_state_dict=model_dict)
    del model_dict

    gc.collect()
    devices.torch_gc()
    gr.Info(EXTENSION_NAME+':  Successfully converted model.')

def download_model(checkbox):
    if not os.path.exists(INPAINT_PANTS):
        gr.Info(EXTENSION_NAME+": Downloading inpainting unet (1.68 GB)...")
        load_file_from_url(INPAINT_PANTS_URL, model_dir=ext2abs('scripts'))
        gr.Info(EXTENSION_NAME+": Download completed.")
    return checkbox