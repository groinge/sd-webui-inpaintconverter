import gradio as gr
from modules import scripts, devices, processing
from modules.sd_models import *
from modules.modelloader import load_file_from_url
from tqdm import tqdm
from copy import deepcopy
import json
import gc
import safetensors.torch
import os
import torch

extension_name = 'Reload-as-inpaint'
EXTENSION_ROOT = scripts.basedir()
ext2abs = lambda x: os.path.join(EXTENSION_ROOT,x)

with open(ext2abs('config.json'), 'r') as f:
    config = json.load(f)

ARCHITECTURES = {
    'V1.5-inpaint': {
        'name' : 'V1.5-inpaint-diff',
        #Which element, shape index, and dimension to ID the architecture by
        'id' : ['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight',1,768],
        'type_id' : ['model.diffusion_model.input_blocks.0.0.weight',1,4],
        'filename' : 'models/V1_5inpaintdifference.safetensors',
        'url' : 'https://huggingface.co/groinge/inpaintdifferencemerge/resolve/main/V1_5inpaintdifference.safetensors'
    }
}

class script(scripts.Script):
        checkpointinfo = None
        display = None
        def title(self):
            return extension_name

        def show(self,is_img2img):
            return scripts.AlwaysVisible
        
        def ui(self,is_img2img):
            with gr.Accordion(label=extension_name,visible=is_img2img,open=False):
                with gr.Row(variant='default'):
                    sets = gr.Checkboxgroup(value = config['sets'],choices=[("Auto convert","auto"),("Use cuda for merging","cuda")],label='Options')
                    runbutton = gr.Button(value = 'Convert to inpaint',variant='primary')
                    unloadbutton = gr.Button(value = 'Revert to standard model',variant='secondary')
                with gr.Row():
                    if is_img2img:
                        script.display = gr.Textbox(label="",value="",interactive=False,max_lines=1)
                
                runbutton.click(fn=self.run,inputs=sets,outputs=script.display)
                unloadbutton.click(fn=self.unload,outputs=script.display)
                sets.change(fn=sets_config,inputs=sets)
            return [sets]

        def setup(self,p,sets):
            if model_data.sd_model:
                if 'auto' in sets and isinstance(p, processing.StableDiffusionProcessingImg2Img) and p.image_mask and not model_data.sd_model.sd_checkpoint_info.name.startswith('temp-inpainting-'):
                    self.run(sets)
                else: self.unload()

        def run(self,use_cuda):
            message,script.checkpointinfo = convert(use_cuda) #checkpoint info of the original model is kept so it can be reloaded
            script.display.update(value=extension_name+':  '+message) 
        
        def unload(self):
            if model_data.sd_model.sd_checkpoint_info.name.startswith('temp-inpainting-'):
                gr.Info(extension_name+':  Reverting to standard model...')
                model_data.__init__()
                load_model(checkpoint_info=script.checkpointinfo)
                print('Reverted to previous checkpoint.')
                gr.Info(extension_name+':  Reverted to previous checkpoint.')
                script.display.update(value='Reverted to previous checkpoint.')


def sets_config(sets):
    config['sets'] = sets
    with open(ext2abs('config.json'), 'w') as f:
        json.dump(config,f)

def convert(sets):
    device = 'cuda' if 'cuda' in sets else 'cpu'

    if not model_data.sd_model: return "No checkpoint loaded.",None
    sd_0, checkpoint_info, arch, message = grab_active_model_sd(device)
    if not sd_0: return message,checkpoint_info
    fake_cp_info = deepcopy(checkpoint_info)

    sd_1 = load_inpaint_statedict(arch, device)

    for k1 in tqdm(list(sd_1.keys()),desc='Merging models'):
        k0 = 'model.diffusion_model.' + k1

        #This makes sure that only the layers shared between a standard and inpainting unet are merged
        #Copied from https://github.com/hako-mikan/sd-webui-supermerger
        a = list(sd_0[k0].shape)
        b = list(sd_1[k1].shape)
        if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
            sd_1[k1][:, 0:4, :, :] = sd_1[k1][:, 0:4, :, :] + sd_0[k0]
            sd_0[k0] = sd_1[k1]
        else:
            sd_0[k0] = sd_1[k1] + sd_0[k0]

    fake_cp_info.name = f"temp-inpainting-{checkpoint_info.name}"

    del sd_1
    load_model(checkpoint_info=fake_cp_info, already_loaded_state_dict=sd_0)
    del sd_0

    gc.collect()
    devices.torch_gc()
    gr.Info(extension_name+':  Successfully converted model.')
    return 'Successfully converted model.',checkpoint_info

def load_inpaint_statedict(arch,device):
    if os.path.isfile(ext2abs(arch['filename'])):
        file = safetensors.torch.load_file(ext2abs(arch['filename']), device=device)
    else:
        gr.Info(f"Downloading {arch['name']} unet (1.68 GB)...")
        file_path = load_file_from_url(arch['url'],model_dir=ext2abs('models'))
        file = safetensors.torch.load_file(file_path, device=device)

    return file.get('state_dict') or file

def grab_active_model_sd(device):
    state_dict = model_data.sd_model.state_dict()

    for info in ARCHITECTURES.values():
        elname, idindex, iddim = info['id']
        elem = state_dict.get(elname)
        #Identifies the architecture of the model by checking if it contains a known unique elem and dimension
        if list(elem) and list(elem.shape)[idindex] == iddim:
            elname, idindex, iddim = info['type_id']
            elem = state_dict.get(elname)
            #Makes sure its a standard model (Not inpaint or pix2pix)
            if list(elem.shape)[idindex] == iddim:
                arch = info
                break
            else:
                del state_dict; return None,None,None,'Loaded checkpoint is already an inpainting or pix-to-pix model'
    else:
        del state_dict; return None,None,None,'Unsupported model architecture, this extension only supports V1.5 checkpoints'
    del state_dict
    
    gr.Info(extension_name+':  Converting model to inpaint...')

    model_data.sd_model.to(torch.device(device))
    from modules import sd_hijack
    sd_hijack.model_hijack.undo_hijack(model_data.sd_model) #Undo hijack to get a state_dict without optimizations
    sd_0 = deepcopy(model_data.sd_model.state_dict())
    checkpoint_info = model_data.sd_model.sd_checkpoint_info

    unload_model_weights(model_data.sd_model)
    gc.collect()
    devices.torch_gc()

    return sd_0,checkpoint_info,arch,None

    