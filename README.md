Automatically converts the current active SD V1.5 model in to an inpainting model when inpainting. It accomplishes this in 5-10 seconds by merging the active model with a unet containing the difference between SD V1-5 inpainting and SD V1-5, which gives the same end result as a normal inpainting merge.

Please note:
- This will download a 1.68 GB model when activated for the first time.
- This only supports checkpoints based on SD 1.5. I'm not planning to add support for other versions of SD.
- Expect bugs. It Works For Me™ but I don't have a ton of faith in it.

This adapts snippets of code and some concepts from https://github.com/hako-mikan/sd-webui-supermerger
