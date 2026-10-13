I'm going to make my own Safetensors loading library for Julia.

- `__metadata__`

I won't do lazy loading since for me most cases I first build a `model` struct then use it in the inference engine...
