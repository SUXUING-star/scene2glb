
# Blender/mmd scene to GLB Converter

a bpy pipeline to automatically convert MMD scene models into the glb format.

Notice: The project is in testing,probably makes few bugs.

---

## Requirements

1. **Blender** (4 ~ 5.2)
2. **Required Add-ons:**
   * **[mmd_tools](https://github.com/powup333/blender_mmd_tools)**

---

## Model Configuration (`config.json`)

Make a empty `config.json` in the model directory.

```json
{
  "jp_name": "",
  "en_name": "",
}
```


---

## Usage

1. **Import PMX:** Import the `.pmx` model into Blender using `mmd_tools` and convert materials to Principled BSDF. Save the `.blend` file.
2. **Verify Setup:** Ensure `config.json` and `Texture2D/`(or other name) are in the same folder as the `.blend` file.
3. **Run Pipeline:** Open `blender.py` in Text Editor and edit the code path(the repsitory clone path),then you just copy the `blender.py` into the Blender's Text Editor,and click **Run Script**,the model will export in your project path.
