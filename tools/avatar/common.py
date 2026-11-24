import sys, bpy, addon_utils, importlib
def load_mpfb():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("bl_ext.user_default.mpfb", default_set=True, persistent=True)
    g = {}
    for name, key in (("services.humanservice","HumanService"),("services.targetservice","TargetService"),
                      ("entities.objectproperties","HumanObjectProperties"),("services.locationservice","LocationService"),
                      ("services.rigservice","RigService"),("services.assetservice","AssetService"),
                      ("services.materialservice","MaterialService"),("services.objectservice","ObjectService")):
        g[key] = getattr(importlib.import_module("bl_ext.user_default.mpfb."+name), key)
    return g
