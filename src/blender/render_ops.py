import os

# --- Third-party (Blender) ---
import bpy


def set_render_settings(
    width: int,
    height: int,
    engine: str = 'CYCLES',
    quality: str = 'balanced',
    device: str = 'GPU',
):
    """
    Configure render engine and quality-speed tradeoffs.
    engine: "CYCLES" | "BLENDER_EEVEE"
    quality: "draft" | "balanced" | "high"
    device: "CPU" | "GPU"
    """
    scene = bpy.context.scene

    # --- Common resolution & color management ---
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.display_settings.display_device = 'sRGB'
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'

    req_engine = engine.upper().strip()
    req_device = device.upper().strip()

    if req_engine == 'CYCLES':
        scene.render.engine = 'CYCLES'

        # ---- GPU path (robust) ----
        if req_device == 'GPU':
            try:
                prefs = bpy.context.preferences
                cprefs = prefs.addons['cycles'].preferences

                # 1) Pick first available backend
                for backend in ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']:
                    try:
                        cprefs.compute_device_type = backend
                        break
                    except Exception:
                        continue

                # 2) Refresh device list (version-safe)
                try:
                    cprefs.get_devices()
                except Exception:
                    try:
                        cprefs.refresh_devices()
                    except Exception:
                        pass

                # 3) Enable all devices discovered
                enabled_devices = []
                try:
                    for d in cprefs.devices:
                        d.use = True
                        enabled_devices.append(
                            f'{d.type}:{getattr(d, "name", "Unknown")}'
                        )
                except Exception:
                    enabled_devices = []

                # 4) If any GPU-like device is enabled, use GPU; else fallback to CPU
                gpu_enabled = any(
                    x.startswith(('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'))
                    for x in enabled_devices
                )
                if gpu_enabled:
                    scene.cycles.device = 'GPU'
                else:
                    print('[WARN] No usable Cycles GPU devices; falling back to CPU.')
                    scene.cycles.device = 'CPU'
                    # Honest backend state if none usable
                    try:
                        cprefs.compute_device_type = 'NONE'
                    except Exception:
                        pass

            except Exception as e:
                print(f'[WARN] GPU setup failed, falling back to CPU: {e}')
                scene.cycles.device = 'CPU'
                # try to mark backend none for clarity
                try:
                    bpy.context.preferences.addons[
                        'cycles'
                    ].preferences.compute_device_type = 'NONE'
                except Exception:
                    pass

        else:
            # CPU requested
            scene.cycles.device = 'CPU'
            try:
                bpy.context.preferences.addons[
                    'cycles'
                ].preferences.compute_device_type = 'NONE'
            except Exception:
                pass

        # ---- Quality presets ----
        presets = {
            'draft': dict(
                samples=32,
                use_adaptive_sampling=True,
                adaptive_threshold=0.05,
                denoise=True,
                max_bounces=3,
                diffuse_bounces=1,
                glossy_bounces=1,
                transmission_bounces=2,
                transparent_max_bounces=4,
                caustics_reflective=False,
                caustics_refractive=False,
                use_simplify=True,
                simplify_subdivision=0,
                texture_limit='1024',
            ),
            'balanced': dict(
                samples=128,
                use_adaptive_sampling=True,
                adaptive_threshold=0.02,
                denoise=True,
                max_bounces=6,
                diffuse_bounces=2,
                glossy_bounces=2,
                transmission_bounces=4,
                transparent_max_bounces=8,
                caustics_reflective=False,
                caustics_refractive=False,
                use_simplify=False,
                simplify_subdivision=0,
                texture_limit='OFF',
            ),
            'high': dict(
                samples=512,
                use_adaptive_sampling=True,
                adaptive_threshold=0.01,
                denoise=True,
                max_bounces=12,
                diffuse_bounces=4,
                glossy_bounces=4,
                transmission_bounces=6,
                transparent_max_bounces=16,
                caustics_reflective=False,
                caustics_refractive=False,
                use_simplify=False,
                simplify_subdivision=0,
                texture_limit='OFF',
            ),
        }
        p = presets.get(quality, presets['balanced'])

        scene.cycles.samples = p['samples']
        scene.cycles.use_adaptive_sampling = p['use_adaptive_sampling']
        scene.cycles.adaptive_threshold = p['adaptive_threshold']
        scene.cycles.max_bounces = p['max_bounces']
        scene.cycles.diffuse_bounces = p['diffuse_bounces']
        scene.cycles.glossy_bounces = p['glossy_bounces']
        scene.cycles.transmission_bounces = p['transmission_bounces']
        scene.cycles.transparent_max_bounces = p['transparent_max_bounces']
        scene.cycles.caustics_reflective = p['caustics_reflective']
        scene.cycles.caustics_refractive = p['caustics_refractive']

        try:
            scene.view_layers[0].cycles.use_denoising = p['denoise']
        except Exception:
            pass

        scene.render.use_simplify = p['use_simplify']
        scene.render.simplify_subdivision = p['simplify_subdivision']
        try:
            scene.render.simplify_texture_limit = p['texture_limit']
        except Exception:
            pass

    else:
        # EEVEE
        scene.render.engine = 'BLENDER_EEVEE'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'

    # ---- Print final state ----
    print('[INFO] Render engine   :', scene.render.engine)

    if scene.render.engine == 'CYCLES':
        print('[INFO] Cycles device   :', getattr(scene.cycles, 'device', 'UNKNOWN'))
        backend = 'UNKNOWN'
        enabled = []
        try:
            cprefs = bpy.context.preferences.addons['cycles'].preferences
            backend = getattr(cprefs, 'compute_device_type', 'UNKNOWN')
            # refresh list to report accurately
            try:
                cprefs.get_devices()
            except Exception:
                try:
                    cprefs.refresh_devices()
                except Exception:
                    pass
            for d in getattr(cprefs, 'devices', []):
                if getattr(d, 'use', False):
                    enabled.append(f'{d.type}:{getattr(d, "name", "Unknown")}')
        except Exception:
            pass
        print('[INFO] Compute backend :', backend)
        print('[INFO] Enabled devices :', ', '.join(enabled) if enabled else '(none)')
    else:
        print('[INFO] Cycles device   : N/A')
        print('[INFO] Compute backend : N/A')
        print('[INFO] Enabled devices : (none)')


def _setup_depth_exr(depth_exr_path: str) -> None:
    """Configure compositor to write Z to EXR."""
    scene = bpy.context.scene
    scene.use_nodes = True

    # 1) activate z pass first (blender 4.x)
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    bpy.context.view_layer.update()

    # 2) initialize node tree
    ntree = scene.node_tree
    ntree.links.clear()
    ntree.nodes.clear()

    # 3) Create RLayers node
    n_rl = ntree.nodes.new('CompositorNodeRLayers')
    n_rl.location = (-300, 0)

    # If depth socket does not exist, recreate node to force
    if 'Depth' not in [s.name for s in n_rl.outputs]:
        ntree.nodes.remove(n_rl)
        bpy.context.view_layer.update()
        n_rl = ntree.nodes.new('CompositorNodeRLayers')
        n_rl.location = (-300, 0)

    # Remove old output nodes except RLayers
    for n in list(ntree.nodes):
        if n.bl_idname != 'CompositorNodeRLayers':
            ntree.nodes.remove(n)

    # Export EXR (raw meters)
    n_exr = ntree.nodes.new('CompositorNodeOutputFile')
    n_exr.label = 'DepthEXR'
    n_exr.format.file_format = 'OPEN_EXR'
    n_exr.format.color_depth = '32'
    n_exr.base_path = os.path.dirname(depth_exr_path)
    base_exr = os.path.splitext(os.path.basename(depth_exr_path))[0]
    n_exr.file_slots[0].path = base_exr + '_'

    # Link (ensure existence of depth socket now)
    ntree.links.new(n_rl.outputs['Depth'], n_exr.inputs[0])


def render_depth_exr(depth_exr_path: str, cam_obj) -> None:
    """Render depth to an EXR image."""
    scene = bpy.context.scene
    prev_camera = scene.camera
    prev_use_nodes = scene.use_nodes
    prev_filepath = scene.render.filepath
    tmp_path = os.path.join(os.path.dirname(depth_exr_path), '__tmp_depth_main.png')
    try:
        scene.camera = cam_obj
        scene.render.filepath = tmp_path
        _setup_depth_exr(depth_exr_path)
        bpy.ops.render.render(write_still=True)
        finalize_file_output(depth_exr_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f'[WARN] failed to remove tmp: {e}')
        scene.camera = prev_camera
        scene.use_nodes = prev_use_nodes
        scene.render.filepath = prev_filepath


def setup_mask_compositor(mask_png_path: str, object_index: int = 1):
    """Build compositor graph to output an 8-bit single-channel ID mask PNG."""
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nt.links.clear()
    nt.nodes.clear()

    n_rl = nt.nodes.new('CompositorNodeRLayers')
    n_rl.location = (-400, 0)

    # Enable Object Index pass
    view_layer = scene.view_layers[0]
    view_layer.use_pass_object_index = True

    n_id = nt.nodes.new('CompositorNodeIDMask')
    n_id.index = object_index
    n_id.location = (-150, 0)

    n_out = nt.nodes.new('CompositorNodeOutputFile')
    n_out.label = 'ObjMaskPNG'
    n_out.format.file_format = 'PNG'
    n_out.format.color_mode = 'BW'  # single-channel
    n_out.format.color_depth = '8'  # 0 or 255
    n_out.base_path = os.path.dirname(mask_png_path)
    base_png = os.path.splitext(os.path.basename(mask_png_path))[0]
    n_out.file_slots[0].path = base_png + '_'  # will produce <name>_0001.png
    n_out.location = (150, 0)

    # Links: RLayers "IndexOB" -> IDMask -> FileOutput
    nt.links.new(n_rl.outputs['IndexOB'], n_id.inputs['ID value'])
    nt.links.new(n_id.outputs['Alpha'], n_out.inputs[0])

    return n_out


def render_obj_mask(mask_png_path: str, cam_obj, object_index: int = 1):
    """Render an object mask (from depth camera view) to a single-channel PNG."""
    scene = bpy.context.scene
    prev_camera = scene.camera
    prev_use_nodes = scene.use_nodes
    prev_filepath = scene.render.filepath
    tmp_path = os.path.join(os.path.dirname(mask_png_path), '__tmp_mask_main.png')

    try:
        scene.camera = cam_obj
        scene.render.filepath = tmp_path
        setup_mask_compositor(mask_png_path, object_index=object_index)
        bpy.ops.render.render(write_still=True)
        finalize_file_output(mask_png_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f'[WARN] failed to remove tmp main render: {e}')
        scene.camera = prev_camera
        scene.use_nodes = prev_use_nodes
        scene.render.filepath = prev_filepath


def finalize_file_output(target_path: str) -> bool:
    """Rename compositor's <name>_0001.ext to target_path (overwrites if exists)."""
    import shutil

    base_dir = os.path.dirname(target_path)
    base_name = os.path.splitext(os.path.basename(target_path))[0]
    ext = os.path.splitext(target_path)[1]
    src = os.path.join(base_dir, base_name + '_0001' + ext)
    if os.path.exists(src):
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.move(src, target_path)
        return True
    return False


def render_rgb(out_path: str, cam_obj):
    """Render an RGB image to file using the given camera."""
    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
