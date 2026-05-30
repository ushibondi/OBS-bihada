--[[
    美肌フィルタ (Bihada Filter) for OBS Studio
    
    Lua script using real-time HLSL custom shaders.
    Optimized for CPU-integrated GPUs (iGPUs) with a 3x3 lightweight bilateral filter
    and a standard 5x5 bilateral filter.
    
    License: MIT
--]]

local obs = obslua
local bit = require("bit")

-- ============================================================================
-- 🌌 HLSL Custom Shader Code (Effect File)
-- ============================================================================
local shader_code = [[
uniform float4x4 ViewProj;
uniform texture2d image;

sampler_state def_sampler {
    Filter   = Linear;
    AddressU = Clamp;
    AddressV = Clamp;
};

// --- uniform parameters ---
uniform float uv_scale_x;
uniform float uv_scale_y;
uniform float smoothing;
uniform float sigma_spatial;
uniform float sigma_range;
uniform float blur_radius;

// Skin detection toggle & parameters (HSV space)
uniform float use_skin_detection; // Float for binding robustness (1.0 = true, 0.0 = false)
uniform float target_hue;
uniform float target_sat;
uniform float target_val;
uniform float hue_tolerance;
uniform float hue_softness;
uniform float sat_min;
uniform float sat_max;
uniform float sat_softness;
uniform float val_min;
uniform float val_softness;

// Brightening & Tinting
uniform float skin_brightness;
uniform float skin_tint_red;
uniform float skin_tint_blue;

uniform float debug_mask; // Float for binding robustness (1.0 = true, 0.0 = false)

struct VertData {
    float4 pos : POSITION;
    float2 uv  : TEXCOORD0;
};

// --- Vertex Shader ---
VertData VSDefault(VertData v_in) {
    VertData vert_out;
    vert_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
    vert_out.uv  = v_in.uv;
    return vert_out;
}

// --- RGB to HSV Helper (Optimized Branchless GPU Formula) ---
float3 rgb2hsv(float3 c) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// --- HSV Skin Mask Detection ---
float get_skin_factor(float3 rgb) {
    float3 hsv = rgb2hsv(rgb);
    
    // Circular Hue difference [0, 1] - Branchless Mathematical Identity
    float hueDiff = abs(hsv.x - target_hue);
    hueDiff = 0.5 - abs(hueDiff - 0.5);
    
    // Hue Soft-thresholding
    float hueFactor = smoothstep(hue_tolerance + hue_softness, hue_tolerance, hueDiff);
    
    // Saturation Soft-thresholding (exclude very low sat like gray, or very high sat)
    float satFactor = smoothstep(sat_min - sat_softness, sat_min, hsv.y) * 
                      smoothstep(sat_max + sat_softness, sat_max, hsv.y);
                      
    // Value (Brightness) Soft-thresholding (exclude dark shadows)
    float valFactor = smoothstep(val_min - val_softness, val_min, hsv.z);
    
    return hueFactor * satFactor * valFactor;
end

// --- 1. Lightweight Pixel Shader (3x3 Kernel - 9 samples for iGPU) ---
float4 PSLightweight(VertData v_in) : TARGET {
    float4 centerColor = image.Sample(def_sampler, v_in.uv);
    
    // Skin factor determination
    float skinFactor = 1.0;
    if (use_skin_detection > 0.5) {
        skinFactor = get_skin_factor(centerColor.rgb);
    }
    
    // Highly stable uniform execution - no early exit branches
    // (Prevents iGPU derivative compiling bugs and flickering)
    float3 blurredColor = 0.0;
    float totalWeight = 0.0;
    float2 uv_scale = float2(uv_scale_x, uv_scale_y);
    
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 offset = float2(x, y) * uv_scale * blur_radius;
            float4 sampleColor = image.Sample(def_sampler, v_in.uv + offset);
            
            // Spatial Weight
            float spatialDist = length(float2(x, y));
            float wSpatial = exp(-(spatialDist * spatialDist) / (2.0 * sigma_spatial * sigma_spatial));
            
            // Range Weight (Color Distance)
            float colorDist = distance(sampleColor.rgb, centerColor.rgb);
            float wRange = exp(-(colorDist * colorDist) / (2.0 * sigma_range * sigma_range));
            
            float weight = wSpatial * wRange;
            blurredColor += sampleColor.rgb * weight;
            totalWeight += weight;
        }
    }
    
    float3 smoothColor = blurredColor / totalWeight;
    
    // Brightness and Tone-up (Skin only)
    float3 skinColor = smoothColor;
    if (skin_brightness > 0.0) {
        skinColor = pow(abs(skinColor), 1.0 - skin_brightness * 0.25);
    }
    
    // Tint adjustments
    skinColor.r += skin_tint_red * 0.04;
    skinColor.b += skin_tint_blue * 0.02;
    skinColor = clamp(skinColor, 0.0, 1.0);
    
    // Interpolate original and beautiful skin
    float3 blendedColor = lerp(centerColor.rgb, skinColor, skinFactor * smoothing);
    
    if (debug_mask > 0.5 && use_skin_detection > 0.5) {
        // Red overlay peaked mask
        return float4(lerp(blendedColor, float3(1.0, 0.0, 0.0), skinFactor * 0.7), centerColor.a);
    }
    
    return float4(blendedColor, centerColor.a);
}

// --- 2. Standard Pixel Shader (5x5 Kernel - 25 samples) ---
float4 PSStandard(VertData v_in) : TARGET {
    float4 centerColor = image.Sample(def_sampler, v_in.uv);
    
    // Skin factor determination
    float skinFactor = 1.0;
    if (use_skin_detection > 0.5) {
        skinFactor = get_skin_factor(centerColor.rgb);
    }
    
    // Highly stable uniform execution - no early exit branches
    // (Prevents iGPU derivative compiling bugs and flickering)
    float3 blurredColor = 0.0;
    float totalWeight = 0.0;
    float2 uv_scale = float2(uv_scale_x, uv_scale_y);
    
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            float2 offset = float2(x, y) * uv_scale * blur_radius;
            float4 sampleColor = image.Sample(def_sampler, v_in.uv + offset);
            
            // Spatial Weight
            float spatialDist = length(float2(x, y));
            float wSpatial = exp(-(spatialDist * spatialDist) / (2.0 * sigma_spatial * sigma_spatial));
            
            // Range Weight (Color Distance)
            float colorDist = distance(sampleColor.rgb, centerColor.rgb);
            float wRange = exp(-(colorDist * colorDist) / (2.0 * sigma_range * sigma_range));
            
            float weight = wSpatial * wRange;
            blurredColor += sampleColor.rgb * weight;
            totalWeight += weight;
        }
    }
    
    float3 smoothColor = blurredColor / totalWeight;
    
    // Brightness and Tone-up (Skin only)
    float3 skinColor = smoothColor;
    if (skin_brightness > 0.0) {
        skinColor = pow(abs(skinColor), 1.0 - skin_brightness * 0.25);
    }
    
    // Tint adjustments
    skinColor.r += skin_tint_red * 0.04;
    skinColor.b += skin_tint_blue * 0.02;
    skinColor = clamp(skinColor, 0.0, 1.0);
    
    // Interpolate original and beautiful skin
    float3 blendedColor = lerp(centerColor.rgb, skinColor, skinFactor * smoothing);
    
    if (debug_mask > 0.5 && use_skin_detection > 0.5) {
        // Red overlay peaked mask
        return float4(lerp(blendedColor, float3(1.0, 0.0, 0.0), skinFactor * 0.7), centerColor.a);
    }
    
    return float4(blendedColor, centerColor.a);
}

// --- Techniques ---
technique DrawLightweight {
    pass {
        vertex_shader = VSDefault(v_in);
        pixel_shader  = PSLightweight(v_in);
    }
}

technique DrawStandard {
    pass {
        vertex_shader = VSDefault(v_in);
        pixel_shader  = PSStandard(v_in);
    }
}
]]

-- ============================================================================
-- 🧮 CPU Mathematical & Conversion Helpers
-- ============================================================================

-- RGB [0, 255] to HSV [0, 1] converter
local function rgb_to_hsv(r, g, b)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    local max_val = math.max(r, g, b)
    local min_val = math.min(r, g, b)
    local delta = max_val - min_val
    
    local h = 0
    local s = 0
    local v = max_val
    
    if max_val ~= 0 then
        s = delta / max_val
    end
    
    if delta ~= 0 then
        if max_val == r then
            h = (g - b) / delta
            if g < b then h = h + 6 end
        elseif max_val == g then
            h = (b - r) / delta + 2
        else
            h = (r - g) / delta + 4
        end
        h = h / 6
    end
    
    return h, s, v
end

-- ============================================================================
-- 🔌 OBS Source Filter Registration API
-- ============================================================================
local source_info = {}
source_info.id = "bihada_filter"
source_info.type = obs.OBS_SOURCE_TYPE_FILTER
source_info.output_flags = obs.OBS_SOURCE_VIDEO

-- Plugin display name
source_info.get_name = function()
    return "美肌フィルタ (Bihada Filter)"
end

-- Plugin filter creation callback
source_info.create = function(settings, source)
    local filter = {}
    filter.context = source

    obs.obs_enter_graphics()
    -- Compile the HLSL effect shader
    filter.effect = obs.gs_effect_create(shader_code, nil, nil)
    if filter.effect ~= nil then
        filter.params = {}
        -- Fetch handles for shader uniforms
        filter.params.uv_scale_x = obs.gs_effect_get_param_by_name(filter.effect, "uv_scale_x")
        filter.params.uv_scale_y = obs.gs_effect_get_param_by_name(filter.effect, "uv_scale_y")
        filter.params.smoothing = obs.gs_effect_get_param_by_name(filter.effect, "smoothing")
        filter.params.sigma_spatial = obs.gs_effect_get_param_by_name(filter.effect, "sigma_spatial")
        filter.params.sigma_range = obs.gs_effect_get_param_by_name(filter.effect, "sigma_range")
        filter.params.blur_radius = obs.gs_effect_get_param_by_name(filter.effect, "blur_radius")
        
        filter.params.use_skin_detection = obs.gs_effect_get_param_by_name(filter.effect, "use_skin_detection")
        filter.params.target_hue = obs.gs_effect_get_param_by_name(filter.effect, "target_hue")
        filter.params.target_sat = obs.gs_effect_get_param_by_name(filter.effect, "target_sat")
        filter.params.target_val = obs.gs_effect_get_param_by_name(filter.effect, "target_val")
        filter.params.hue_tolerance = obs.gs_effect_get_param_by_name(filter.effect, "hue_tolerance")
        filter.params.hue_softness = obs.gs_effect_get_param_by_name(filter.effect, "hue_softness")
        filter.params.sat_min = obs.gs_effect_get_param_by_name(filter.effect, "sat_min")
        filter.params.sat_max = obs.gs_effect_get_param_by_name(filter.effect, "sat_max")
        filter.params.sat_softness = obs.gs_effect_get_param_by_name(filter.effect, "sat_softness")
        filter.params.val_min = obs.gs_effect_get_param_by_name(filter.effect, "val_min")
        filter.params.val_softness = obs.gs_effect_get_param_by_name(filter.effect, "val_softness")
        
        filter.params.skin_brightness = obs.gs_effect_get_param_by_name(filter.effect, "skin_brightness")
        filter.params.skin_tint_red = obs.gs_effect_get_param_by_name(filter.effect, "skin_tint_red")
        filter.params.skin_tint_blue = obs.gs_effect_get_param_by_name(filter.effect, "skin_tint_blue")
        
        filter.params.debug_mask = obs.gs_effect_get_param_by_name(filter.effect, "debug_mask")
    else
        obs.blog(obs.LOG_ERROR, "Bihada Filter: HLSL shader compile failed!")
    end
    obs.obs_leave_graphics()

    -- Initial load of settings
    source_info.update(filter, settings)
    return filter
end

-- Plugin filter destruction callback
source_info.destroy = function(filter)
    if filter.effect ~= nil then
        obs.obs_enter_graphics()
        obs.gs_effect_destroy(filter.effect)
        obs.obs_leave_graphics()
    end
end

-- Plugin filter update callback (called when properties are modified)
source_info.update = function(filter, settings)
    filter.perf_mode = obs.obs_data_get_string(settings, "perf_mode")
    filter.smoothing = obs.obs_data_get_double(settings, "smoothing")
    filter.blur_radius = obs.obs_data_get_double(settings, "blur_radius")
    
    -- Map Edge Protection UI (0.05 to 0.30) to sigma_range
    -- Higher protection means a smaller sigma_range (preserving subtle transitions)
    local edge_prot = obs.obs_data_get_double(settings, "edge_protection")
    
    -- Determine spatial sigma based on performance mode
    if filter.perf_mode == "lightweight" then
        filter.sigma_spatial = 1.0 -- Smaller spatial sigma for 3x3 kernel
    else
        filter.sigma_spatial = 2.0 -- Standard spatial sigma for 5x5 kernel
    end
    filter.sigma_range = 0.35 - edge_prot
    
    filter.brightness = obs.obs_data_get_double(settings, "brightness")
    filter.tint_red = obs.obs_data_get_double(settings, "tint_red")
    filter.tint_blue = obs.obs_data_get_double(settings, "tint_blue")
    
    -- Decode ABGR skin color integer from OBS color picker
    local color = obs.obs_data_get_int(settings, "skin_color")
    local r = bit.band(color, 0xFF)
    local g = bit.band(bit.rshift(color, 8), 0xFF)
    local b = bit.band(bit.rshift(color, 16), 0xFF)
    
    -- Convert picked color to HSV for the shader skin detection
    local h, s, v = rgb_to_hsv(r, g, b)
    filter.target_hue = h
    filter.target_sat = s
    filter.target_val = v
    
    filter.use_skin_detection = obs.obs_data_get_bool(settings, "use_skin_detection")
    filter.hue_tolerance = obs.obs_data_get_double(settings, "color_range")
    filter.hue_softness = obs.obs_data_get_double(settings, "softness")
    filter.sat_min = obs.obs_data_get_double(settings, "min_saturation")
    filter.sat_max = 0.70 -- Constant upper bounds
    filter.sat_softness = 0.05
    filter.val_min = 0.20 -- Threshold out very dark shadows
    filter.val_softness = 0.05
    
    filter.debug_mask = obs.obs_data_get_bool(settings, "debug_mask")
end

-- Plugin filter properties GUI layout
source_info.get_properties = function(filter)
    local props = obs.obs_properties_create()
    
    -- --- System Category ---
    local p_mode = obs.obs_properties_add_list(props, "perf_mode", "動作モード (Performance Mode)", obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING)
    obs.obs_property_list_add_string(p_mode, "軽量モード (Lightweight - iGPU/内蔵GPU向け)", "lightweight")
    obs.obs_property_list_add_string(p_mode, "標準モード (Standard - 高画質/外付けGPU向け)", "standard")
    
    -- --- Beautification Category ---
    obs.obs_properties_add_float_slider(props, "smoothing", "美肌の強度 (Strength)", 0.0, 1.0, 0.05)
    obs.obs_properties_add_float_slider(props, "blur_radius", "ぼかし半径 (Blur Radius)", 1.0, 5.0, 0.1)
    obs.obs_properties_add_float_slider(props, "edge_protection", "輪郭の維持 (Edge Protection)", 0.05, 0.30, 0.01)
    
    -- --- Tone-up & Tint Category ---
    obs.obs_properties_add_float_slider(props, "brightness", "肌の明るさ (Skin Brightness)", 0.0, 1.0, 0.05)
    obs.obs_properties_add_float_slider(props, "tint_red", "血色感・赤味 (Red Tint)", 0.0, 1.0, 0.05)
    obs.obs_properties_add_float_slider(props, "tint_blue", "透明感・青味 (Blue Tint)", 0.0, 1.0, 0.05)
    
    -- --- Skin Detection Category ---
    obs.obs_properties_add_bool(props, "use_skin_detection", "肌色部分のみに適用する (Enable Skin Detection)")
    obs.obs_properties_add_color(props, "skin_color", "基準肌色 (Base Skin Color)")
    obs.obs_properties_add_float_slider(props, "color_range", "検出の広さ (Color Range)", 0.01, 0.15, 0.01)
    obs.obs_properties_add_float_slider(props, "softness", "境界の滑らかさ (Softness)", 0.01, 0.10, 0.01)
    obs.obs_properties_add_float_slider(props, "min_saturation", "彩度の下限 (Min Saturation)", 0.05, 0.30, 0.01)
    
    -- --- Debug Category ---
    obs.obs_properties_add_bool(props, "debug_mask", "肌色マスクのプレビュー (Debug Mask)")
    
    return props
end

-- Plugin defaults
source_info.get_defaults = function(settings)
    obs.obs_data_set_default_string(settings, "perf_mode", "lightweight")
    obs.obs_data_set_default_double(settings, "smoothing", 0.70) -- Raised to 70% by default for more immediate impact
    obs.obs_data_set_default_double(settings, "blur_radius", 2.0)
    obs.obs_data_set_default_double(settings, "edge_protection", 0.15)
    
    obs.obs_data_set_default_double(settings, "brightness", 0.20)
    obs.obs_data_set_default_double(settings, "tint_red", 0.15)
    obs.obs_data_set_default_double(settings, "tint_blue", 0.10)
    
    obs.obs_data_set_default_bool(settings, "use_skin_detection", true)
    -- Default skin color: ABGR 0xFFB4C8F0 which is RGB(240, 200, 180)
    obs.obs_data_set_default_int(settings, "skin_color", 0xFFB4C8F0)
    obs.obs_data_set_default_double(settings, "color_range", 0.08) -- Raised to 0.08 to detect wider skin tones by default
    obs.obs_data_set_default_double(settings, "softness", 0.03)
    obs.obs_data_set_default_double(settings, "min_saturation", 0.10)
    
    obs.obs_data_set_default_bool(settings, "debug_mask", false)
end

-- Plugin filter rendering pipeline
source_info.video_render = function(filter, effect)
    if filter.effect == nil then
        obs.obs_source_skip_video_filter(filter.context)
        return
    end

    -- Get dimensions of the filtered video source
    local target = obs.obs_filter_get_target(filter.context)
    local width = 0
    local height = 0
    if target ~= nil then
        width = obs.obs_source_get_base_width(target)
        height = obs.obs_source_get_base_height(target)
    end
    if width == 0 then width = 1920 end
    if height == 0 then height = 1080 end

    -- Begin drawing pass
    if not obs.obs_source_process_filter_begin(filter.context, obs.GS_RGBA, obs.OBS_NO_DIRECT_RENDERING) then
        return
    end

    obs.obs_enter_graphics()

    -- Feed properties to uniform shader variables
    obs.gs_effect_set_float(filter.params.uv_scale_x, 1.0 / width)
    obs.gs_effect_set_float(filter.params.uv_scale_y, 1.0 / height)
    obs.gs_effect_set_float(filter.params.smoothing, filter.smoothing)
    obs.gs_effect_set_float(filter.params.sigma_spatial, filter.sigma_spatial)
    obs.gs_effect_set_float(filter.params.sigma_range, filter.sigma_range)
    obs.gs_effect_set_float(filter.params.blur_radius, filter.blur_radius)
    
    -- Pass floats instead of booleans for maximum API binding robustness (prevents Lua binding crashes)
    obs.gs_effect_set_float(filter.params.use_skin_detection, filter.use_skin_detection and 1.0 or 0.0)
    obs.gs_effect_set_float(filter.params.target_hue, filter.target_hue)
    obs.gs_effect_set_float(filter.params.target_sat, filter.target_sat)
    obs.gs_effect_set_float(filter.params.target_val, filter.target_val)
    obs.gs_effect_set_float(filter.params.hue_tolerance, filter.hue_tolerance)
    obs.gs_effect_set_float(filter.params.hue_softness, filter.hue_softness)
    obs.gs_effect_set_float(filter.params.sat_min, filter.sat_min)
    obs.gs_effect_set_float(filter.params.sat_max, filter.sat_max)
    obs.gs_effect_set_float(filter.params.sat_softness, filter.sat_softness)
    obs.gs_effect_set_float(filter.params.val_min, filter.val_min)
    obs.gs_effect_set_float(filter.params.val_softness, filter.val_softness)
    
    obs.gs_effect_set_float(filter.params.skin_brightness, filter.brightness)
    obs.gs_effect_set_float(filter.params.skin_tint_red, filter.tint_red)
    obs.gs_effect_set_float(filter.params.skin_tint_blue, filter.tint_blue)
    
    obs.gs_effect_set_float(filter.params.debug_mask, filter.debug_mask and 1.0 or 0.0)

    obs.obs_leave_graphics()

    -- Decide which technique pass to invoke
    local tech_name = "DrawLightweight"
    if filter.perf_mode == "standard" then
        tech_name = "DrawStandard"
    end

    -- End drawing pass using selected technique
    -- (Specifying the exact width and height guarantees no size mismatches or flickering)
    obs.obs_source_process_filter_tech_end(filter.context, filter.effect, width, height, tech_name)
end

-- Register the filter with OBS Studio
obs.obs_register_source(source_info)
