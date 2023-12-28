#ifndef _GRAPHICS_HPP
#define _GRAPHICS_HPP

// #include <vulkan/vulkan.h>
// #include <glm/glm.hpp>

namespace graphics {

using glm::vec2;
using glm::vec3;
using glm::mat4;

//
// ===========================================================================================================
//

struct SurfaceResources {
    void* impl;
};
struct RenderResources {
    void* impl;
};

struct CameraInfo {
    alignas(16) vec3 camera_direction_unit;
    alignas(16) vec3 camera_right_direction_unit;
    alignas(16) vec3 camera_up_direction_unit;
    alignas(16) vec3 eye_pos;
    alignas(16) vec2 viewport_size;
    alignas(8)  vec2 frustum_near_side_size;
    alignas(8)  float frustum_near_side_distance;
};

enum class [[nodiscard]] Result {
    success,
    error_window_size_zero,
};

enum class [[nodiscard]] RenderResult {
    success,
    error_surface_resources_out_of_date,
    success_surface_resources_out_of_date,
};


//
// ===========================================================================================================
//

/// If `specific_device_request` isn't NULL, attempts to select a device with that name.
/// If no such device exists or no such device satisfies requirements, silently selects a different device.
void init(const char* app_name, const char* specific_named_device_request);

/// The fallback size is used if it fails to determine the window size via Vulkan. I recommend you pass the
/// current window size, as reported by the window library you're using (GLFW, X11, etc).
/// Returns `error_window_size_zero` if the max surface size is found to be zero, or if the fallback size is
/// used and found to be zero.
Result createSurfaceResources(
    VkSurfaceKHR surface,
    VkExtent2D fallback_window_size,
    SurfaceResources* surface_resources_out
);
Result updateSurfaceResources(SurfaceResources, VkExtent2D fallback_window_size);
void destroySurfaceResources(SurfaceResources);

void attachSurfaceToRenderer(SurfaceResources surface, RenderResources renderer);
void detachSurfaceFromRenderer(SurfaceResources surface, RenderResources renderer);

Result createVoxelRenderer(RenderResources* render_resources_out);

RenderResult render(
    SurfaceResources,
    const mat4* world_to_screen_transform,
    const CameraInfo* camera_info
);


/// `init()` must have been called before this; otherwise returns VK_NULL_HANDLE.
VkInstance getVkInstance(void);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
