#ifndef _GRAPHICS_HPP
#define _GRAPHICS_HPP

#include <vulkan/vulkan.h>

//
// ===========================================================================================================
//

struct SurfaceResources {
    void* impl;
};

enum class GraphicsResult {
    success,
    error_window_size_zero,
    error_surface_resources_out_of_date,
    success_surface_resources_out_of_date
};


//
// ===========================================================================================================
//

/// If `specific_device_request` isn't NULL, attempts to select a device with that name.
/// If no such device exists or doesn't satisfactory requirements, silently selects a different device.
void init(const char* app_name, const char* specific_named_device_request);

/// The fallback size is used if it fails to determine the window size via Vulkan. I recommend you pass the
/// current window size, as reported by the window library you're using (GLFW, X11, etc).
/// Returns `error_window_size_zero` if the max surface size is found to be zero, or if the fallback size is
/// used and found to be zero.
GraphicsResult createSurfaceResources(
    VkSurfaceKHR surface,
    VkExtent2D fallback_window_size,
    SurfaceResources* surface_resources_out
);
GraphicsResult updateSurfaceResources(SurfaceResources, VkExtent2D fallback_window_size);
void destroySurfaceResources(SurfaceResources);

GraphicsResult render(SurfaceResources);

//
// ===========================================================================================================
//

#endif // include guard
