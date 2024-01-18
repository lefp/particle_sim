#ifndef _GRAPHICS_HPP
#define _GRAPHICS_HPP

// #include <vulkan/vulkan.h>
// #include <glm/glm.hpp>
// #include <imgui/imgui.h>
// #include "types.hpp"

namespace graphics {

using glm::vec2;
using glm::vec3;
using glm::mat4;

//
// ===========================================================================================================
//

const u32 MAX_VOXEL_COUNT = 1'000'000;
const u32 MAX_LINE_SEGMENT_COUNT = 1'000'000;

//
// ===========================================================================================================
//

using VoxelCoord3D = glm::ivec3;

struct LineSegment {
    vec3 start;
    vec3 end;
};
static_assert(alignof(LineSegment) == 4);
static_assert(sizeof(LineSegment) == 4*6);

struct SurfaceResources {
    void* impl;
};
struct RenderResources {
    void* impl;
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

/// Calls `ImGui_ImplVulkan_Init()`. You might need to call `ImGui::CreateContext()` earlier, idk.
bool initImGuiVulkanBackend(void);

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

Result createRenderer(RenderResources* render_resources_out);

/// If `imgui_draw_data` is non-null, calls `ImGui_ImplVulkan_RenderDrawData`.
RenderResult render(
    SurfaceResources surface,
    VkRect2D window_subregion,
    const mat4* world_to_screen_transform,
    const mat4* world_to_screen_transform_inverse,
    ImDrawData* imgui_draw_data,
    u32 voxel_count,
    const VoxelCoord3D* p_voxels,
    u32 line_segment_count,
    const LineSegment* p_line_segments
);

void setVoxels(u32 voxel_count, const VoxelCoord3D* p_voxels);

/// `init()` must have been called before this; otherwise returns VK_NULL_HANDLE.
VkInstance getVkInstance(void);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
