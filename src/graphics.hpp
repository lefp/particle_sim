#ifndef _GRAPHICS_HPP
#define _GRAPHICS_HPP

// #include <vulkan/vulkan.h>
// #include <glm/glm.hpp>
// #include <imgui/imgui.h>
// #include "types.hpp"

namespace graphics {

using glm::vec2;
using glm::vec3;
using glm::ivec3;
using glm::mat4;

//
// ===========================================================================================================
//

const u32 MAX_VOXEL_COUNT = 1'000'000;
const u32 MAX_OUTLINED_VOXEL_COUNT = 1'000'000;

//
// ===========================================================================================================
//

struct Voxel {
    ivec3 coord;
    // TODO We definitely need no more than u16 for material index; even u8 would probably suffice. Convert
    // this to u8 or u16 if/when we need to make room for more data here.
    u32 material_index;
};
static_assert(alignof(Voxel) == 4);
static_assert(sizeof(Voxel) == 4 * 4);

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
    const Voxel* p_voxels,
    u32 outlined_voxel_index_count,
    const u32* p_outlined_voxel_indices
);

/// `init()` must have been called before this; otherwise returns VK_NULL_HANDLE.
VkInstance getVkInstance(void);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
