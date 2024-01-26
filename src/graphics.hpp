#ifndef _GRAPHICS_HPP
#define _GRAPHICS_HPP

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <imgui/imgui.h>
#include "types.hpp"

namespace graphics {

using glm::vec2;
using glm::vec3;
using glm::ivec3;
using glm::u8vec4;
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
    u8vec4 color;
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

enum class [[nodiscard]] ShaderReloadResult {
    success,
    no_shaders_need_reloading,
    error,
};

enum PresentMode {
    PRESENT_MODE_IMMEDIATE = 0,
    PRESENT_MODE_MAILBOX = 1,
    PRESENT_MODE_FIFO = 2,
    PRESENT_MODE_ENUM_COUNT
};
static_assert((int)PRESENT_MODE_IMMEDIATE == (int)VK_PRESENT_MODE_IMMEDIATE_KHR);
static_assert((int)PRESENT_MODE_MAILBOX == (int)VK_PRESENT_MODE_MAILBOX_KHR);
static_assert((int)PRESENT_MODE_FIFO == (int)VK_PRESENT_MODE_FIFO_KHR);

enum PresentModeFlagBits : u8 {
    PRESENT_MODE_IMMEDIATE_BIT = 1 << PRESENT_MODE_IMMEDIATE,
    PRESENT_MODE_MAILBOX_BIT = 1 << PRESENT_MODE_MAILBOX,
    PRESENT_MODE_FIFO_BIT = 1 << PRESENT_MODE_FIFO,
};
using PresentModeFlags = u8;
static inline PresentModeFlagBits PresentModeFlagBits_fromMode(PresentMode mode) {
    return (PresentModeFlagBits)(1 << mode);
}

/// Initialize using the PresentMode enum as an index.
/// Larger number indicates higher priority.
/// 0 means "don't use this present mode in any case".
using PresentModePriorities = u8[PRESENT_MODE_ENUM_COUNT];

//
// ===========================================================================================================
//

/// If `specific_device_request` isn't NULL, attempts to select a device with that name.
/// If no such device exists or no such device satisfies requirements, silently selects a different device.
void init(const char* app_name, const char* specific_named_device_request);

/// Calls `ImGui_ImplVulkan_Init()`. You might need to call `ImGui::CreateContext()` earlier, idk.
bool initImGuiVulkanBackend(void);

/// Not expensive.
PresentModeFlags getSupportedPresentModes(SurfaceResources);

/// `selected_present_mode_out` may be NULL.
/// The fallback size is used if it fails to determine the window size via Vulkan. I recommend you pass the
/// current window size, as reported by the window library you're using (GLFW, X11, etc).
/// Returns `error_window_size_zero` if the max surface size is found to be zero, or if the fallback size is
/// used and found to be zero.
Result createSurfaceResources(
    VkSurfaceKHR surface,
    const PresentModePriorities present_mode_priorities,
    VkExtent2D fallback_window_size,
    SurfaceResources* surface_resources_out,
    PresentMode* selected_present_mode_out
);
Result updateSurfaceResources(
    SurfaceResources surface_resources,
    const PresentModePriorities present_mode_priorities,
    VkExtent2D fallback_window_size,
    PresentMode* selected_present_mode_out
);
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


/// Watch shader source files, taking note when they are modified.
/// Returns false on failure to enable or disable.
[[nodiscard]] bool setShaderSourceFileModificationTracking(bool enable);

/// Shader source tracking must be enabled before running this.
[[nodiscard]] ShaderReloadResult reloadModifiedShaderSourceFiles(RenderResources renderer);

/// This can be run without source-file tracking enabled.
[[nodiscard]] bool reloadAllShaders(RenderResources renderer);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
