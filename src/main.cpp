#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cinttypes>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <loguru/loguru.hpp>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_vulkan.h>
#include <implot/implot.h>
#include <tracy/tracy/Tracy.hpp>
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

#include "types.hpp"
#include "math_util.hpp"
#include "error_util.hpp"
#include "vk_procs.hpp"
#include "vulkan_context.hpp"
#include "graphics.hpp"
#include "alloc_util.hpp"
#include "str_util.hpp"
#include "defer.hpp"
#include "thread_pool.hpp"
#include "sort.hpp"

#include "plugin.hpp"
#include "../plugins_src/fluid_sim/fluid_sim_types.hpp"
#include "../build/A_generatePluginHeaders/fluid_sim/plugin_fluid_sim.hpp"

#include "../build/env_vars.hpp"

#include "main_internal.hpp"

namespace gfx = graphics;

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::dvec2;
using glm::ivec2;
using glm::ivec3;
using glm::u8vec4;

using fluid_sim::FluidSimProcs;

//
// Global constants ==========================================================================================
//

const char* APP_NAME = "an game";

constexpr VkExtent2D DEFAULT_WINDOW_EXTENT { 800, 600 }; // TODO weird default, because everything else is 16:9

constexpr double ASPECT_RATIO_X_OVER_Y = 16.0 / 9.0;
constexpr double ASPECT_RATIO_Y_OVER_X = 1.0 / ASPECT_RATIO_X_OVER_Y;

constexpr double VIEW_FRUSTUM_NEAR_SIDE_DISTANCE = 0.15; // unit: m
constexpr double VIEW_FRUSTUM_FAR_SIDE_DISTANCE = 500.0;

constexpr double FOV_Y = 0.25 * M_PI; // Full angle from top to bottom of the frustum.
static_assert(FOV_Y < M_PI - 1e-5);

// NOTE: as x -> 90deg, tan(x) -> inf. Don't let any FOV come close to 180 deg.
const float VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y = (f32)(VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * 2.0*glm::tan(0.5*FOV_Y));
const float VIEW_FRUSTUM_NEAR_SIDE_SIZE_X = VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y * (f32)ASPECT_RATIO_X_OVER_Y;
const vec2 VIEW_FRUSTUM_NEAR_SIDE_SIZE { VIEW_FRUSTUM_NEAR_SIDE_SIZE_X, VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y };

const f32 VIEW_FRUSTUM_FAR_SIDE_SIZE_X =
    VIEW_FRUSTUM_NEAR_SIDE_SIZE_X / (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE;
const f32 VIEW_FRUSTUM_FAR_SIDE_SIZE_Y =
    VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y / (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE;
const vec2 VIEW_FRUSTUM_FAR_SIDE_SIZE { VIEW_FRUSTUM_FAR_SIDE_SIZE_X, VIEW_FRUSTUM_FAR_SIDE_SIZE_Y };

const u32fast INVALID_VOXEL_IDX = UINT32_MAX;

constexpr f64 FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS = 10.0;
constexpr f64 FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS = 1. / 30.;
constexpr u32fast FRAMETIME_PLOT_MAX_SAMPLE_COUNT =
    (u32fast)(FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS / FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS);

const u8 DEFAULT_PRESENT_MODE_PRIORITIES[3] {
    [gfx::PRESENT_MODE_IMMEDIATE] = 2,
    [gfx::PRESENT_MODE_MAILBOX] = 3,
    [gfx::PRESENT_MODE_FIFO] = 1,
};

//
// Global variables ==========================================================================================
//

vec3 camera_pos_ { 0, 0, 0 };
f32 camera_speed_ = 3.0; // unit: m/s

/// Spherical coordinates.
// TODO do it from Z- instead of from X+, for consistency?
/// .x = rotation in XZ plane. Range [0, 2 pi). 0 = along x-axis, 0.5 pi = along negative z-axis
/// .y = angle from XZ plane. Range [-0.5 pi, 0.5 pi]. 0.5 pi = along y-axis.
vec2 camera_angles_ { 0, 0 };

dvec2 cursor_pos_ { 0, 0 };

ivec2 window_size_ { 0, 0 };
ivec2 window_pos_ { 0, 0 };
VkRect2D window_draw_region_ {};

bool window_or_surface_out_of_date_ = false;

f64 frame_start_time_seconds_ = 0;

bool cursor_visible_ = false;

bool left_alt_is_pressed_ = false;
bool left_ctrl_is_pressed_ = false;
bool left_ctrl_g_is_pressed_ = false;
bool left_ctrl_r_is_pressed_ = false;
bool left_mouse_is_pressed_ = false;
bool right_mouse_is_pressed_ = false;

bool imgui_overlay_visible_ = false;

u32fast voxel_count_ = 0;
gfx::Voxel* p_voxels_ = NULL;

u32fast voxels_in_frustum_count_ = 0;
VoxelPosAndIndex* p_voxels_in_frustum_ = NULL;

u32fast selected_voxel_index_count_ = 0;
u32 p_selected_voxel_indices_[gfx::MAX_OUTLINED_VOXEL_COUNT];

bool selection_active_;
vec2 selection_point1_windowspace_;
vec2 selection_point2_windowspace_;

bool shader_autoreload_enabled_ = true;
bool shader_file_tracking_enabled_ = false;
bool last_shader_reload_failed_ = false;

bool grid_shader_enabled_ = true;

struct FrametimePlot {
    u32fast first_sample_index = 0;
    u32fast sample_count = 0;
    f32 samples_avg_milliseconds[FRAMETIME_PLOT_MAX_SAMPLE_COUNT];
    f32 samples_max_milliseconds[FRAMETIME_PLOT_MAX_SAMPLE_COUNT];

    inline void push(f32 sample_avg_seconds, f32 sample_max_seconds) {
        if (sample_count < FRAMETIME_PLOT_MAX_SAMPLE_COUNT) {
            this->samples_avg_milliseconds[this->sample_count] = sample_avg_seconds * 1000.f;
            this->samples_max_milliseconds[this->sample_count] = sample_max_seconds * 1000.f;
            this->sample_count++;
        }
        else {
            this->samples_avg_milliseconds[first_sample_index] = sample_avg_seconds * 1000.f;
            this->samples_max_milliseconds[first_sample_index] = sample_max_seconds * 1000.f;
            this->first_sample_index = (this->first_sample_index + 1) % FRAMETIME_PLOT_MAX_SAMPLE_COUNT;
        }
    }

    inline void reset(void) {
        this->sample_count = 0;
        this->first_sample_index = 0;
    }
} frametimeplot_samples_scrolling_buffer_;
f64 frametimeplot_last_sample_time_ = 0.0;
u32fast frametimeplot_frames_since_last_sample_ = 0;
f64 frametimeplot_largest_reading_since_last_sample_ = 0;
bool frametimeplot_paused_ = false;

gfx::PresentMode present_mode_ = gfx::PRESENT_MODE_ENUM_COUNT;
gfx::PresentModePriorities present_mode_priorities_ {};


constexpr fluid_sim::SimParameters FLUID_SIM_PARAMS_DEFAULT {
    .rest_particle_density = 1000,
    .rest_particle_interaction_count_approx = 50,
    .spring_stiffness = 0.05f, // TODO FIXME didn't really think about this
};
fluid_sim::SimParameters fluid_sim_params_ = FLUID_SIM_PARAMS_DEFAULT;

bool fluid_sim_paused_ = false;

// TODO maybe the `fluid_sim` namespace should be a subspace of `plugin`?
const FluidSimProcs* fluid_sim_procs_ = NULL;

struct FluidSimPluginVersionUiElement {
    const FluidSimProcs* procs;
    char* user_annotation;

    char radio_button_label[4];
    char textinput_label[6];
    char button_label[7];

    bool hidden;


    static constexpr u32fast USER_ANNOTATION_BUFFER_SIZE = 64;

    static FluidSimPluginVersionUiElement create(u32fast version) {
        FluidSimPluginVersionUiElement element {};

        element.hidden = false;

        element.user_annotation = mallocArray(USER_ANNOTATION_BUFFER_SIZE, char);

        // NOTE If you make the default annotation empty, make sure to write a '\0' to the first byte.
        const char* default_annotation = "no annotation";
        assert(strlen(default_annotation) < USER_ANNOTATION_BUFFER_SIZE);
        strcpy(element.user_annotation, default_annotation);


        // Within a window, for any given widget type,
        //     ImGui requires each instance of that widget type to have a unique label.
        alwaysAssert(version <= 10 * (sizeof(radio_button_label)-1) - 1); // e.g. (sizeof == 4) -> (v <= 999)
        {
            int sz = snprintf(
                element.radio_button_label, sizeof(element.radio_button_label), "%" PRIuFAST32, version
            );
            alwaysAssert(sz > 0);
            assert((size_t)sz < sizeof(element.radio_button_label));


            assert(sizeof(element.textinput_label) >= 2 + sizeof(element.radio_button_label));

            char* ptr = element.textinput_label;
            *ptr = '#';
            ptr++;
            *ptr = '#';
            ptr++;
            strcpy(ptr, element.radio_button_label);


            assert(sizeof(element.button_label) >= 1 + sizeof(element.textinput_label));

            ptr = element.button_label;
            *ptr = 'X';
            ptr++;
            strcpy(ptr, element.textinput_label);
        }


        return element;
    }
};

static struct {
    ArrayList<const FluidSimProcs*> procs = ArrayList<const FluidSimProcs*>::create();
    ArrayList<FluidSimPluginVersionUiElement> ui_elements = ArrayList<FluidSimPluginVersionUiElement>::create();
    u32fast hidden_ui_element_count = 0;

    void push(const FluidSimProcs* new_procs) {

        u32fast version = this->procs.size;
        auto new_ui_element = FluidSimPluginVersionUiElement::create(version);

        this->procs.push(new_procs);
        this->ui_elements.push(new_ui_element);
        assert(this->procs.size == this->ui_elements.size);
    }
} fluid_sim_plugin_versions_;

u32fast fluid_sim_selected_plugin_version_ = 0;
bool fluid_sim_plugin_last_reload_failed_ = false;
bool fluid_sim_plugin_filewatch_enabled_ = false;
bool fluid_sim_plugin_autoreload_enabled_ = false;


// TODO FIXME I fucking hate this. It will lead to bugs. We need a proper GPU dependency graph/queue system.
VkSemaphore render_finished_semaphore_ = VK_NULL_HANDLE;
bool render_finished_semaphore_will_be_signalled_ = false;
VkSemaphore sim_finished_semaphore_ = VK_NULL_HANDLE;
bool sim_finished_semaphore_will_be_signalled_ = false;

VkFence general_purpose_fence_ = VK_NULL_HANDLE;


thread_pool::ThreadPool* thread_pool_ = NULL;


//
// ===========================================================================================================
//

static void _assertGlfw(bool condition, const char* file, int line) {

    if (condition) return;


    const char* err_description = NULL;
    int err_code = glfwGetError(&err_description);
    if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";

    ABORT_F(
        "Assertion failed! GLFW error code %i, file `%s`, line %i, description `%s`",
        err_code, file, line, err_description
    );
};
#define assertGlfw(condition) _assertGlfw(condition, __FILE__, __LINE__)


static void _abortIfGlfwError(const char* file, int line) {

    const char* err_description = NULL;
    int err_id = glfwGetError(&err_description);
    if (err_id == GLFW_NO_ERROR) return;

    if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";
    ABORT_F(
        "Aborting due to GLFW error code %i, file `%s`, line %i, description `%s`",
        err_id, file, line, err_description
    );
};
#define abortIfGlfwError() _abortIfGlfwError(__FILE__, __LINE__)


static void _assertVk(VkResult result, const char* file, int line) {

    if (result == VK_SUCCESS) return;

    LOG_F(
        FATAL, "VkResult is %i, file `%s`, line %i",
        result, file, line
    );
    abort();
}
#define assertVk(result) _assertVk(result, __FILE__, __LINE__)


static void _assertGraphics(gfx::Result result, const char* file, int line) {

    if (result == gfx::Result::success) return;

    ABORT_F("GraphicsResult is %i, file `%s`, line %i", (int)result, file, line);
}
#define assertGraphics(result) _assertGraphics(result, __FILE__, __LINE__)


static void checkedGlfwGetCursorPos(GLFWwindow* window, double* x_out, double* y_out) {

    glfwGetCursorPos(window, x_out, y_out);


    const char* err_description = NULL;
    int err_code = glfwGetError(&err_description);
    if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";

    if (err_code == GLFW_NO_ERROR) {}
    // TODO maybe downgrade this from ERROR to INFO, if it's an expected occurence and is fine.
    else if (err_code == GLFW_PLATFORM_ERROR) LOG_F(ERROR, "Failed to get cursor position: GLFW_PLATFORM_ERROR");
    else ABORT_F("GLFW error %i, description: `%s`.", err_code, err_description);
}


// The fence must be in the unsignalled state.
// This procedure will signal, wait for, and reset the fence before returning.
static void clearSemaphore(const VulkanContext* vk_ctx, VkSemaphore sem, VkFence fence) {

    assert(sem != VK_NULL_HANDLE);

    VkResult result = VK_ERROR_UNKNOWN;


    VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &sem,
        .pWaitDstStageMask = &wait_dst_stage_mask,
    };
    result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, fence);
    assertVk(result);


    result = vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &fence, true, UINT64_MAX);
    assertVk(result);

    result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &fence);
    assertVk(result);
}


static inline vec2 flip_screenXY_to_cameraXY(vec2 screen_coords) {
    return vec2(screen_coords.x, -screen_coords.y);
}


static inline vec2 windowspaceToNormalizedScreenspace(vec2 p, const VkRect2D* viewport) {
    return
        (p - vec2 { viewport->offset.x, viewport->offset.y })
        / vec2 { viewport->extent.width, viewport->extent.height }
        * 2.0f - 1.0f;
}


/// Returns a 16:9 subregion centered in an image, which maximizes the subregion's area.
static VkRect2D centeredSubregion_16x9(u32 image_width, u32 image_height) {

    const bool limiting_dim_is_x = image_width * 9 <= image_height * 16;

    VkOffset2D offset {};
    VkExtent2D extent {};
    if (limiting_dim_is_x) {
        extent.width = image_width;
        extent.height = image_width * 9 / 16;
        offset.x = 0;
        offset.y = ((i32)image_height - (i32)extent.height) / 2;
    }
    else {
        extent.height = image_height;
        extent.width = image_height * 16 / 9;
        offset.y = 0;
        offset.x = ((i32)image_width - (i32)extent.width) / 2;
    }

    assert(offset.x >= 0);
    assert(offset.y >= 0);

    return VkRect2D {
        .offset = offset,
        .extent = extent,
    };
}


static inline vec3 indexspaceToWorldspace(ivec3 idx) {
    return vec3(idx) * gfx::VOXEL_DIAMETER;
}


static inline vec3 worldspaceToIndexspaceFloat(vec3 pos) {
    return pos * (1.f / gfx::VOXEL_DIAMETER);
}


static inline ivec3 worldspaceToIndexspaceInt(vec3 pos) {
    vec3 indexspace_float = worldspaceToIndexspaceFloat(pos);
    return ivec3(glm::round(indexspace_float) + 0.1f); // + 0.1 to avoid truncation of e.g. 2.999 to 2.0
}


struct AxisAlignedBox {
    f32 x_min;
    f32 y_min;
    f32 z_min;
    f32 x_max;
    f32 y_max;
    f32 z_max;
};
struct RaycastRay {
    vec3 origin;
    vec3 direction_reciprocal;
};
// TODO OPTIMIZE
// src: https://tavianator.com/2022/ray_box_boundary.html
/// Returns a number <= 0 if there is no collision.
/// TODO FIXME:
/// 1. Doesn't handle the case where the ray is parallel to an axis.
/// 2. Don't know if it handles the case where the ray origin is inside the box.
static inline f32 rayBoxInteriorCollisionTime(const RaycastRay* ray, const AxisAlignedBox* box) {

    f32 t_x0 = (box->x_min - ray->origin.x) * ray->direction_reciprocal.x;
    f32 t_y0 = (box->y_min - ray->origin.y) * ray->direction_reciprocal.y;
    f32 t_z0 = (box->z_min - ray->origin.z) * ray->direction_reciprocal.z;

    f32 t_x1 = (box->x_max - ray->origin.x) * ray->direction_reciprocal.x;
    f32 t_y1 = (box->y_max - ray->origin.y) * ray->direction_reciprocal.y;
    f32 t_z1 = (box->z_max - ray->origin.z) * ray->direction_reciprocal.z;


    f32 t_min_x = glm::min(t_x0, t_x1);
    f32 t_max_x = glm::max(t_x0, t_x1);

    f32 t_min_y = glm::min(t_y0, t_y1);
    f32 t_max_y = glm::max(t_y0, t_y1);

    f32 t_min_z = glm::min(t_z0, t_z1);
    f32 t_max_z = glm::max(t_z0, t_z1);


    f32 t_entry = glm::max(t_min_x, t_min_y, t_min_z);
    f32 t_exit  = glm::min(t_max_x, t_max_y, t_max_z);
    f32 time_spent_in_box = t_exit - t_entry;
    bool box_entered = time_spent_in_box > 0.0f;
    return (f32)box_entered * t_entry;
}


/// Returns the index of the voxel with the earliest collision, or INVALID_VOXEL_IDX if there are no collisions.
static u32fast rayCast(
    vec3 ray_origin,
    vec3 ray_direction,
    u32fast voxel_count,
    const VoxelPosAndIndex* p_voxels
) {
    ZoneScoped;

    u32fast earliest_collision_idx = INVALID_VOXEL_IDX;
    f32 earliest_collision_time = INFINITY;

    RaycastRay ray {
        .origin = worldspaceToIndexspaceFloat(ray_origin),
        .direction_reciprocal = 1.f / ray_direction,
    };

    for (u32fast voxel_idx = 0; voxel_idx < voxel_count; voxel_idx++) {

        vec3 voxel_coord = p_voxels[voxel_idx].pos;
        AxisAlignedBox box {
            .x_min = (f32)voxel_coord.x - 0.5f,
            .y_min = (f32)voxel_coord.y - 0.5f,
            .z_min = (f32)voxel_coord.z - 0.5f,
            .x_max = (f32)voxel_coord.x + 0.5f,
            .y_max = (f32)voxel_coord.y + 0.5f,
            .z_max = (f32)voxel_coord.z + 0.5f,
        };

        f32 t = rayBoxInteriorCollisionTime(&ray, &box);
        if (0.0f < t and t < earliest_collision_time) {
            earliest_collision_time = t;
            earliest_collision_idx = p_voxels[voxel_idx].idx;
        }
    }

    return earliest_collision_idx;
}


struct Hexahedron {
    // TODO Rename the member variables; they were originally named with a frustum in mind, but a hexahedron
    // is not necessarily a frustum.

    vec3 near_bot_left_p;
    vec3 far_top_right_p;

    vec3 near_normal;
    vec3 bot_normal;
    vec3 left_normal;

    vec3 far_normal;
    vec3 top_normal;
    vec3 right_normal;
};


/// p1 and p2 must be in normalized screenspace;
///     i.e. the top-left of the screen is [-1, -1], and the bottom-right is [1, 1].
static Hexahedron frustumFromScreenspacePoints(
    vec3 camera_pos,
    vec3 camera_direction_unit,
    vec3 camera_horizontal_right_direction_unit,
    vec3 camera_relative_up_direction_unit,
    vec2 p1,
    vec2 p2
) {

    p1 = flip_screenXY_to_cameraXY(p1);
    p2 = flip_screenXY_to_cameraXY(p2);

    vec2 min = glm::min(p1, p2);
    vec2 max = glm::max(p1, p2);

    vec3 near_bot_left_p = camera_pos
        + (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * camera_direction_unit
        + min.x * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + min.y * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 near_top_left_p = camera_pos
        + (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * camera_direction_unit
        + min.x * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + max.y * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 near_bot_right_p = camera_pos
        + (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * camera_direction_unit
        + max.x * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + min.y * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 near_top_right_p = camera_pos
        + (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * camera_direction_unit
        + max.x * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + max.y * 0.5f * (f32)VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 far_bot_left_p = camera_pos
        + (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE * camera_direction_unit
        + min.x * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + min.y * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 far_top_left_p = camera_pos
        + (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE * camera_direction_unit
        + min.x * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + max.y * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 far_bot_right_p = camera_pos
        + (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE * camera_direction_unit
        + max.x * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + min.y * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    vec3 far_top_right_p = camera_pos
        + (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE * camera_direction_unit
        + max.x * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_X * camera_horizontal_right_direction_unit
        + max.y * 0.5f * (f32)VIEW_FRUSTUM_FAR_SIDE_SIZE_Y * camera_relative_up_direction_unit;

    return Hexahedron {
        .near_bot_left_p = near_bot_left_p,
        .far_top_right_p = far_top_right_p,

        .near_normal = camera_direction_unit,
        .bot_normal = glm::normalize(glm::cross(far_bot_right_p - near_bot_right_p, near_bot_left_p - near_bot_right_p)),
        .left_normal = glm::normalize(glm::cross(far_bot_left_p - near_bot_left_p, near_top_left_p - near_bot_left_p)),
        .far_normal = -camera_direction_unit,
        .top_normal = glm::normalize(glm::cross(far_top_left_p - near_top_left_p, near_top_right_p - near_top_left_p)),
        .right_normal = glm::normalize(glm::cross(far_top_right_p - near_top_right_p, near_bot_right_p - near_top_right_p)),
    };
};


static inline bool pointIsInHexahedron(const Hexahedron* f, vec3 p) {

    bool inside = true;

    inside &= glm::dot(p - f->near_bot_left_p, f->near_normal) > 0.0f;
    inside &= glm::dot(p - f->near_bot_left_p, f->bot_normal) > 0.0f;
    inside &= glm::dot(p - f->near_bot_left_p, f->left_normal) > 0.0f;

    inside &= glm::dot(p - f->far_top_right_p, f->far_normal) > 0.0f;
    inside &= glm::dot(p - f->far_top_right_p, f->top_normal) > 0.0f;
    inside &= glm::dot(p - f->far_top_right_p, f->right_normal) > 0.0f;

    return inside;
}


/// The points in `frustum` must be in index space.
/// The normals in `frustum` must be unit vectors.
/// Returns the number of points remaining after culling.
static u32fast frustumCull(
    const Hexahedron* frustum,
    u32fast voxel_count,
    const gfx::Voxel* p_voxels,
    VoxelPosAndIndex* p_voxels_out
) {
    ZoneScoped;

    assert(glm::abs(1.f - glm::length(frustum->near_normal)) < 1e-5);
    assert(glm::abs(1.f - glm::length(frustum->far_normal)) < 1e-5);
    assert(glm::abs(1.f - glm::length(frustum->bot_normal)) < 1e-5);
    assert(glm::abs(1.f - glm::length(frustum->top_normal)) < 1e-5);
    assert(glm::abs(1.f - glm::length(frustum->left_normal)) < 1e-5);
    assert(glm::abs(1.f - glm::length(frustum->right_normal)) < 1e-5);

    constexpr f32 voxel_bounding_sphere_radius = 0.707106781186548f + 1e-5f; // sqrt(0.5*0.5 + 0.5*0.5)

    u32fast voxel_out_idx = 0;
    for (u32fast voxel_idx = 0; voxel_idx < voxel_count; voxel_idx++) {

        ivec3 voxel_coord_int = p_voxels[voxel_idx].coord;
        vec3 p = vec3(voxel_coord_int);

        f32 signed_distance = INFINITY;

        signed_distance = glm::min(signed_distance, glm::dot(p - frustum->near_bot_left_p, frustum->near_normal));
        signed_distance = glm::min(signed_distance, glm::dot(p - frustum->near_bot_left_p, frustum->bot_normal));
        signed_distance = glm::min(signed_distance, glm::dot(p - frustum->near_bot_left_p, frustum->left_normal));

        signed_distance = glm::min(signed_distance, glm::dot(p - frustum->far_top_right_p, frustum->far_normal));
        signed_distance = glm::min(signed_distance, glm::dot(p - frustum->far_top_right_p, frustum->top_normal));
        signed_distance = glm::min(signed_distance, glm::dot(p - frustum->far_top_right_p, frustum->right_normal));

        if (signed_distance >= -voxel_bounding_sphere_radius) {
            p_voxels_out[voxel_out_idx] = VoxelPosAndIndex {
                .pos = voxel_coord_int,
                .idx = (u32)voxel_idx,
            };
            voxel_out_idx++;
        }
    }

    return voxel_out_idx;
}


static fluid_sim::SimData initFluidSim(const fluid_sim::SimParameters* params) {

    // OPTIMIZE if needed. This was written without much thought.

    srand(2039519);

    fluid_sim::SimData sim_data {};
    {
        u32fast particle_count = 100000;

        vec4* p_initial_particles = callocArray(particle_count, vec4);
        defer(free(p_initial_particles));

        for (u32fast particle_idx = 0; particle_idx < particle_count; particle_idx++) {

            vec3 random_0_to_1 {
                (f32)rand() / (f32)RAND_MAX,
                (f32)rand() / (f32)RAND_MAX,
                (f32)rand() / (f32)RAND_MAX,
            };

            u8vec4 color = u8vec4(
                0.f, 50.f + 150.f * ((f32)particle_idx / (f32)particle_count), 255.f,
                255.f
            );

            *(vec3*)(&p_initial_particles[particle_idx]) = (random_0_to_1 - 0.5f) * 5.0f;

            // TODO FIXME: This relies on the fact that the fluid simulator doesn't modify the w component;
            // but the fluid simulator makes no such guarantee.
            p_initial_particles[particle_idx].w = *(f32*)(&color);
        }

        sim_data = fluid_sim_procs_->create(
            params,
            gfx::getVkContext(),
            particle_count,
            p_initial_particles
        );
    }

    return sim_data;
}

static void updateFluidSimPluginVersionAndProcs(const FluidSimProcs* new_procs) {
    assert(new_procs != NULL);

    u32fast new_version = fluid_sim_plugin_versions_.procs.size;
    assert(new_version == plugin::getLatestVersionNumber(PluginID_FluidSim));

    fluid_sim_plugin_versions_.push(new_procs);
    fluid_sim_procs_ = new_procs;
    fluid_sim_selected_plugin_version_ = new_version;

    LOG_F(
        INFO, "Newly loaded fluid sim plugin version is %" PRIuFAST32 ".",
        fluid_sim_selected_plugin_version_
    );
}

//
// ImGui windows =============================================================================================
//

static inline int guiGetCommonWindowFlags(void) {
    int flags = ImGuiWindowFlags_NoFocusOnAppearing;
    if (!cursor_visible_) flags |= ImGuiWindowFlags_NoInputs;
    return flags;
}

static void guiWindow_camera(vec3* p_camera_pos, vec2* p_camera_angles, f32* p_camera_speed) {

    int window_flags = guiGetCommonWindowFlags() | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::Begin("Camera", NULL, window_flags);
    defer(ImGui::End());


    f32 user_pos_input[3] { p_camera_pos->x, p_camera_pos->y, p_camera_pos->z };
    if (ImGui::DragFloat3("Position", user_pos_input, 0.1f, 0.0, 0.0, "%.1f")) {
        p_camera_pos->x = user_pos_input[0];
        p_camera_pos->y = user_pos_input[1];
        p_camera_pos->z = user_pos_input[2];
    };

    ImGui::SliderAngle("Rotation X", &p_camera_angles->x, 0.0, 360.0);
    ImGui::SliderAngle("Rotation Y", &p_camera_angles->y, -90.0, 90.0);

    ImGui::DragFloat("Movement speed", p_camera_speed, 1.0f, 0.0f, FLT_MAX / (f32)INT_MAX);
}

static void guiWindow_selection(void) {

    int window_flags = guiGetCommonWindowFlags() | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::Begin("Selection", NULL, window_flags);
    defer(ImGui::End());


    ImGui::Text("Selected voxels: %" PRIuFAST32, selected_voxel_index_count_);
}

struct GuiWindowGraphicsResult {
    bool button_pressed_reload_all_shaders;
};
[[nodiscard]] static GuiWindowGraphicsResult guiWindow_graphics(
    const bool last_shader_reload_failed,
    const bool shader_file_tracking_enabled,
    bool* p_shader_autoreload_enabled,
    bool* p_grid_shader_enabled,
    const gfx::PresentModeFlags supported_present_modes,
    gfx::PresentMode* p_selected_present_mode
) {

    int window_flags = guiGetCommonWindowFlags() | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::Begin("Graphics", NULL, window_flags);
    defer(ImGui::End());


    GuiWindowGraphicsResult ret {};

    ImGui::SeparatorText("Shaders");
    {
        ImGui::Text("Last reload:");
        ImGui::SameLine();
        if (last_shader_reload_failed) ImGui::TextColored(ImVec4 { 1., 0., 0., 1. }, "failed");
        else ImGui::TextColored(ImVec4 { 0., 1., 0., 1.}, "success");

        ret.button_pressed_reload_all_shaders = ImGui::Button("Reload all");

        if (shader_file_tracking_enabled) ImGui::Checkbox("Auto-reload", p_shader_autoreload_enabled);
        else {
            ImGui::BeginDisabled();
            ImGui::Checkbox("Auto-reload (unavailable)", p_shader_autoreload_enabled);
            ImGui::EndDisabled();
        }

        ImGui::Checkbox("Grid", p_grid_shader_enabled);
    }
    ImGui::SeparatorText("Present mode");
    {
        int selected_present_mode = (int)*p_selected_present_mode;
        {
            ImGui::BeginDisabled(!(supported_present_modes & gfx::PRESENT_MODE_MAILBOX_BIT));
            ImGui::RadioButton(
                "Mailbox", &selected_present_mode, gfx::PRESENT_MODE_MAILBOX
            );
            ImGui::EndDisabled();

            ImGui::BeginDisabled(!(supported_present_modes & gfx::PRESENT_MODE_FIFO_BIT));
            ImGui::RadioButton(
                "FIFO", &selected_present_mode, gfx::PRESENT_MODE_FIFO
            );
            ImGui::EndDisabled();

            ImGui::BeginDisabled(!(supported_present_modes & gfx::PRESENT_MODE_IMMEDIATE_BIT));
            ImGui::RadioButton(
                "Immediate", &selected_present_mode, gfx::PRESENT_MODE_IMMEDIATE
            );
            ImGui::EndDisabled();
        }
        *p_selected_present_mode = (gfx::PresentMode)selected_present_mode;
    }

    return ret;
}

static void guiWindow_performance(
    const char* axis_label,
    const FrametimePlot* frametime_plot_data,
    bool* p_plot_paused
) {

    int window_flags = guiGetCommonWindowFlags();
    ImGui::Begin("Performance", NULL, window_flags);
    defer(ImGui::End());

    ImGui::Checkbox("Pause plot", p_plot_paused);

    if (ImPlot::BeginPlot(axis_label, ImVec2(-1,-1))) {
        defer(ImPlot::EndPlot());

        ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_Lock);
        ImPlot::SetupAxisLimits(ImAxis_X1, -FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS, 0.0);
        ImPlot::SetupAxisFormat(ImAxis_X1, "%.0fs");

        ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_LockMin);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 3.0);

        ImPlot::PlotShaded<f32>(
            "Avg",
            (const f32*)&frametime_plot_data->samples_avg_milliseconds,
            (int)frametime_plot_data->sample_count,
            0.0, // yref
            FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS, // xscale
            -FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS, // xstart
            ImPlotShadedFlags_None, // flags
            (int)frametime_plot_data->first_sample_index // offset
        );

        ImPlot::PlotLine<f32>(
            "Max",
            (const f32*)&frametime_plot_data->samples_max_milliseconds,
            (int)frametime_plot_data->sample_count,
            FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS, // xscale
            -FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS, // xstart
            ImPlotShadedFlags_None, // flags
            (int)frametime_plot_data->first_sample_index // offset
        );
    }
}

struct GuiWindowFluidSimResult {
    bool sim_params_modified;
    bool button_pressed_reset_state;
    bool button_pressed_reload;
};
[[nodiscard]] static GuiWindowFluidSimResult guiWindow_fluidSim(
    const bool last_sim_plugin_reload_failed,
    const bool filewatch_enabled,
    bool* p_sim_paused,
    bool* p_autoreload_enabled,
    ArrayList<FluidSimPluginVersionUiElement>* p_ui_elements,
    u32fast *const p_hidden_ui_element_count,
    u32fast *const p_selected_plugin_version,
    fluid_sim::SimParameters* p_sim_params
) {

    int window_flags = guiGetCommonWindowFlags() | ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::Begin("Fluid sim", NULL, window_flags);
    defer(ImGui::End());


    GuiWindowFluidSimResult ret {};

    {
        const char* pause_button_label = *p_sim_paused ? "Resume" : "Pause";
        if (ImGui::Button(pause_button_label)) *p_sim_paused = !*p_sim_paused;

        ret.button_pressed_reset_state = ImGui::Button("Reset state");
    }
    ImGui::SeparatorText("Parameters");
    {
        bool params_modified = false;

        if (ImGui::Button("Reset params")) {
            params_modified = true;
            *p_sim_params = FLUID_SIM_PARAMS_DEFAULT;
        }

        params_modified |= ImGui::DragFloat("Rest particle density", &p_sim_params->rest_particle_density, 10.0f, 1.0f, FLT_MAX / (f32)INT_MAX);
        params_modified |= ImGui::DragFloat("Rest interaction count", &p_sim_params->rest_particle_interaction_count_approx, 2.0f, 1.0f, FLT_MAX / (f32)INT_MAX);
        params_modified |= ImGui::DragFloat("Spring stiffness", &p_sim_params->spring_stiffness, 0.01f, 0.0f, FLT_MAX / (f32)INT_MAX);

        ret.sim_params_modified = params_modified;
    }
    ImGui::SeparatorText("Plugin");
    {
        ImGui::Text("Last reload:");
        ImGui::SameLine();
        if (last_sim_plugin_reload_failed) ImGui::TextColored(ImVec4 { 1., 0., 0., 1. }, "failed");
        else ImGui::TextColored(ImVec4 { 0., 1., 0., 1.}, "success");


        ret.button_pressed_reload = ImGui::Button("Reload");

        if (filewatch_enabled) {
            ImGui::Checkbox("Auto-reload", p_autoreload_enabled);
        }
        else {
            ImGui::BeginDisabled();
            ImGui::Checkbox("Auto-reload (unavailable)", p_autoreload_enabled);
            ImGui::EndDisabled();
        }


        ImGui::Text("Versions");

        for (u32fast version = 0; version < p_ui_elements->size; version++) {

            FluidSimPluginVersionUiElement* p_ui_element = &p_ui_elements->ptr[version];
            if (p_ui_element->hidden) continue;

            int selected_plugin_version = (int)*p_selected_plugin_version;
            ImGui::RadioButton(p_ui_element->radio_button_label, &selected_plugin_version, (int)version);
            *p_selected_plugin_version = (u32fast)selected_plugin_version;

            ImGui::SameLine();
            ImGui::InputText(
                p_ui_element->textinput_label,
                p_ui_element->user_annotation,
                p_ui_element->USER_ANNOTATION_BUFFER_SIZE
            );

            if (p_ui_elements->size - *p_hidden_ui_element_count > 1) {
                ImGui::SameLine();
                if (ImGui::Button(p_ui_element->button_label)) {
                    p_ui_element->hidden = true;
                    (*p_hidden_ui_element_count)++;
                }
            }
        }
    }

    return ret;
}

static thread_pool::ThreadPool* createThreadPool(void)
{
    long processor_count = sysconf(_SC_NPROCESSORS_ONLN);

    if (processor_count < 0)
    {
        const int err = errno;
        const char* err_description = strerror(err);
        if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";

        LOG_F(
            ERROR, "Failed to get processor count; (errno %i, description `%s`).",
            err, err_description
        );
        processor_count = 1;
    }
    else if (processor_count == 0)
    {
        LOG_F(ERROR, "Failed to get processor count (got 0).");
        processor_count = 1;
    }

    LOG_F(INFO, "Using processor_count=%li for thread pool.", processor_count);

    // TODO FIXME max_queue_size chosen arbitrarily. Should probably make it growable so you don't have to
    //     worry about overflowing it.
    return thread_pool::create((u32)processor_count, 100);
};

//
// ===========================================================================================================
//

int main(int argc, char** argv) {

    ZoneScoped;

    loguru::init(argc, argv);
    #ifndef NDEBUG
        LOG_F(INFO, "NDEBUG is not defined.");
    #else
        LOG_F(INFO, "NDEBUG is defined.");
    #endif

    // Set the environment variables corresponding to the build configuration.
    // This is partly for general consistency (have the same config vars set during build and runtime),
    // but the main reason is that we must use the same configuration when calling build scripts when
    // hot-reloading.
    // I do not like this complexity, it feels fragile and easy to fuck up.
    for (u32fast i = 0; i < ANGAME_ENV_VAR_COUNT; i++) {
        const char* name = ANGAME_ENV_NAMES[i];
        const char* value = ANGAME_ENV_VALUES[i];
        LOG_F(INFO, "Setting environment variable `%s` to `%s`.", name, value);
        setenv(name, value, 1);
    }


    thread_pool_ = createThreadPool();
    alwaysAssert(thread_pool_ != NULL);


    int success = glfwInit();
    assertGlfw(success);


    const char* specific_device_request = getenv("PHYSICAL_DEVICE_NAME"); // can be NULL
    gfx::init(APP_NAME, specific_device_request);

    success = gfx::setShaderSourceFileModificationTracking(true);
    shader_file_tracking_enabled_ = success;
    if (!success) {
        LOG_F(ERROR, "Failed to enable shader source file tracking.");
        shader_autoreload_enabled_ = false;
    }

    gfx::setGridEnabled(grid_shader_enabled_);


    {
        VkResult result = VK_ERROR_UNKNOWN;
        const VulkanContext* vk_ctx = gfx::getVkContext();


        VkFenceCreateInfo fence_info { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        result = vk_ctx->procs_dev.CreateFence(vk_ctx->device, &fence_info, NULL, &general_purpose_fence_);
        assertVk(result);


        VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

        result = vk_ctx->procs_dev.CreateSemaphore(
            vk_ctx->device, &semaphore_info, NULL, &sim_finished_semaphore_
        );
        assertVk(result);

        result = vk_ctx->procs_dev.CreateSemaphore(
            vk_ctx->device, &semaphore_info, NULL, &render_finished_semaphore_
        );
        assertVk(result);


        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished_semaphore_,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, general_purpose_fence_);
        assertVk(result);
        render_finished_semaphore_will_be_signalled_ = true;

        result = vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &general_purpose_fence_, true, UINT64_MAX);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &general_purpose_fence_);
        assertVk(result);
    }


    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // don't initialize OpenGL, because we're using Vulkan
    GLFWwindow* window = glfwCreateWindow(
        (int)DEFAULT_WINDOW_EXTENT.width, (int)DEFAULT_WINDOW_EXTENT.height, "an game", NULL, NULL
    );
    assertGlfw(window != NULL);


    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    abortIfGlfwError();

    bool raw_mouse_motion_supported = glfwRawMouseMotionSupported();
    abortIfGlfwError();

    if (!raw_mouse_motion_supported) ABORT_F("GLFW claims that raw mouse motion is unsupported.");
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    abortIfGlfwError();

    cursor_visible_ = false;


    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(gfx::getVkInstance(), window, NULL, &vk_surface);
    assertVk(result);

    // TODO FIXME sometimes we want glfwGetWindowSize, sometimes we want glfwGetFramebufferSize.
    //     Check every instance of a window size being passed into a function call, and determine which one
    //     we should actually be passing.
    glfwGetWindowSize(window, &window_size_.x, &window_size_.y);
    glfwGetWindowPos(window, &window_pos_.x, &window_pos_.y);
    abortIfGlfwError();
    // TODO if current_width == current_height == 0, check if window is minimized or something; if it is, do
    // something that doesn't waste resources


    ImGuiContext* imgui_context = ImGui::CreateContext();
    alwaysAssert(imgui_context != NULL);

    ImPlotContext* implot_context = ImPlot::CreateContext();
    alwaysAssert(implot_context != NULL);

    const char* frametimeplot_axis_label = allocSprintf(
        "Frame time (ms)\nOver %.1lf ms intervals", FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS * 1000.
    );

    ImGuiIO& imgui_io = ImGui::GetIO();

    success = ImGui_ImplGlfw_InitForVulkan(window, true);
    alwaysAssert(success);

    success = gfx::initImGuiVulkanBackend();
    alwaysAssert(success);


    voxel_count_ = 100'000;
    p_voxels_ = mallocArray(gfx::MAX_VOXEL_COUNT, gfx::Voxel);
    p_voxels_in_frustum_ = mallocArray(gfx::MAX_VOXEL_COUNT, typeof(*p_voxels_in_frustum_));

    for (u32fast voxel_idx = 0; voxel_idx < voxel_count_; voxel_idx++) {

        vec3 random_0_to_1 {
            (f32)rand() / (f32)RAND_MAX,
            (f32)rand() / (f32)RAND_MAX,
            (f32)rand() / (f32)RAND_MAX,
        };

        p_voxels_[voxel_idx] = gfx::Voxel {
            .coord = worldspaceToIndexspaceInt((random_0_to_1 - 0.5f) * 500.0f),
            .color = vec4(random_0_to_1 * 255.0f, 255.0f),
        };
    }


    plugin::init();
    success = plugin::setFilewatchEnabled(PluginID_FluidSim, true);
    fluid_sim_plugin_filewatch_enabled_ = success;
    LOG_IF_F(ERROR, !success, "Failed to enable filewatch for fluid sim plugin.");

    PLUGIN_LOAD(fluid_sim_procs_, FluidSim);
    alwaysAssert(fluid_sim_procs_ != NULL);

    fluid_sim_plugin_versions_.push(fluid_sim_procs_);

    fluid_sim::SimData sim_data = initFluidSim(&fluid_sim_params_);


    gfx::RenderResources gfx_renderer {};

    {
        VkBuffer sim_vkbuffer = VK_NULL_HANDLE;
        VkDeviceSize sim_vkbuffer_size = 0;
        fluid_sim_procs_->getPositionsVertexBuffer(&sim_data, &sim_vkbuffer, &sim_vkbuffer_size);

        gfx::Result result_gfx = gfx::createRenderer(&gfx_renderer, sim_vkbuffer_size, sim_vkbuffer);
        assertGraphics(result_gfx);
    }


    memcpy(present_mode_priorities_, DEFAULT_PRESENT_MODE_PRIORITIES, sizeof(present_mode_priorities_));

    gfx::SurfaceResources gfx_surface {};
    gfx::Result result_gfx = gfx::createSurfaceResources(
        vk_surface,
        present_mode_priorities_,
        VkExtent2D { .width = (u32)window_size_.x, .height = (u32)window_size_.y },
        &gfx_surface,
        &present_mode_
    );
    // TODO handle error_window_size_zero
    assertGraphics(result_gfx);

    window_draw_region_ = centeredSubregion_16x9((u32)window_size_.x, (u32)window_size_.y);


    gfx::attachSurfaceToRenderer(gfx_surface, gfx_renderer);


    checkedGlfwGetCursorPos(window, &cursor_pos_.x, &cursor_pos_.y);
    u32fast frame_counter = 0;


    // main loop
    while (true) {

        ZoneScopedN("main loop");

        f64 delta_t_seconds = 0.0;
        {
            f64 time = glfwGetTime();
            delta_t_seconds = time - frame_start_time_seconds_;
            frame_start_time_seconds_ = time;

            frametimeplot_frames_since_last_sample_++;
            frametimeplot_largest_reading_since_last_sample_ = glm::max(
                frametimeplot_largest_reading_since_last_sample_,
                delta_t_seconds
            );

            f64 time_since_last_sample = time - frametimeplot_last_sample_time_;
            if (time_since_last_sample >= FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS) {
                if (!frametimeplot_paused_) frametimeplot_samples_scrolling_buffer_.push(
                    (f32)(time_since_last_sample / (f64)frametimeplot_frames_since_last_sample_),
                    (f32)frametimeplot_largest_reading_since_last_sample_
                );
                frametimeplot_last_sample_time_ = time;
                frametimeplot_frames_since_last_sample_ = 0;
                frametimeplot_largest_reading_since_last_sample_ = 0.;
            }
        }

        if (!fluid_sim_paused_) {
            // TODO FIXME:
            //     This if-statement a hack to handle lag spikes.
            //     This assumes that any dt over an 8th of a second is an anomaly.
            //     If the dt is consistently that large, this will just freeze the sim, which is bad.
            // if (delta_t_seconds > (1.0 / 8.0)) {
            //     LOG_F(WARNING, "Lag spike detected; not advancing fluid sim for this frame. This is a hack and you should find another solution.");
            // }
            // else {
            ZoneScopedN("fluid_sim::advance");
            fluid_sim_procs_->advance(
                &sim_data,
                gfx::getVkContext(),
                thread_pool_,
                (f32)delta_t_seconds,
                render_finished_semaphore_will_be_signalled_ ? render_finished_semaphore_ : VK_NULL_HANDLE,
                sim_finished_semaphore_
            );
            render_finished_semaphore_will_be_signalled_ = false;
            sim_finished_semaphore_will_be_signalled_ = true;
            // }
        }
        else if (render_finished_semaphore_will_be_signalled_) {
            clearSemaphore(gfx::getVkContext(), render_finished_semaphore_, general_purpose_fence_);
            render_finished_semaphore_will_be_signalled_ = false;
        }

        // autoreload
        {
            if (shader_autoreload_enabled_) {
                gfx::ShaderReloadResult reload_result = gfx::reloadModifiedShaderSourceFiles(gfx_renderer);
                switch (reload_result) {
                    case gfx::ShaderReloadResult::no_shaders_need_reloading : break;
                    case gfx::ShaderReloadResult::success : last_shader_reload_failed_ = false; break;
                    case gfx::ShaderReloadResult::error : last_shader_reload_failed_ = true; break;
                }
            }
            if (fluid_sim_plugin_autoreload_enabled_) {

                bool success_bool = false;

                const FluidSimProcs* new_plugin_procs = NULL;

                f64 reload_start_time = glfwGetTime();
                PLUGIN_RELOAD_IF_MODIFIED(new_plugin_procs, FluidSim, &success_bool);
                f64 reload_duration = glfwGetTime() - reload_start_time;

                if (!success_bool) {
                    LOG_F(ERROR, "Fluid sim plugin auto-reload failed.");
                    fluid_sim_plugin_last_reload_failed_ = true;
                }
                else if (new_plugin_procs != NULL) {
                    LOG_F(INFO, "Fluid sim plugin auto-reloaded (%.1lf s).", reload_duration);
                    fluid_sim_plugin_last_reload_failed_ = false;
                    updateFluidSimPluginVersionAndProcs(new_plugin_procs);
                }
            }
        }

        glfwPollEvents();
        if (glfwWindowShouldClose(window)) goto LABEL_EXIT_MAIN_LOOP;

        { ZoneScopedN("ImGui_ImplVulkan_NewFrame"); ImGui_ImplVulkan_NewFrame(); }
        { ZoneScopedN("ImGui_ImplGlfw_NewFrame"); ImGui_ImplGlfw_NewFrame(); }
        { ZoneScopedN("ImGui::NewFrame"); ImGui::NewFrame(); }


        bool left_ctrl_was_pressed = left_ctrl_is_pressed_;
        left_ctrl_is_pressed_ = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL);
        abortIfGlfwError();

        if (!left_ctrl_was_pressed and left_ctrl_is_pressed_) camera_speed_ *= 8.0f;
        if (left_ctrl_was_pressed and !left_ctrl_is_pressed_) camera_speed_ *= 0.125f;


        bool left_ctrl_r_was_pressed = left_ctrl_r_is_pressed_;
        left_ctrl_r_is_pressed_ =
            glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS and
            glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
        abortIfGlfwError();

        if (!left_ctrl_r_was_pressed and left_ctrl_r_is_pressed_) {

            LOG_F(INFO, "Shader-reload keybind pressed. Triggering reload of modified shaders.");

            if (shader_file_tracking_enabled_) {
                gfx::ShaderReloadResult reload_result = gfx::reloadModifiedShaderSourceFiles(gfx_renderer);
                switch (reload_result) {
                    case gfx::ShaderReloadResult::no_shaders_need_reloading : break;
                    case gfx::ShaderReloadResult::success : last_shader_reload_failed_ = false; break;
                    case gfx::ShaderReloadResult::error : last_shader_reload_failed_ = true; break;
                }
            }
            else {
                LOG_F(ERROR, "Shader-reload keybind pressed, but shader file tracking is disabled. Doing nothing.");
                // TODO also make some visual indication of failure via imgui
            };
        }


        bool left_alt_was_pressed = left_alt_is_pressed_;
        left_alt_is_pressed_ = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS;
        abortIfGlfwError();

        if (!left_alt_was_pressed and left_alt_is_pressed_) {

            cursor_visible_ = !cursor_visible_;

            if (cursor_visible_) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
            }
            else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

                // If the cursor was outside the window when we disabled the cursor and enabled raw mouse
                // motion, the cursor position jumps to the center of the window.
                // At least, this was true in my (X11 + i3wm) environment.
                // The resulting cursor delta-position calculation causes the camera to change direction,
                // which is annoying because you lose track of what you were looking at.
                //
                // I solve the problem by overwriting the mouse pos and restarting the frame, so that our
                // delta pos calculation in the next frame does not use the out-of-window position.
                //
                // We might not have to restart the frame, but I'm doing so for simplicity; I don't want to
                // think about potential state consistency issues caused by calling glfwPollEvents() at
                // different times in the same frame.
                glfwPollEvents();
                checkedGlfwGetCursorPos(window, &cursor_pos_.x, &cursor_pos_.y);

                // TODO FIXME: having to remember to reset these things every time you `continue` to the next
                // main loop iteration is going to lead to bugs. Do something more sane.
                ImGui::EndFrame();
                if (sim_finished_semaphore_will_be_signalled_) {
                    clearSemaphore(gfx::getVkContext(), sim_finished_semaphore_, general_purpose_fence_);
                    sim_finished_semaphore_will_be_signalled_ = false;
                }
                continue; // main loop
            }
        }


        bool left_ctrl_o_was_pressed = left_ctrl_g_is_pressed_;
        left_ctrl_g_is_pressed_ =
            glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS and
            glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
        abortIfGlfwError();

        if (!left_ctrl_o_was_pressed and left_ctrl_g_is_pressed_) {
            imgui_overlay_visible_ = !imgui_overlay_visible_;
        }

        if (imgui_overlay_visible_) {

            ZoneScopedN("Imgui");

            // Crosshair.
            // TODO Should we do this in our own renderer?
            ImGui::GetBackgroundDrawList()->AddCircleFilled(
                ImVec2 {
                    (f32)window_draw_region_.offset.x + (f32)window_draw_region_.extent.width / 2.0f,
                    (f32)window_draw_region_.offset.y + (f32)window_draw_region_.extent.height / 2.0f,
                },
                2.0f, // radius
                IM_COL32(128, 128, 128, 255), // color
                0 // num_segments
            );

            guiWindow_camera(&camera_pos_, &camera_angles_, &camera_speed_);
            guiWindow_selection();

            {
                bool grid_shader_enabled = grid_shader_enabled_;
                gfx::PresentModeFlags supported_present_modes = gfx::getSupportedPresentModes(gfx_surface);
                gfx::PresentMode selected_present_mode = present_mode_;

                GuiWindowGraphicsResult res = guiWindow_graphics(
                    last_shader_reload_failed_,
                    shader_file_tracking_enabled_,
                    &shader_autoreload_enabled_,
                    &grid_shader_enabled,
                    supported_present_modes,
                    &selected_present_mode
                );
                assert(shader_file_tracking_enabled_ or !shader_autoreload_enabled_);

                if (grid_shader_enabled != grid_shader_enabled_) {
                    grid_shader_enabled_ = grid_shader_enabled;
                    gfx::setGridEnabled(grid_shader_enabled_);
                }

                if (res.button_pressed_reload_all_shaders) {
                    LOG_F(INFO, "Reload-all-shaders button pressed. Triggering reload.");
                    success = gfx::reloadAllShaders(gfx_renderer);
                    last_shader_reload_failed_ = !success;
                }

                if (selected_present_mode != present_mode_) {

                    memcpy(present_mode_priorities_, DEFAULT_PRESENT_MODE_PRIORITIES, sizeof(present_mode_priorities_));
                    assert(0 <= selected_present_mode and selected_present_mode < gfx::PRESENT_MODE_ENUM_COUNT);
                    present_mode_priorities_[selected_present_mode] = UINT8_MAX;

                    gfx::Result gfx_result = gfx::updateSurfaceResources(
                        gfx_surface, present_mode_priorities_, DEFAULT_WINDOW_EXTENT, &present_mode_
                    );
                    assertGraphics(gfx_result);

                    // TODO FIXME: having to remember to reset these things every time you `continue` to the
                    // next main loop iteration is going to lead to bugs. Do something more sane.
                    ImGui::EndFrame();
                    if (sim_finished_semaphore_will_be_signalled_) {
                        clearSemaphore(gfx::getVkContext(), sim_finished_semaphore_, general_purpose_fence_);
                        sim_finished_semaphore_will_be_signalled_ = false;
                    }
                    continue; // main loop
                }
            }

            {
                bool pause = frametimeplot_paused_;
                guiWindow_performance(
                    frametimeplot_axis_label, &frametimeplot_samples_scrolling_buffer_, &pause
                );
                if (frametimeplot_paused_ and !pause) frametimeplot_samples_scrolling_buffer_.reset();
                frametimeplot_paused_ = pause;
            }

            {

                u32fast selected_plugin_version = fluid_sim_selected_plugin_version_;
                GuiWindowFluidSimResult res = guiWindow_fluidSim(
                    fluid_sim_plugin_last_reload_failed_,
                    fluid_sim_plugin_filewatch_enabled_,
                    &fluid_sim_paused_,
                    &fluid_sim_plugin_autoreload_enabled_,
                    &fluid_sim_plugin_versions_.ui_elements,
                    &fluid_sim_plugin_versions_.hidden_ui_element_count,
                    &selected_plugin_version,
                    &fluid_sim_params_
                );

                if (res.sim_params_modified) fluid_sim_procs_->setParams(&sim_data, &fluid_sim_params_);

                if (res.button_pressed_reset_state) {
                    fluid_sim_procs_->destroy(&sim_data, gfx::getVkContext());
                    sim_data = initFluidSim(&fluid_sim_params_);
                }

                if (selected_plugin_version != fluid_sim_selected_plugin_version_) {
                    LOG_F(INFO, "Switching to fluid sim plugin version %" PRIuFAST32 " due to user selection.", selected_plugin_version);
                    fluid_sim_selected_plugin_version_ = selected_plugin_version;
                    fluid_sim_procs_ = fluid_sim_plugin_versions_.procs.ptr[selected_plugin_version];
                }

                if (res.button_pressed_reload) {

                    LOG_F(INFO, "Reloading fluid sim plugin due to GUI button pressed.");

                    const FluidSimProcs* new_plugin_procs = NULL;

                    f64 reload_start_time = glfwGetTime();
                    PLUGIN_RELOAD(new_plugin_procs, FluidSim);
                    f64 reload_duration = glfwGetTime() - reload_start_time;

                    fluid_sim_plugin_last_reload_failed_ = new_plugin_procs == NULL;

                    if (fluid_sim_plugin_last_reload_failed_) {
                        LOG_F(ERROR, "Failed to reload fluid sim plugin.");
                    }
                    else {
                        LOG_F(INFO, "Fluid sim plugin reloaded (%.1lf s).", reload_duration);
                        updateFluidSimPluginVersionAndProcs(new_plugin_procs);
                    }
                }
            }
        }


        ivec2 prev_window_pos = window_pos_;
        {
            ZoneScopedN("glfwGetWindowPos");
            glfwGetWindowPos(window, &window_pos_.x, &window_pos_.y);
        }
        abortIfGlfwError();
        bool window_repositioned = prev_window_pos != window_pos_;

        ivec2 prev_window_size = window_size_;
        {
            ZoneScopedN("glfwGetWindowSize");
            glfwGetWindowSize(window, &window_size_.x, &window_size_.y);
        }
        abortIfGlfwError();

        bool window_resized = prev_window_size != window_size_;
        window_or_surface_out_of_date_ |= window_resized;


        if (window_or_surface_out_of_date_) {

            result_gfx = gfx::updateSurfaceResources(
                gfx_surface,
                present_mode_priorities_,
                VkExtent2D { (u32)window_size_.x, (u32)window_size_.y },
                &present_mode_
            );
            assertGraphics(result_gfx);

            window_draw_region_ = centeredSubregion_16x9((u32)window_size_.x, (u32)window_size_.y);

            window_or_surface_out_of_date_ = false;
        }


        vec3 camera_direction_unit = glm::rotate(vec3(1, 0, 0), camera_angles_.y, vec3(0, 0, 1));
        camera_direction_unit = glm::rotate(camera_direction_unit, camera_angles_.x, vec3(0, 1, 0));

        vec3 camera_horizontal_direction_unit = vec3(
            glm::cos(camera_angles_.x),
            0,
            -glm::sin(camera_angles_.x)
        );
        vec3 camera_horizontal_right_direction_unit = vec3(
            -camera_horizontal_direction_unit.z,
            0,
            camera_horizontal_direction_unit.x
        );
        vec3 camera_y_axis_unit = glm::rotate(
            camera_direction_unit,
            (f32)(0.5*M_PI),
            camera_horizontal_right_direction_unit
        );

        vec3 camera_vel = vec3(0);
        if (!cursor_visible_) {

            int w_key_state = glfwGetKey(window, GLFW_KEY_W);
            int s_key_state = glfwGetKey(window, GLFW_KEY_S);
            int a_key_state = glfwGetKey(window, GLFW_KEY_A);
            int d_key_state = glfwGetKey(window, GLFW_KEY_D);
            int space_key_state = glfwGetKey(window, GLFW_KEY_SPACE);
            int lshift_key_state = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
            abortIfGlfwError();

            if (w_key_state == GLFW_PRESS) camera_vel += camera_horizontal_direction_unit;
            if (s_key_state == GLFW_PRESS) camera_vel -= camera_horizontal_direction_unit;
            if (d_key_state == GLFW_PRESS) camera_vel += camera_horizontal_right_direction_unit;
            if (a_key_state == GLFW_PRESS) camera_vel -= camera_horizontal_right_direction_unit;
            if (space_key_state == GLFW_PRESS) camera_vel.y += 1.0f;
            if (lshift_key_state == GLFW_PRESS) camera_vel.y -= 1.0f;

            f32 tmp_speed = length(camera_vel);
            if (tmp_speed > 1e-5) {
                camera_vel /= tmp_speed;
                camera_vel *= camera_speed_;
            }
        }
        camera_pos_ += camera_vel * (f32)delta_t_seconds;


        dvec2 prev_cursor_pos = cursor_pos_;
        {
            ZoneScopedN("checkedGlfwGetCursorPos");
            checkedGlfwGetCursorPos(window, &cursor_pos_.x, &cursor_pos_.y);
        }

        // If window is resized or moved, the virtual cursor position reported by glfw changes even if the
        // user did not move it. We must ignore the cursor position change in this case.
        // TODO this doesn't seem to have actually solved the problem
        if (window_resized || window_repositioned) prev_cursor_pos = cursor_pos_;


        // compute new camera direction
        if (!cursor_visible_) {
            // We don't need to scale anything by delta_t here; `cursor_pos - prev_cursor_pos` already scales
            // linearly with frame duration.

            vec2 delta_cursor_pos = cursor_pos_ - prev_cursor_pos;
            vec2 delta_cursor_in_camera_frame = flip_screenXY_to_cameraXY(delta_cursor_pos);

            f32 angle_coef = 0.005f; // arbitrarily chosen
            vec2 delta_angles = angle_coef * delta_cursor_in_camera_frame;
            delta_angles.x = -delta_angles.x; // positive angle is counterclockwise about y-axis

            vec2 new_cam_angles = camera_angles_ + delta_angles;
            new_cam_angles.x = (f32)glm::mod((f64)new_cam_angles.x, 2.*M_PI);
            new_cam_angles.y = (f32)glm::clamp((f64)new_cam_angles.y, -0.5*M_PI, 0.5*M_PI);

            camera_angles_ = new_cam_angles;
        }


        bool left_mouse_was_pressed = left_mouse_is_pressed_;
        left_mouse_is_pressed_ = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        abortIfGlfwError();
        if (imgui_io.WantCaptureMouse) left_mouse_is_pressed_ = false;

        bool right_mouse_was_pressed = right_mouse_is_pressed_;
        right_mouse_is_pressed_ = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        abortIfGlfwError();
        if (imgui_io.WantCaptureMouse) right_mouse_is_pressed_ = false;
        (void)right_mouse_was_pressed; // TODO FIXME delet dis


        Hexahedron view_frustum = frustumFromScreenspacePoints(
            camera_pos_,
            camera_direction_unit,
            camera_horizontal_right_direction_unit,
            camera_y_axis_unit,
            vec2 { -1.f, -1.f },
            vec2 { 1.f, 1.f }
        );
        view_frustum.near_bot_left_p = worldspaceToIndexspaceFloat(view_frustum.near_bot_left_p);
        view_frustum.far_top_right_p = worldspaceToIndexspaceFloat(view_frustum.far_top_right_p);

        voxels_in_frustum_count_ = frustumCull(&view_frustum, voxel_count_, p_voxels_, p_voxels_in_frustum_);


        if (cursor_visible_) {
            if (!left_mouse_was_pressed and left_mouse_is_pressed_) {
                selection_active_ = true;
                selection_point1_windowspace_ = cursor_pos_;
                selection_point2_windowspace_ = selection_point1_windowspace_;
            }
            if (left_mouse_was_pressed and !left_mouse_is_pressed_) selection_active_ = false;

            if (selection_active_) {

                selection_point2_windowspace_ = cursor_pos_;

                Hexahedron frustum = frustumFromScreenspacePoints(
                    camera_pos_,
                    camera_direction_unit,
                    camera_horizontal_right_direction_unit,
                    camera_y_axis_unit,
                    windowspaceToNormalizedScreenspace(selection_point1_windowspace_, &window_draw_region_),
                    windowspaceToNormalizedScreenspace(selection_point2_windowspace_, &window_draw_region_)
                );
                frustum.near_bot_left_p = worldspaceToIndexspaceFloat(frustum.near_bot_left_p);
                frustum.far_top_right_p = worldspaceToIndexspaceFloat(frustum.far_top_right_p);

                u32fast selected_voxel_idx = 0;
                {
                    ZoneScopedN("Get selected voxels");
                    for (u32fast voxel_idx = 0; voxel_idx < voxels_in_frustum_count_; voxel_idx++) {
                        if (pointIsInHexahedron(&frustum, p_voxels_in_frustum_[voxel_idx].pos)) {
                            p_selected_voxel_indices_[selected_voxel_idx] = p_voxels_in_frustum_[voxel_idx].idx;
                            selected_voxel_idx++;
                        };
                    }
                }
                selected_voxel_index_count_ = selected_voxel_idx;

                ImGui::GetBackgroundDrawList()->AddRect(
                    ImVec2 { selection_point1_windowspace_.x, selection_point1_windowspace_.y },
                    ImVec2 { selection_point2_windowspace_.x, selection_point2_windowspace_.y },
                    IM_COL32(255, 0, 0, 255)
                );
            }
        }


        mat4 world_to_screen_transform = glm::identity<mat4>();
        {
            mat4 world_to_camera_transform = glm::lookAt(
                camera_pos_, // eye
                camera_pos_ + camera_direction_unit, // position you're looking at
                camera_y_axis_unit // "Normalized up vector, how the camera is oriented."
            );
            mat4 camera_to_clip_transform = glm::perspective(
                (f32)FOV_Y, // fovy
                (f32)ASPECT_RATIO_X_OVER_Y, // aspect
                (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE, // zNear
                (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE // zFar
            );

            // By default:
            //     In GLM clip coordinates: the Y axis points upward.
            //     In Vulkan normalized device coordinates: the Y axis points downward.
            mat4 glm_to_vulkan = glm::identity<mat4>();
            glm_to_vulkan[1][1] = -1.0f; // mirror y axis

            world_to_screen_transform = world_to_camera_transform * world_to_screen_transform;
            world_to_screen_transform = camera_to_clip_transform * world_to_screen_transform;
            world_to_screen_transform = glm_to_vulkan * world_to_screen_transform;
        }
        mat4 world_to_screen_transform_inverse = glm::inverse(world_to_screen_transform);


        u32fast voxel_being_looked_at_idx = INVALID_VOXEL_IDX;
        {
            vec3 ray_origin = camera_pos_ + camera_direction_unit * (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE;
            vec3 ray_direction_unit = camera_direction_unit;

            vec2 cursor_pos_screenspace = windowspaceToNormalizedScreenspace(cursor_pos_, &window_draw_region_);
            vec2 cursor_pos_viewportspace = flip_screenXY_to_cameraXY(cursor_pos_screenspace);

            if (cursor_visible_) {

                ray_origin +=
                    camera_horizontal_right_direction_unit
                    * cursor_pos_viewportspace.x
                    * 0.5f * VIEW_FRUSTUM_NEAR_SIDE_SIZE_X;

                ray_origin +=
                    camera_y_axis_unit
                    * cursor_pos_viewportspace.y
                    * 0.5f * VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y;

                ray_direction_unit = glm::normalize(ray_origin - camera_pos_);
            }

            voxel_being_looked_at_idx = rayCast(
                ray_origin,
                ray_direction_unit,
                voxels_in_frustum_count_,
                p_voxels_in_frustum_
            );
        }
        if (voxel_being_looked_at_idx != INVALID_VOXEL_IDX) {

            ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
            assert(draw_list != NULL);

            const vec3 cube_center = indexspaceToWorldspace(p_voxels_[voxel_being_looked_at_idx].coord);
            const f32 rad = gfx::VOXEL_RADIUS;

            vec4 vec4_p000 = world_to_screen_transform * vec4(cube_center + vec3(-rad, -rad, -rad), 1.0f);
            vec4 vec4_p001 = world_to_screen_transform * vec4(cube_center + vec3(-rad, -rad,  rad), 1.0f);
            vec4 vec4_p010 = world_to_screen_transform * vec4(cube_center + vec3(-rad,  rad, -rad), 1.0f);
            vec4 vec4_p011 = world_to_screen_transform * vec4(cube_center + vec3(-rad,  rad,  rad), 1.0f);
            vec4 vec4_p100 = world_to_screen_transform * vec4(cube_center + vec3( rad, -rad, -rad), 1.0f);
            vec4 vec4_p101 = world_to_screen_transform * vec4(cube_center + vec3( rad, -rad,  rad), 1.0f);
            vec4 vec4_p110 = world_to_screen_transform * vec4(cube_center + vec3( rad,  rad, -rad), 1.0f);
            vec4 vec4_p111 = world_to_screen_transform * vec4(cube_center + vec3( rad,  rad,  rad), 1.0f);

            vec2 p000 = vec4_p000 / vec4_p000.w;
            vec2 p001 = vec4_p001 / vec4_p001.w;
            vec2 p010 = vec4_p010 / vec4_p010.w;
            vec2 p011 = vec4_p011 / vec4_p011.w;
            vec2 p100 = vec4_p100 / vec4_p100.w;
            vec2 p101 = vec4_p101 / vec4_p101.w;
            vec2 p110 = vec4_p110 / vec4_p110.w;
            vec2 p111 = vec4_p111 / vec4_p111.w;

            p000 = (p000 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p001 = (p001 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p010 = (p010 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p011 = (p011 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p100 = (p100 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p101 = (p101 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p110 = (p110 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);
            p111 = (p111 + 1.0f) * 0.5f * vec2(window_draw_region_.extent.width, window_draw_region_.extent.height) + vec2(window_draw_region_.offset.x, window_draw_region_.offset.y);

            draw_list->AddLine({ p000.x, p000.y }, { p001.x, p001.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p001.x, p001.y }, { p011.x, p011.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p011.x, p011.y }, { p010.x, p010.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p010.x, p010.y }, { p000.x, p000.y }, IM_COL32(0, 128, 255, 255), 2.0f);

            draw_list->AddLine({ p100.x, p100.y }, { p101.x, p101.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p101.x, p101.y }, { p111.x, p111.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p111.x, p111.y }, { p110.x, p110.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p110.x, p110.y }, { p100.x, p100.y }, IM_COL32(0, 128, 255, 255), 2.0f);

            draw_list->AddLine({ p000.x, p000.y }, { p100.x, p100.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p001.x, p001.y }, { p101.x, p101.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p011.x, p011.y }, { p111.x, p111.y }, IM_COL32(0, 128, 255, 255), 2.0f);
            draw_list->AddLine({ p010.x, p010.y }, { p110.x, p110.y }, IM_COL32(0, 128, 255, 255), 2.0f);
        }


        ImGui::Render();
        ImDrawData* imgui_draw_data = ImGui::GetDrawData();

        VkBuffer sim_vkbuffer = VK_NULL_HANDLE;
        VkDeviceSize sim_vkbuffer_size = 0;
        fluid_sim_procs_->getPositionsVertexBuffer(&sim_data, &sim_vkbuffer, &sim_vkbuffer_size);

        gfx::RenderResult render_result = gfx::render(
            gfx_surface,
            window_draw_region_,
            &world_to_screen_transform,
            &world_to_screen_transform_inverse,
            (1.f / 128.f), // particle_radius
            (f32)(VIEW_FRUSTUM_FAR_SIDE_DISTANCE - VIEW_FRUSTUM_NEAR_SIDE_DISTANCE), // raymarch_max_travel_distance
            imgui_draw_data,
            (u32)voxel_count_,
            p_voxels_,
            (u32)selected_voxel_index_count_,
            p_selected_voxel_indices_,
            (u32)sim_data.particle_count,
            sim_vkbuffer,
            false,
            sim_finished_semaphore_will_be_signalled_ ? sim_finished_semaphore_ : VK_NULL_HANDLE,
            render_finished_semaphore_
        );
        sim_finished_semaphore_will_be_signalled_ = false;
        render_finished_semaphore_will_be_signalled_ = true;

        switch (render_result) {
            case gfx::RenderResult::error_surface_resources_out_of_date:
            case gfx::RenderResult::success_surface_resources_out_of_date:
            {
                LOG_F(INFO, "Surface resources out of date.");
                window_or_surface_out_of_date_ = true;
                break;
            }
            case gfx::RenderResult::success: break;
        }

        frame_counter++;
        FrameMark;
    };

    LABEL_EXIT_MAIN_LOOP: {}


    // glfwTerminate(); // commented out because this sometimes adds significant shutdown time
    exit(0);
}
