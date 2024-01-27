#include <cstdint>
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

#include "types.hpp"
#include "error_util.hpp"
#include "graphics.hpp"
#include "alloc_util.hpp"

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

//
// Global constants ==========================================================================================
//

const char* APP_NAME = "an game";

constexpr VkExtent2D DEFAULT_WINDOW_EXTENT { 800, 600 }; // TODO weird default, because everything else is 16:9

constexpr double ASPECT_RATIO_X_OVER_Y = 16.0 / 9.0;
constexpr double ASPECT_RATIO_Y_OVER_X = 1.0 / ASPECT_RATIO_X_OVER_Y;

constexpr double CAMERA_MOVEMENT_SPEED = 3.0; // unit: m/s

constexpr double VIEW_FRUSTUM_NEAR_SIDE_DISTANCE = 0.15; // unit: m
constexpr double VIEW_FRUSTUM_FAR_SIDE_DISTANCE = 500.0;

constexpr double FOV_Y = 0.25 * M_PI; // Full angle from top to bottom of the frustum.
static_assert(FOV_Y < M_PI - 1e-5);

// NOTE: as x -> 90deg, tan(x) -> inf. Don't let any FOV come close to 180 deg.
const float VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y = (f32)(VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * 2.0*glm::tan(0.5*FOV_Y));
const float VIEW_FRUSTUM_NEAR_SIDE_SIZE_X = VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y * (f32)ASPECT_RATIO_X_OVER_Y;
const vec2 VIEW_FRUSTUM_NEAR_SIDE_SIZE { VIEW_FRUSTUM_NEAR_SIDE_SIZE_X, VIEW_FRUSTUM_NEAR_SIDE_SIZE_Y };

const u32fast INVALID_VOXEL_IDX = UINT32_MAX;

constexpr f64 FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS = 10.0;
constexpr f64 FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS = 1. / 30.;
constexpr u32fast FRAMETIME_PLOT_MAX_SAMPLE_COUNT =
    (u32fast)(FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS / FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS);

const u8 DEFAULT_PRESENT_MODE_PRIORITIES[3] {
    [gfx::PRESENT_MODE_IMMEDIATE] = 1,
    [gfx::PRESENT_MODE_MAILBOX] = 3,
    [gfx::PRESENT_MODE_FIFO] = 2,
};

//
// Global variables ==========================================================================================
//

vec3 camera_pos_ { 0, 0, 0 };

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
bool left_ctrl_g_is_pressed_ = false;
bool left_ctrl_r_is_pressed_ = false;

bool imgui_overlay_visible_ = false;

u32fast voxel_count_ = 0;
gfx::Voxel* p_voxels_ = NULL;

bool shader_autoreload_enabled_ = true;
bool shader_file_tracking_enabled_ = false;
bool shader_reload_all_button_is_pressed_ = false;
bool last_shader_reload_failed_ = false;

struct {
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


static vec2 flip_screenXY_to_cameraXY(vec2 screen_coords) {
    return vec2(screen_coords.x, -screen_coords.y);
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


struct AxisAlignedBox {
    f32 x_min;
    f32 y_min;
    f32 z_min;
    f32 x_max;
    f32 y_max;
    f32 z_max;
};
struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 direction_reciprocal;
};
// TODO OPTIMIZE
// src: https://tavianator.com/2022/ray_box_boundary.html
/// Returns a number <= 0 if there is no collision.
/// TODO FIXME:
/// 1. Doesn't handle the case where the ray is parallel to an axis.
/// 2. Don't know if it handles the case where the ray origin is inside the box.
static inline f32 rayBoxInteriorCollisionTime(const Ray* ray, const AxisAlignedBox* box) {

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


/// Returns the index of the earliest collision, or INVALID_VOXEL_IDX if there are no collisions.
static u32fast rayCast(
    vec3 ray_origin,
    vec3 ray_direction,
    u32fast voxel_count,
    const gfx::Voxel* p_voxels
) {
    ZoneScoped;

    u32fast earliest_collision_idx = INVALID_VOXEL_IDX;
    f32 earliest_collision_time = INFINITY;

    Ray ray {
        .origin = ray_origin,
        .direction = ray_direction,
        .direction_reciprocal = 1.f / ray_direction,
    };

    for (u32fast voxel_idx = 0; voxel_idx < voxel_count; voxel_idx++) {

        ivec3 voxel_coord = p_voxels[voxel_idx].coord;
        AxisAlignedBox box {
            .x_min = (f32)voxel_coord.x - gfx::VOXEL_RADIUS,
            .y_min = (f32)voxel_coord.y - gfx::VOXEL_RADIUS,
            .z_min = (f32)voxel_coord.z - gfx::VOXEL_RADIUS,
            .x_max = (f32)voxel_coord.x + gfx::VOXEL_RADIUS,
            .y_max = (f32)voxel_coord.y + gfx::VOXEL_RADIUS,
            .z_max = (f32)voxel_coord.z + gfx::VOXEL_RADIUS,
        };

        f32 t = rayBoxInteriorCollisionTime(&ray, &box);
        if (0.0f < t and t < earliest_collision_time) {
            earliest_collision_time = t;
            earliest_collision_idx = voxel_idx;
        }
    }

    return earliest_collision_idx;
}


int main(int argc, char** argv) {

    ZoneScoped;

    loguru::init(argc, argv);
    #ifndef NDEBUG
        LOG_F(INFO, "Debug build.");
    #else
        LOG_F(INFO, "Release build.");
    #endif

    int success = glfwInit();
    assertGlfw(success);


    const char* specific_device_request = getenv("PHYSICAL_DEVICE_NAME"); // can be NULL
    gfx::init(APP_NAME, specific_device_request);

    success = gfx::setShaderSourceFileModificationTracking(true);
    shader_file_tracking_enabled_ = success;
    LOG_IF_F(ERROR, !success, "Failed to enable shader source file tracking.");


    gfx::RenderResources gfx_renderer {};
    gfx::Result res = gfx::createRenderer(&gfx_renderer);
    assertGraphics(res);


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


    memcpy(present_mode_priorities_, DEFAULT_PRESENT_MODE_PRIORITIES, sizeof(present_mode_priorities_));

    gfx::SurfaceResources gfx_surface {};
    res = gfx::createSurfaceResources(
        vk_surface,
        present_mode_priorities_,
        VkExtent2D { .width = (u32)window_size_.x, .height = (u32)window_size_.y },
        &gfx_surface,
        &present_mode_
    );
    // TODO handle error_window_size_zero
    assertGraphics(res);

    window_draw_region_ = centeredSubregion_16x9((u32)window_size_.x, (u32)window_size_.y);


    gfx::attachSurfaceToRenderer(gfx_surface, gfx_renderer);


    ImGuiContext* imgui_context = ImGui::CreateContext();
    alwaysAssert(imgui_context != NULL);

    ImPlotContext* implot_context = ImPlot::CreateContext();
    alwaysAssert(implot_context != NULL);

    char* frametimeplot_axis_label = NULL;
    {
        int len_without_nul = snprintf(
            NULL, 0,
            "Frame time (ms)\nOver %.1lf ms intervals", FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS * 1000.
        );
        alwaysAssert(len_without_nul > 0);

        const size_t len_with_nul = (size_t)len_without_nul + 1;
        frametimeplot_axis_label = (char*)mallocAsserted(len_with_nul);

        len_without_nul = snprintf(
            frametimeplot_axis_label, len_with_nul,
            "Frame time (ms)\nOver %.1lf ms intervals", FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS * 1000.
        );
        alwaysAssert(len_without_nul > 0);
    };

    // TODO Why is this code here? Delete if not needed.
    ImGuiIO& imgui_io = ImGui::GetIO();
    (void)imgui_io;

    success = ImGui_ImplGlfw_InitForVulkan(window, true);
    alwaysAssert(success);

    success = gfx::initImGuiVulkanBackend();
    alwaysAssert(success);


    voxel_count_ = 100'000;
    p_voxels_ = mallocArray(voxel_count_, gfx::Voxel);

    for (u32fast voxel_idx = 0; voxel_idx < voxel_count_; voxel_idx++) {

        vec3 random_0_to_1 {
            (f32)rand() / (f32)RAND_MAX,
            (f32)rand() / (f32)RAND_MAX,
            (f32)rand() / (f32)RAND_MAX,
        };

        p_voxels_[voxel_idx] = gfx::Voxel {
            .coord = (random_0_to_1 - 0.5f) * 500.0f,
            .color = vec4(random_0_to_1 * 255.0f, 255.0f),
        };
    }


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

        if (shader_autoreload_enabled_ and shader_file_tracking_enabled_) {
            gfx::ShaderReloadResult reload_result = gfx::reloadModifiedShaderSourceFiles(gfx_renderer);
            switch (reload_result) {
                case gfx::ShaderReloadResult::no_shaders_need_reloading : break;
                case gfx::ShaderReloadResult::success : last_shader_reload_failed_ = false; break;
                case gfx::ShaderReloadResult::error : last_shader_reload_failed_ = true; break;
            }
        }

        glfwPollEvents();
        if (glfwWindowShouldClose(window)) goto LABEL_EXIT_MAIN_LOOP;

        { ZoneScopedN("ImGui_ImplVulkan_NewFrame"); ImGui_ImplVulkan_NewFrame(); }
        { ZoneScopedN("ImGui_ImplGlfw_NewFrame"); ImGui_ImplGlfw_NewFrame(); }
        { ZoneScopedN("ImGui::NewFrame"); ImGui::NewFrame(); }


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

            ImGuiWindowFlags common_imgui_window_flags = ImGuiWindowFlags_NoFocusOnAppearing;
            if (!cursor_visible_) common_imgui_window_flags |= ImGuiWindowFlags_NoInputs;


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


            ImGui::Begin("Camera", NULL, common_imgui_window_flags | ImGuiWindowFlags_AlwaysAutoResize);

            f32 user_pos_input[3] { camera_pos_.x, camera_pos_.y, camera_pos_.z };
            if (ImGui::DragFloat3("Position", user_pos_input, 0.1f, 0.0, 0.0, "%.1f")) {
                camera_pos_.x = user_pos_input[0];
                camera_pos_.y = user_pos_input[1];
                camera_pos_.z = user_pos_input[2];
            };

            ImGui::SliderAngle("Rotation X", &camera_angles_.x, 0.0, 360.0);
            ImGui::SliderAngle("Rotation Y", &camera_angles_.y, -90.0, 90.0);

            ImGui::End();


            ImGui::Begin("Shaders", NULL, common_imgui_window_flags | ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("Last reload:");
            ImGui::SameLine();
            if (last_shader_reload_failed_) ImGui::TextColored(ImVec4 { 1., 0., 0., 1. }, "failed");
            else ImGui::TextColored(ImVec4 { 0., 1., 0., 1.}, "success");

            bool shader_reload_all_button_was_pressed = shader_reload_all_button_is_pressed_;
            shader_reload_all_button_is_pressed_ = ImGui::Button("Reload all");

            if (shader_file_tracking_enabled_) ImGui::Checkbox("Auto-reload", &shader_autoreload_enabled_);
            else {
                ImGui::BeginDisabled();
                ImGui::Checkbox("Auto-reload (unavailable)", &shader_autoreload_enabled_);
                ImGui::EndDisabled();
            }

            ImGui::End();

            if (!shader_reload_all_button_was_pressed and shader_reload_all_button_is_pressed_) {
                LOG_F(INFO, "Reload-all-shaders button pressed. Triggering reload.");
                success = gfx::reloadAllShaders(gfx_renderer);
                last_shader_reload_failed_ = !success;
            }


            ImGui::Begin("Performance", NULL, common_imgui_window_flags);

            {
                bool checkbox_clicked = ImGui::Checkbox("Pause plot", &frametimeplot_paused_);
                if (checkbox_clicked and !frametimeplot_paused_) {
                    frametimeplot_samples_scrolling_buffer_.reset();
                }
            }

            if (ImPlot::BeginPlot(frametimeplot_axis_label, ImVec2(-1,-1))) {

                ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_Lock);
                ImPlot::SetupAxisLimits(ImAxis_X1, -FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS, 0.0);
                ImPlot::SetupAxisFormat(ImAxis_X1, "%.0fs");

                ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_LockMin);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 3.0);

                ImPlot::PlotShaded<f32>(
                    "Avg",
                    (const f32*)&frametimeplot_samples_scrolling_buffer_.samples_avg_milliseconds,
                    (int)frametimeplot_samples_scrolling_buffer_.sample_count,
                    0.0, // yref
                    FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS, // xscale
                    -FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS, // xstart
                    ImPlotShadedFlags_None, // flags
                    (int)frametimeplot_samples_scrolling_buffer_.first_sample_index // offset
                );

                ImPlot::PlotLine<f32>(
                    "Max",
                    (const f32*)&frametimeplot_samples_scrolling_buffer_.samples_max_milliseconds,
                    (int)frametimeplot_samples_scrolling_buffer_.sample_count,
                    FRAMETIME_PLOT_SAMPLE_INTERVAL_SECONDS, // xscale
                    -FRAMETIME_PLOT_DISPLAY_DOMAIN_SECONDS, // xstart
                    ImPlotShadedFlags_None, // flags
                    (int)frametimeplot_samples_scrolling_buffer_.first_sample_index // offset
                );

                ImPlot::EndPlot();
            }

            ImGui::End();


            ImGui::Begin("Present mode");

            gfx::PresentModeFlags supported_present_modes = gfx::getSupportedPresentModes(gfx_surface);
            bool present_mode_button_pressed = false;
            int selected_present_mode = present_mode_;
            {
                ImGui::BeginDisabled(!(supported_present_modes & gfx::PRESENT_MODE_MAILBOX_BIT));
                present_mode_button_pressed |= ImGui::RadioButton(
                    "Mailbox", &selected_present_mode, gfx::PRESENT_MODE_MAILBOX
                );
                ImGui::EndDisabled();

                ImGui::BeginDisabled(!(supported_present_modes & gfx::PRESENT_MODE_FIFO_BIT));
                present_mode_button_pressed |= ImGui::RadioButton(
                    "FIFO", &selected_present_mode, gfx::PRESENT_MODE_FIFO
                );
                ImGui::EndDisabled();

                ImGui::BeginDisabled(!(supported_present_modes & gfx::PRESENT_MODE_IMMEDIATE_BIT));
                present_mode_button_pressed |= ImGui::RadioButton(
                    "Immediate", &selected_present_mode, gfx::PRESENT_MODE_IMMEDIATE
                );
                ImGui::EndDisabled();
            }
            ImGui::End();

            if (present_mode_button_pressed and selected_present_mode != present_mode_) {

                memcpy(present_mode_priorities_, DEFAULT_PRESENT_MODE_PRIORITIES, sizeof(present_mode_priorities_));
                assert(0 <= selected_present_mode and selected_present_mode < gfx::PRESENT_MODE_ENUM_COUNT);
                present_mode_priorities_[selected_present_mode] = UINT8_MAX;

                gfx::Result gfx_result = gfx::updateSurfaceResources(
                    gfx_surface, present_mode_priorities_, DEFAULT_WINDOW_EXTENT, &present_mode_
                );
                assertGraphics(gfx_result);

                ImGui::EndFrame();
                continue; // main loop
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

            res = gfx::updateSurfaceResources(
                gfx_surface,
                present_mode_priorities_,
                VkExtent2D { (u32)window_size_.x, (u32)window_size_.y },
                &present_mode_
            );
            assertGraphics(res);

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
                camera_vel *= CAMERA_MOVEMENT_SPEED;
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


        u32fast voxel_being_look_at_idx = rayCast(
            camera_pos_ + camera_direction_unit * (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE,
            camera_direction_unit,
            voxel_count_,
            p_voxels_
        );

        u32 outlined_voxel_index_count = 0;
        u32 outlined_voxel_index = (u32)voxel_being_look_at_idx;
        if (outlined_voxel_index != INVALID_VOXEL_IDX) outlined_voxel_index_count = 1;


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


        ImGui::Render();
        ImDrawData* imgui_draw_data = ImGui::GetDrawData();

        gfx::RenderResult render_result = gfx::render(
            gfx_surface,
            window_draw_region_,
            &world_to_screen_transform,
            &world_to_screen_transform_inverse,
            imgui_draw_data,
            (u32)voxel_count_,
            p_voxels_,
            (u32)outlined_voxel_index_count,
            &outlined_voxel_index
        );

        switch (render_result) {
            case gfx::RenderResult::error_surface_resources_out_of_date:
            case gfx::RenderResult::success_surface_resources_out_of_date:
            {
                LOG_F(INFO, "Surface resources out of date.");
                window_or_surface_out_of_date_ = true;
                continue; // main loop
            }
            case gfx::RenderResult::success: break;
        }

        frame_counter++;
        FrameMark;
    };

    LABEL_EXIT_MAIN_LOOP: {}


    glfwTerminate(); // TODO delet this; it sometimes adds (unnecessary, I think?) significant shutdown time.
    exit(0);
}
