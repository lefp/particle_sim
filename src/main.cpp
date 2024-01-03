#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cinttypes>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <loguru/loguru.hpp>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "types.hpp"
#include "error_util.hpp"
#include "graphics.hpp"

namespace gfx = graphics;

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::dvec2;
using glm::ivec2;

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

// These are the full angles from left to right (or top to bottom) of the view frustum.
constexpr double FOV_X = (f32)(0.5*M_PI);
constexpr double FOV_Y = FOV_X * ASPECT_RATIO_Y_OVER_X;
static_assert(FOV_X < M_PI - 1e-5);
static_assert(FOV_Y < M_PI - 1e-5);

const vec2 VIEW_FRUSTUM_NEAR_SIDE_SIZE {
    // NOTE: as x -> 90deg, tan(x) -> inf. Don't let any FOV come close to 180 deg.
    (f32)(VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * 2.0*glm::tan(0.5*FOV_X)),
    (f32)(VIEW_FRUSTUM_NEAR_SIDE_DISTANCE * 2.0*glm::tan(0.5*FOV_Y)),
};

//
// Global variables ==========================================================================================
//

vec3 camera_pos_ { 0, 0, 0 };

/// Spherical coordinates.
/// .x = rotation in XZ plane. Range [0, 2 pi). 0 = along x-axis, 0.5 pi = along negative z-axis
/// .y = angle from XZ plane. Range [-0.5 pi, 0.5 pi]. 0.5 pi = along y-axis.
vec2 camera_angles_ { 0, 0 };

dvec2 cursor_pos_ { 0, 0 };

ivec2 window_size_ { 0, 0 };
ivec2 window_pos_ { 0, 0 };
VkRect2D window_draw_region_ {};

bool window_or_surface_out_of_date_ = false;

f64 frame_start_time_seconds = 0;

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


int main(int argc, char** argv) {

    loguru::init(argc, argv);
    #ifndef NDEBUG
        LOG_F(INFO, "Debug build.");
    #else
        LOG_F(INFO), "Release build.");
    #endif

    int success = glfwInit();
    assertGlfw(success);


    const char* specific_device_request = getenv("PHYSICAL_DEVICE_NAME"); // can be NULL
    gfx::init(APP_NAME, specific_device_request);


    gfx::RenderResources gfx_renderer {};
    gfx::Result res = gfx::createVoxelRenderer(&gfx_renderer);
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

    gfx::SurfaceResources gfx_surface {};
    res = gfx::createSurfaceResources(
        vk_surface,
        VkExtent2D { .width = (u32)window_size_.x, .height = (u32)window_size_.y },
        &gfx_surface
    );
    // TODO handle error_window_size_zero
    assertGraphics(res);

    window_draw_region_ = centeredSubregion_16x9((u32)window_size_.x, (u32)window_size_.y);


    gfx::attachSurfaceToRenderer(gfx_surface, gfx_renderer);


    checkedGlfwGetCursorPos(window, &cursor_pos_.x, &cursor_pos_.y);
    u32fast frame_counter = 0;

    while (true) {

        LABEL_MAIN_LOOP_START: {}

        f64 delta_t_seconds;
        {
            f64 time = glfwGetTime();
            delta_t_seconds = time - frame_start_time_seconds;
            frame_start_time_seconds = time;
        }

        glfwPollEvents();
        if (glfwWindowShouldClose(window)) goto LABEL_EXIT_MAIN_LOOP;


        ivec2 prev_window_pos = window_pos_;
        glfwGetWindowPos(window, &window_pos_.x, &window_pos_.y);
        abortIfGlfwError();
        bool window_repositioned = prev_window_pos != window_pos_;

        ivec2 prev_window_size = window_size_;
        glfwGetWindowSize(window, &window_size_.x, &window_size_.y);
        abortIfGlfwError();

        bool window_resized = prev_window_size != window_size_;
        window_or_surface_out_of_date_ |= window_resized;


        if (window_or_surface_out_of_date_) {

            res = gfx::updateSurfaceResources(
                gfx_surface,
                VkExtent2D { (u32)window_size_.x, (u32)window_size_.y }
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

        int w_key_state = glfwGetKey(window, GLFW_KEY_W);
        int s_key_state = glfwGetKey(window, GLFW_KEY_S);
        int a_key_state = glfwGetKey(window, GLFW_KEY_A);
        int d_key_state = glfwGetKey(window, GLFW_KEY_D);
        int space_key_state = glfwGetKey(window, GLFW_KEY_SPACE);
        int lshift_key_state = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
        abortIfGlfwError();

        vec3 camera_vel = vec3(0);
        {
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
        checkedGlfwGetCursorPos(window, &cursor_pos_.x, &cursor_pos_.y);

        // If window is resized or moved, the virtual cursor position reported by glfw changes even if the
        // user did not move it. We must ignore the cursor position change in this case.
        // TODO this doesn't seem to have actually solved the problem
        if (window_resized || window_repositioned) prev_cursor_pos = cursor_pos_;

        // compute new camera direction
        {
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

            // In GLM normalized device coordinates, by default:
            //     1. The Y axis points upward.
            //     2. The Z axis has range [-1, 1] and points out of the screen.
            // In Vulkan normalized device coordinates, by default:
            //     1. The Y axis points downward.
            //     2. The Z axis has range [0, 1] and points into the screen.
            mat4 glm_to_vulkan = glm::identity<mat4>();
            glm_to_vulkan[1][1] = -1.0f; // mirror y axis
            glm_to_vulkan[2][2] = -0.5f; // mirror and halve z axis: [1, -1] -> [-0.5, 0.5]
            glm_to_vulkan[3][2] = 0.5f; // shift z axis forward: [-0.5, 0.5] -> [0, 1]

            world_to_screen_transform = world_to_camera_transform * world_to_screen_transform;
            world_to_screen_transform = camera_to_clip_transform * world_to_screen_transform;
            world_to_screen_transform = glm_to_vulkan * world_to_screen_transform;
        }

        gfx::CameraInfo camera_info {
            .camera_direction_unit = camera_direction_unit,
            .camera_right_direction_unit = camera_horizontal_right_direction_unit,
            .camera_up_direction_unit = camera_y_axis_unit,
            .eye_pos = camera_pos_,
            .viewport_size = vec2(window_draw_region_.extent.width, window_draw_region_.extent.height),
            .frustum_near_side_size = VIEW_FRUSTUM_NEAR_SIDE_SIZE,
            .frustum_near_side_distance = (f32)VIEW_FRUSTUM_NEAR_SIDE_DISTANCE,
            .frustum_far_side_distance = (f32)VIEW_FRUSTUM_FAR_SIDE_DISTANCE,
        };
        gfx::RenderResult render_result = gfx::render(
            gfx_surface,
            window_draw_region_,
            &world_to_screen_transform,
            &camera_info
        );

        switch (render_result) {
            case gfx::RenderResult::error_surface_resources_out_of_date:
            case gfx::RenderResult::success_surface_resources_out_of_date:
            {
                LOG_F(INFO, "Surface resources out of date.");
                window_or_surface_out_of_date_ = true;
                goto LABEL_MAIN_LOOP_START;
            }
            case gfx::RenderResult::success: break;
        }

        frame_counter++;
    };

    LABEL_EXIT_MAIN_LOOP: {}


    glfwTerminate();
    exit(0);
}
