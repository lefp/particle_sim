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

#include "types.hpp"
#include "error_util.hpp"
#include "graphics.hpp"

namespace gfx = graphics;

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

//
// Global constants ==========================================================================================
//

const char* APP_NAME = "an game";

const VkExtent2D DEFAULT_WINDOW_EXTENT { 800, 600 };

const double ASPECT_RATIO = 16.0 / 9.0;

//
// Global variables ==========================================================================================
//

vec3 camera_pos_ { 0, 0, 0 };
vec3 camera_direction_unit_ { 1, 0, 0 };

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

    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(gfx::getVkInstance(), window, NULL, &vk_surface);
    assertVk(result);

    int current_window_width = 0;
    int current_window_height = 0;
    glfwGetWindowSize(window, &current_window_width, &current_window_height);
    abortIfGlfwError();
    // TODO if current_width == current_height == 0, check if window is minimized or something; if it is, do
    // something that doesn't waste resources

    gfx::SurfaceResources gfx_surface {};
    res = gfx::createSurfaceResources(
        vk_surface,
        VkExtent2D { .width = (u32)current_window_width, .height = (u32)current_window_height },
        &gfx_surface
    );
    // TODO handle error_window_size_zero
    assertGraphics(res);


    gfx::attachSurfaceToRenderer(gfx_surface, gfx_renderer);


    u32fast frame_counter = 0;

    while (true) {

        LABEL_RENDER_LOOP_START: {}

        glfwPollEvents();
        if (glfwWindowShouldClose(window)) break;


        vec3 camera_horizontal_left_direction_unit = vec3(
            camera_direction_unit_.z,
            0,
            -camera_direction_unit_.x
        );
        vec3 camera_y_axis = vec3(0, 1, 0);

        int w_key_state = glfwGetKey(window, GLFW_KEY_W);
        int s_key_state = glfwGetKey(window, GLFW_KEY_S);
        int a_key_state = glfwGetKey(window, GLFW_KEY_A);
        int d_key_state = glfwGetKey(window, GLFW_KEY_D);
        int space_key_state = glfwGetKey(window, GLFW_KEY_SPACE);
        int lshift_key_state = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
        abortIfGlfwError();

        // TODO do the delta_t thing here
        if (w_key_state == GLFW_PRESS) camera_pos_ += 0.01f * camera_direction_unit_;
        if (s_key_state == GLFW_PRESS) camera_pos_ -= 0.01f * camera_direction_unit_;
        if (a_key_state == GLFW_PRESS) camera_pos_ += 0.01f * camera_horizontal_left_direction_unit;
        if (d_key_state == GLFW_PRESS) camera_pos_ -= 0.01f * camera_horizontal_left_direction_unit;
        if (space_key_state == GLFW_PRESS) camera_pos_.y += 0.01f;
        if (lshift_key_state == GLFW_PRESS) camera_pos_.y -= 0.01f;


        mat4 world_to_screen_transform = glm::identity<mat4>();
        {
            mat4 world_to_camera_transform = glm::lookAt(
                camera_pos_, // eye
                camera_pos_ + camera_direction_unit_, // position you're looking at
                camera_y_axis // "Normalized up vector, how the camera is oriented."
            );
            mat4 camera_to_clip_transform = glm::perspective(
                (f32)(0.25*M_PI * ASPECT_RATIO), // fovy
                (f32)ASPECT_RATIO, // aspect
                0.15f, // zNear
                500.0f // zFar
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

        gfx::RenderResult render_result = gfx::render(gfx_surface, &world_to_screen_transform);

        switch (render_result) {
            case gfx::RenderResult::error_surface_resources_out_of_date:
            case gfx::RenderResult::success_surface_resources_out_of_date:
            {
                current_window_width = 0;
                current_window_height = 0;
                glfwGetWindowSize(window, &current_window_width, &current_window_height);
                abortIfGlfwError();

                LOG_F(INFO, "Surface resources out of date; updating.");
                res = gfx::updateSurfaceResources(
                    gfx_surface,
                    VkExtent2D { .width = (u32)current_window_width, .height = (u32)current_window_height }
                );

                assertGraphics(res);
                goto LABEL_RENDER_LOOP_START;
            }
            case gfx::RenderResult::success: {}
        }

        frame_counter++;
    };


    glfwTerminate();
    exit(0);
}
