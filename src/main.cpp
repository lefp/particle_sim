#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cinttypes>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <loguru/loguru.hpp>

#include "types.hpp"
#include "error_util.hpp"
namespace graphics {
    #include "graphics.hpp"
}

//
// Global constants ==========================================================================================
//

const char* APP_NAME = "an game";

const VkExtent2D DEFAULT_WINDOW_EXTENT { 800, 600 };

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


int main(int argc, char** argv) {

    loguru::init(argc, argv);
    #ifndef NDEBUG
        LOG_F(INFO, "Debug build.");
    #else
        LOG_F(INFO), "Release build.");
    #endif

    int success = glfwInit();
    assertGlfw(success);


    const char* specific_device_request = getenv("PHYSICAL_DEVICE_NAME");
    graphics::init(APP_NAME, specific_device_request);


    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // don't initialize OpenGL, because we're using Vulkan
    GLFWwindow* window = glfwCreateWindow(
        (int)DEFAULT_WINDOW_EXTENT.width, (int)DEFAULT_WINDOW_EXTENT.height, "an game", NULL, NULL
    );
    assertGlfw(window != NULL);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(instance_, window, NULL, &surface);
    assertVk(result);

    int current_window_width = 0;
    int current_window_height = 0;
    glfwGetWindowSize(window, &current_window_width, &current_window_height);
    abortIfGlfwError();
    // TODO if current_width == current_height == 0, check if window is minimized or something; if it is, do
    // something that doesn't waste resources


    VkImage p_swapchain_images[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkImageView p_swapchain_image_views[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkFramebuffer p_simple_render_pass_swapchain_framebuffers[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer p_command_buffers[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkFence p_command_buffer_pending_fences[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkSemaphore p_render_finished_semaphores[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkSemaphore p_swapchain_image_acquired_semaphores[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    u32 swapchain_image_count = 0;
    {
        swapchain_image_count = getSwapchainImages(
            device_, swapchain, MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT, p_swapchain_images
        );
        if (swapchain_image_count > MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT) ABORT_F(
            "Unexpectedly large swapchain image count; assumed at most %" PRIu32 ", actually %" PRIu32 ".",
            MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT, swapchain_image_count
        );

        success = createImageViewsForSwapchain(
            device_, swapchain_image_count, p_swapchain_images, p_swapchain_image_views
        );
        alwaysAssert(success);

        success = createFramebuffersForSwapchain(
            device_, render_pass, swapchain_extent, swapchain_image_count, p_swapchain_image_views,
            p_simple_render_pass_swapchain_framebuffers
        );
        alwaysAssert(success);


        const VkCommandPoolCreateInfo command_pool_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_,
        };

        result = vk_dev_procs.createCommandPool(device_, &command_pool_info, NULL, &command_pool);
        assertVk(result);

        VkCommandBufferAllocateInfo command_buffer_alloc_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = swapchain_image_count,
        };

        result = vk_dev_procs.allocateCommandBuffers(device_, &command_buffer_alloc_info, p_command_buffers);
        assertVk(result);


        VkFenceCreateInfo fence_info {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        for (u32 im_idx = 0; im_idx < swapchain_image_count; im_idx++) {
            result = vk_dev_procs.createFence(
                device_, &fence_info, NULL, &p_command_buffer_pending_fences[im_idx]
            );
            assertVk(result);
        }

        VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (u32 im_idx = 0; im_idx < swapchain_image_count; im_idx++) {
            result = vk_dev_procs.createSemaphore(
                device_, &semaphore_info, NULL, &p_render_finished_semaphores[im_idx]
            );
            assertVk(result);
        }
        for (u32 im_idx = 0; im_idx < swapchain_image_count; im_idx++) {
            result = vk_dev_procs.createSemaphore(
                device_, &semaphore_info, NULL, &p_swapchain_image_acquired_semaphores[im_idx]
            );
            assertVk(result);
        }
    }


    bool swapchain_needs_rebuild = false;
    u32fast frame_counter = 0;

    glfwPollEvents();
    while (!glfwWindowShouldClose(window)) {

        // TODO Make sure this makes sense, and if so, write down why it makes sense here.
        VkSemaphore swapchain_image_acquired_semaphore =
            p_swapchain_image_acquired_semaphores[frame_counter % swapchain_image_count];

        // Acquire swapchain image; if out of date, rebuild swapchain until it works. ------------------------

        u32fast swapchain_rebuild_count = 0;
        u32 acquired_swapchain_image_idx = UINT32_MAX;
        while (true) {

            if (swapchain_needs_rebuild) {

                int window_width = 0;
                int window_height = 0;
                glfwGetWindowSize(window, &window_width, &window_height);
                abortIfGlfwError();
                // TODO if current_width == current_height == 0, check if window is minimized or something;
                // if it is, do something that doesn't waste resources

                VkSwapchainKHR old_swapchain = swapchain;
                swapchain = createSwapchain(
                    physical_device_, device_, surface,
                    VkExtent2D { .width = (u32)window_width, .height = (u32)window_height },
                    queue_family_, present_mode_, swapchain, &swapchain_extent
                );
                alwaysAssert(swapchain != VK_NULL_HANDLE);

                swapchain_rebuild_count++;

                // Before destroying the old swapchain, make sure it's not in use.
                result = vk_dev_procs.queueWaitIdle(queue_);
                assertVk(result);
                vk_dev_procs.destroySwapchainKHR(device_, old_swapchain, NULL);
            }

            // TODO: (2023-11-20) The only purpose of this block is to work around this validation layer bug:
            // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/6068
            // Remove this after that bug is fixed.
            {
                u32 im_count = MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT;
                result = vk_dev_procs.getSwapchainImagesKHR(device_, swapchain, &im_count, p_swapchain_images);
            }

            result = vk_dev_procs.acquireNextImageKHR(
                device_, swapchain, UINT64_MAX, swapchain_image_acquired_semaphore, VK_NULL_HANDLE,
                &acquired_swapchain_image_idx
            );
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                swapchain_needs_rebuild = true;
                continue;
            }

            // If the swapchain is Suboptimal, we will rebuild it _after_ this frame. This is because
            // acquireNextImageKHR succeeded, so it may have signaled `swapchain_image_acquired_semaphore`;
            // we must not call it again with the same semaphore until the semaphore is unsignaled.
            if (result == VK_SUBOPTIMAL_KHR) swapchain_needs_rebuild = true;
            else {
                assertVk(result);
                swapchain_needs_rebuild = false;
            }
            break;
        };

        if (swapchain_rebuild_count > 0) {

            LOG_F(INFO, "Swapchain rebuilt (%" PRIuFAST32 " times).", swapchain_rebuild_count);

            // Before destroying resources, make sure they're not in use.
            result = vk_dev_procs.queueWaitIdle(queue_);
            assertVk(result);


            result = vk_dev_procs.resetCommandPool(device_, command_pool, 0);
            assertVk(result);

            for (u32 i = 0; i < swapchain_image_count; i++) vk_dev_procs.destroyFramebuffer(
                device_, p_simple_render_pass_swapchain_framebuffers[i], NULL
            );

            for (u32 i = 0; i < swapchain_image_count; i++) vk_dev_procs.destroyImageView(
                device_, p_swapchain_image_views[i], NULL
            );


            swapchain_roi = centeredSubregion_16x9(swapchain_extent);

            u32 old_image_count = swapchain_image_count;
            swapchain_image_count = getSwapchainImages(
                device_, swapchain, MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT, p_swapchain_images
            );
            // I don't feel like handling a changing number of swapchain images.
            // We'd have to change the number of command buffers, fences, semaphores.
            alwaysAssert(swapchain_image_count == old_image_count);

            success = createImageViewsForSwapchain(
                device_, swapchain_image_count, p_swapchain_images, p_swapchain_image_views
            );
            alwaysAssert(success);

            success = createFramebuffersForSwapchain(
                device_, render_pass, swapchain_extent, swapchain_image_count, p_swapchain_image_views,
                p_simple_render_pass_swapchain_framebuffers
            );
            alwaysAssert(success);
        }

        // ---------------------------------------------------------------------------------------------------

        VkCommandBuffer command_buffer = p_command_buffers[acquired_swapchain_image_idx];
        VkFence command_buffer_pending_fence = p_command_buffer_pending_fences[acquired_swapchain_image_idx];
        VkSemaphore render_finished_semaphore = p_render_finished_semaphores[acquired_swapchain_image_idx];


        result = vk_dev_procs.waitForFences(device_, 1, &command_buffer_pending_fence, VK_TRUE, UINT64_MAX);
        assertVk(result);
        result = vk_dev_procs.resetFences(device_, 1, &command_buffer_pending_fence);
        assertVk(result);


        vk_dev_procs.resetCommandBuffer(command_buffer, 0);

        mat4 world_to_screen_transform = glm::identity<mat4>();
        {
            f32 angle_radians = (f32) ( 2.0*M_PI * (1.0/150.0)*fmod((f64)frame_counter, 150.0) );
            vec3 rotation_axis = glm::normalize(vec3(cos(angle_radians), sin(angle_radians), -cos(angle_radians)));

            f32 camera_distance = 3;
            vec3 camera_pos = camera_distance * glm::normalize(vec3(1, -1, 1));
            {
                mat4 rot_mat = glm::rotate(glm::identity<mat4>(), angle_radians, rotation_axis);
                vec4 cam_pos = rot_mat * vec4(camera_pos, 1.0);
                camera_pos = vec3(cam_pos); // drops 4th component
            }

            mat4 world_to_camera_transform = glm::lookAt(
                camera_pos, // eye
                vec3(0, 0, 0), // position you're looking at
                rotation_axis // "Normalized up vector, how the camera is oriented."
            );
            mat4 camera_to_clip_transform = glm::perspective(
                (f32)(0.25*M_PI) * ASPECT_RATIO, // fovy
                ASPECT_RATIO, // aspect
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
        VoxelPipelineVertexShaderPushConstants voxel_pipeline_push_constants {
            .transform = world_to_screen_transform
        };

        success = recordVoxelCommandBuffer(
            command_buffer,
            render_pass,
            pipelines_.voxel_pipeline,
            pipeline_layouts_.voxel_pipeline_layout,
            swapchain_extent,
            swapchain_roi,
            p_simple_render_pass_swapchain_framebuffers[acquired_swapchain_image_idx],
            &voxel_pipeline_push_constants
        );
        alwaysAssert(success);


        const VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &swapchain_image_acquired_semaphore,
            .pWaitDstStageMask = &wait_dst_stage_mask,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished_semaphore,
        };
        result = vk_dev_procs.queueSubmit(queue_, 1, &submit_info, command_buffer_pending_fence);
        assertVk(result);


        VkPresentInfoKHR present_info {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &acquired_swapchain_image_idx,
        };
        result = vk_dev_procs.queuePresentKHR(queue_, &present_info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR or result == VK_SUBOPTIMAL_KHR) swapchain_needs_rebuild = true;
        else assertVk(result);

        frame_counter++;
        glfwPollEvents();
    };


    glfwTerminate();
    exit(0);
}
