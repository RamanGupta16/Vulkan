//https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Base_code

#define GLFW_INCLUDE_VULKAN
#define VK_ENABLE_BETA_EXTENSIONS
#include <GLFW/glfw3.h> // GLFW will include its own definitions and automatically load the Vulkan header with it.

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

//#include <vulkan/vulkan.h>
#include <array>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2; // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Frames_in_flight

static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

static std::vector<char> readFile(std::string filename) {

    // Flags:
    // ate: Start reading at the end of the file
    // binary: Read the file as binary file (avoid text transformations)
    // The advantage of starting to read at the end of the file (ate flag) is that we
    // can use the read position to determine the size of the file and allocate a buffer:
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::stringstream s; s << "failed to open file=" << filename;
        throw std::runtime_error(s.str());
    }
    
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    
    file.close();
    std::cout << "Load file=" << filename << " of size=" << buffer.size() << std::endl;
    return buffer;
}

// Validation Layer
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// required device extensions
const std::vector<const char*> deviceExtensions = {
    // Vulkan does not have the concept of a "default framebuffer", hence it requires an
    // infrastructure that will own the buffers we will render to before we visualize them on the screen.
    // This infrastructure is known as the swap chain and must be created explicitly in Vulkan.
    // The swap chain is a queue of images that are waiting to be presented to the screen.
    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Presentation/Swap_chain
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

/*
NOT DONE
1) Debugging instance creation and destruction : https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers
 */
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    std::stringstream s;
    if(messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) s << "VERBOSE";
    else if(messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) s << "INFO";
    else if(messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) s << "WARNING";
    else if(messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) s << "ERROR";
    
    std::cerr << s.str() << " Validation Layer debugCallback message : " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

//vkCreateDebugUtilsMessengerEXT function is an extension function, it is not automatically loaded. We need to lookup and load this function
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

//vkDestroyDebugUtilsMessengerEXT function is an extension function, it is not automatically loaded. We need to lookup its address and load this function
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// commands to be submitted to a queue. There are different types of queues that originate from different queue
// families and each family of queues allows only a subset of commands.
// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // Queue Family Index
    std::optional<uint32_t> presentFamily; // Image presentation support
    
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// Swap Chain Properties
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Vertex data
// https://vulkan-tutorial.com/en/Vertex_buffers/Vertex_input_description
struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    
    
    /* tell Vulkan how to pass vertex data format to the vertex shader once it's been uploaded into GPU memory. */

    // Binding descriptions
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // per-vertex data

        return bindingDescription;
    }

    // Attribute descriptions
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        // Vertex shader layout(location = 0) in vec2 inPosition
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; // vec2: VK_FORMAT_R32G32_SFLOAT
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        // Vertex shader layout(location = 1) in vec3 inColor
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3: VK_FORMAT_R32G32B32_SFLOAT
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }

    
    
};

// interleaving vertex attributes
/* Same as combined
vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5) );
vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0) );
*/
const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

class HelloTriangleApplication {

public:
    void run() {
        initWindow(); // Initialise GLFW library and Window
        initVulkan();
        mainLoop();
        cleanup();
    }
    
     void setFramebufferResized() {
        framebufferResized = true;
        std::cout << "FramebufferResized" << std::endl;
    }
    
private:

    void initVulkan() {
        createInstance(); // initialize the Vulkan library by creating an instance.
        setupDebugMessenger(); // setup Validation Layer
        createSurface(); // Create Surface using GLFW
        pickPhysicalDevice(); // Physical devices and queue families
        createLogicalDevice();
        createSwapChain();
        createImageViews(); // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions
        createRenderPass(); // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
        createGraphicsPipeline(); //
        createFramebuffers();
        createCommandPool(); // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Command_buffers
        createVertexBuffer(); // https://vulkan-tutorial.com/en/Vertex_buffers/Vertex_buffer_creation
        createCommandBuffers();
        createSyncObjects();
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        std::cout << "Created Surface" << std::endl;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            
            // look for a queue family that has the capability of presenting to our window surface.
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }
        return indices;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        
        float queuePriority = 1.0f;
        for(uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        VkPhysicalDeviceFeatures deviceFeatures{};
        
        //Creating the logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
        
        // We can use the vkGetDeviceQueue function to retrieve queue handles for each queue family.
        // The parameters are the logical device, queue family, queue index and a pointer to the
        // variable to store the queue handle in. Because we're only creating a single queue from
        // this family, we'll simply use index 0
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }
    
    bool isDeviceSuitable(VkPhysicalDevice device) {
        /*VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
           deviceFeatures.geometryShader;*/
           
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            
            // Swap chain support is sufficient for this tutorial if there is at least one supported image format
            // and one supported presentation mode given the window surface we have.
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        
        // query the new window resolution to make sure that the swap chain images have the latest size
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        
        // The implementation specifies the minimum number that it requires to function.
        // However, simply sticking to this minimum means that we may sometimes have to wait
        // on the driver to complete internal operations before we can acquire another image to render to.
        // Therefore it is recommended to request at least one more image than the minimum:
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; // 2 + 1 = 3
        
        // Do not exceed the maximum number of images while doing this, where 0 is a special value that means that there is no maximum:
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }
        
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
        
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        std::cout << "SwapChain created" << std::endl;
    }

    // An image view is quite literally a view into an image.
    // It describes how to access the image and which part of the image to access,
    // for example if it should be treated as a 2D texture depth texture without any mipmapping levels.
    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    // tell Vulkan about the framebuffer attachments that will be used while rendering.
    void createRenderPass() {
        /* Render pass: the attachments referenced by the pipeline stages and their usage */
        // Attachment description
        // attachments are descriptions of image resources used during render pass.
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        
        // Subpasses and attachment references
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0; // specifies which attachment to reference by its index in the attachment descriptions array.
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef; // The index of the attachment in this array is directly referenced from the fragment shader

        // Subpass dependencies: https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules
        // The compilation and linking of the SPIR-V bytecode to machine code for execution by
        // the GPU doesn't happen until the graphics pipeline is created.
        // That means that we're allowed to destroy the shader modules again as soon as pipeline creation is finished,
        // which is why we'll make them local variables in the createGraphicsPipeline function instead of class members
        
        /* Shader stages: the shader modules that define the functionality of the programmable stages of the graphics pipeline */
        
        // May need to Adjust the shader path
        std::string filename {"/Users/ramangup/Code/try_vulkan/shaders/vert.spv"};
        auto vertShaderCode = readFile(filename);
        
        filename = "/Users/ramangup/Code/try_vulkan/shaders/frag.spv";
        auto fragShaderCode = readFile(filename);
        
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        
        // shaders we'll need to assign them to a specific pipeline stage
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data() ;

        /* Fixed-function state: all of the structures that define the fixed-function stages of the pipeline, like input assembly, rasterizer, viewport and color blendin */
        // Input assembly: We intend to draw triangles throughout this tutorial
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        /*
        // size of the swap chain and its images may differ from the WIDTH and HEIGHT of the window.
        // The swap chain images will be used as framebuffers later on, so we should stick to their size.
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        
        // Any pixels outside the scissor rectangles will be discarded by the rasterizer.
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        */
        
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

    
        // The Rasterizer takes the geometry that is shaped by the vertices from the vertex shader
        // and turns it into fragments to be colored by the fragment shader.
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE; // If VK_TRUE, then geometry never passes through rasterizer stage. This basically disables output to framebuffer.
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // fill the area of the polygon with fragments
        rasterizer.lineWidth = 1.0f; // thickness of lines in terms of number of fragments.
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
        
        // Multisampling Anti Aliasing
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional
        
        // Color blending : fragment colors will be written to the framebuffer unmodified.
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional
        
        /* Pipeline layout: the uniform and push values referenced by the shader that can be updated at draw time */
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        
        // Pipeline layout : uniform values need to be specified during pipeline creation by creating a VkPipelineLayout object.
        // These uniform values need to be specified during pipeline creation by creating a VkPipelineLayout object.
        // Even though we won't be using them until a future chapter, we are still required to create an empty pipeline layout.
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0; // Optional
        pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = nullptr; // Optional
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        
        std::cout << "Created Graphics Pipeline" << std::endl;
    }

    /*void createComputePipeline() {
        auto computeShaderCode = readFile("shaders/comp.spv");

        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = nullptr; //&computeDescriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }*/
    
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        
        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules
        // The one catch is that the size of the bytecode is specified in bytes,
        // but the bytecode pointer is a uint32_t pointer rather than a char pointer.
        // Therefore we will need to cast the pointer with reinterpret_cast as shown below.
        // When you perform a cast like this, you also need to ensure that the data satisfies the alignment requirements of uint32_t.
        // Lucky for us, the data is stored in an std::vector where the default allocator already ensures that the data satisfies the worst case alignment requirements.
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Framebuffers
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // Commands in Vulkan, like drawing operations and memory transfers, are not executed directly using function calls.
    // You have to record all of the operations you want to perform in command buffer objects.
    // The advantage of this is that when we are ready to tell the Vulkan what we want to do,
    // all of the commands are submitted together and Vulkan can more efficiently process the commands
    // since all of them are available together. In addition, this allows command recording to happen in multiple threads if so desired.
    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Command_buffers
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        // We have to create a command pool before we can create command buffers.
        // Command pools manage the memory that is used to store the buffers and command buffers are allocated from them.
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        // Command buffers are executed by submitting them on one of the device queues,
        // like the graphics and presentation queues we retrieved.
        // Each command pool can only allocate command buffers that are submitted on a single type of queue.
        // We're going to record commands for drawing, which is why we've chosen the graphics queue family.
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }
    
    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Command_buffers
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();;

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
        
        std::cout << "createCommandBuffer" << std::endl;
    }

    // writes the commands we want to execute into a command buffer.
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex /*index of the current swapchain image we want to write to*/) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; // We created a framebuffer for each swap chain image where it is specified as a color attachment.
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        // clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // Drawing starts by beginning the render pass with vkCmdBeginRenderPass
        // VK_SUBPASS_CONTENTS_INLINE : render pass commands will be embedded in the primary command buffer itself and no secondary command buffers will be executed.
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Basic drawing commands
        // All of the functions that record commands can be recognized by their vkCmd prefix.
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // Binding the vertex buffer
        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            
        // draw command for the triangle:
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
        
        //std::cout << "Recorded command buffer for swap chain imageIndex=" << imageIndex << std::endl;
    }
    
    // https://vulkan-tutorial.com/en/Vertex_buffers/Vertex_buffer_creation
    // GPU Memory Allocation
    // Buffers in Vulkan are regions of memory used for storing arbitrary data that can be read by the graphics card
    void createVertexBuffer() {
    
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // only used from the graphics queue and not shared between multiple queues

        // The flags parameter is used to configure sparse buffer memory, which is not relevant right now. Leave it at default 0.
        //bufferInfo.flags = 0;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        // The first step of allocating memory for the buffer is to query its memory requirements
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        std::cout << "VERTEX_BUFFER allocationSize=" << memRequirements.size << std::endl;
        
        // Allocate vertexBuffer memory
        if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }

        // associate this vertexBufferMemory with the vertexBuffer
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

        // Memory-mapped I/O
        void* data;
        vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferInfo.size);
        
        vkUnmapMemory(device, vertexBufferMemory);

        /*
        Unfortunately the driver may not immediately copy the data into the buffer memory, for example because of caching.
        It is also possible that writes to the buffer are not visible in the mapped memory yet.
        
        There are two ways to deal with that problem:
        Use a memory heap that is host coherent, indicated with VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        Call vkFlushMappedMemoryRanges after writing to the mapped memory, and call vkInvalidateMappedMemoryRanges before reading from the mapped memory.
        
        We went for the first approach, which ensures that the mapped memory always matches the contents of the allocated memory.
        Do keep in mind that this may lead to slightly worse performance than explicit flushing, but we'll see why that doesn't matter in the next chapter.

        Flushing memory ranges or using a coherent memory heap means that the driver will be aware of our writes to the buffer,
        but it doesn't mean that they are actually visible on the GPU yet. The transfer of data to the GPU is an operation that
        happens in the background and the specification simply tells us that it is guaranteed to be complete as of the next call to vkQueueSubmit.
        */
    }
    
    // Graphics cards can offer different types of memory to allocate from.
    // Each type of memory varies in terms of allowed operations and performance characteristics.
    // We need to combine the requirements of the buffer and our own application requirements to find the right type of memory to use.
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        // The different types of memory exist within these heaps. Right now we'll only concern ourselves with
        // the type of memory and not the heap it comes from, but you can imagine that this can affect performance.
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
    
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // sRGB
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        // The swap chain is a queue where the display takes an image from the front of the queue when
        // the display is refreshed and the program inserts rendered images at the back of the queue.
        // If the queue is full then the program has to wait. This is most similar to vertical sync as found in modern games.
        // The moment that the display is refreshed is known as "vertical blank".
        // Default
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    
    // The swap extent is the resolution of the swap chain images and it's almost always exactly equal
    // to the resolution of the window that we're drawing to in pixels
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
        
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;
        std::cout << " validation layer setupDebugMessenger" << std::endl;
        
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // Optional
        
        if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up validation layer debug messenger!");
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        
        // wait for the logical device to finish operations before exiting mainLoop and destroying the window:
        vkDeviceWaitIdle(device);
    }

    /*
    Synchronization:
    A core design philosophy in Vulkan is that synchronization of execution on the GPU is explicit.
    The order of operations is up to us to define using various synchronization primitives which tell
    the driver the order we want things to run in. This means that many Vulkan API calls which start
    executing work on the GPU are asynchronous, the functions will return before the operation has finished.
    https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation
     */
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(swapChainImages.size()); // see https://github.com/Overv/VulkanTutorial/issues/407
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    
        // A semaphore is used to add order between queue operations.
        // Queue operations refer to the work we submit to a queue, either in a command buffer or
        // from within a function as we will see later. Examples of queues are the graphics queue and the presentation queue.
        // Semaphores are used both to order work inside the same queue and between different queues.
        // Function returns and the waiting only happens on the GPU. The CPU continues running without blocking.
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // A fence has a similar purpose, in that it is used to synchronize execution,
        // but it is for ordering the execution on the CPU, otherwise known as the host.
        // Simply put, if the host needs to wait or know when the GPU has finished something, we use a fence.
        // we can make the host (CPU) wait for the fence to be signaled, guaranteeing that the work has finished before the host continues.
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // Create the fence in the signaled state, so that the first call to vkWaitForFences() returns immediately since the fence is already signaled.
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                //vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
        
        // // see https://github.com/Overv/VulkanTutorial/issues/407
        for (size_t i = 0; i < swapChainImages.size(); i++) {
         if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
           throw std::runtime_error(
               "failed to create graphics synchronization objects for a frame!");
         }
    }
        
        // In summary, semaphores are used to specify the execution order of operations on the GPU while fences are used to keep the CPU and GPU in sync with each-other.
    }

    /*
    At a high level, rendering a frame in Vulkan consists of a common set of steps:
    1. Wait for the previous frame to finish
    2. Acquire an image from the swap chain
    3. Record a command buffer which draws the scene onto that image
    4. Submit the recorded command buffer
    5. Present the swap chain image
    While we will expand the drawing function in later chapters, for now this is the core of our render loop.
    https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation
     */
    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquiring an image from the swap chain
        // swap chain is an extension feature, so we must use a function with the vk*KHR naming convention
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        // Suboptimal or out-of-date swap chain : https://vulkan-tutorial.com/en/Drawing_a_triangle/Swap_chain_recreation
        if (result == VK_ERROR_OUT_OF_DATE_KHR) { // possibly on window resize
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Only reset the fence if we are submitting work
        vkResetFences(device, 1, &inFlightFences[currentFrame]); // vkResetFences resets the fence to the unsignaled state.

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // Queue submission and synchronization is configured through parameters in the VkSubmitInfo structure.
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages; //Each entry in the waitStages array corresponds to the semaphore with the same index in pWaitSemaphores.

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]}; // See https://github.com/Overv/VulkanTutorial/issues/407
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // Presentation : The last step of drawing a frame is submitting the result back to the swap chain to have it eventually show up on the screen.
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional

        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Swap_chain_recreation
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    
    void cleanup() {
        std::cout << "cleanup" << std::endl;
        
        // cleanup swapChainFramebuffers, swapChainImageViews, v
        cleanupSwapChain();
        
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            //vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);  // See https://github.com/Overv/VulkanTutorial/issues/407
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        
        vkDestroyRenderPass(device, renderPass, nullptr);
        
        vkDestroyDevice(device, nullptr);
        
        vkDestroySurfaceKHR(instance, surface, nullptr);
        
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        
        vkDestroyInstance(instance, nullptr); // createInstance
        glfwDestroyWindow(window); // initWindow
        glfwTerminate();
    }
    
    void initWindow() {
        std::cout << "initWindow" << std::endl;
        
        // Initialize the GLFW library
        if (!glfwInit())
            return;
            
        // Because GLFW was originally designed to create an OpenGL context, we need to tell it to not create an OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        // detect window resize callback
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // Validation Layer Message callback
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        //extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        return extensions;
    }

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Swap_chain_recreation
    void cleanupSwapChain() {
        // we don't recreate the renderpass here for simplicity.

        for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Swap_chain_recreation
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        
        // Window minimization. This case is special because it will result in a frame buffer size of 0
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
    
        vkDeviceWaitIdle(device);

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void createInstance() {
        std::cout << "create VkInstance" << std::endl;
        // Initialize the Vulkan library by creating an instance.
        // The instance is the connection between your application and the Vulkan library
        
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        
        /***********************************************************************/
        
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        std::vector<const char*> requiredExtensions = getRequiredExtensions();
    
        createInfo.enabledExtensionCount = requiredExtensions.size();
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        createInfo.enabledLayerCount = 0;
        
        /***********************************************************************/
        
        // Check Validation Layers
        if(enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        
        if(enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        /***********************************************************************/
        
        // create new Vulkan instance
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            if(result != VK_ERROR_INCOMPATIBLE_DRIVER)
            {
                std::cerr << "failed to create instance Error=" << result << std::endl;
                throw std::runtime_error("failed to create instance!");
            }
            else
            {
                std::cout << "Fixing VK_ERROR_INCOMPATIBLE_DRIVER and try to create instance again..." << std::endl;
            }
        }
        
        // On MacOS with the latest MoltenVK sdk, you may get VK_ERROR_INCOMPATIBLE_DRIVER(-9)
        // The VK_KHR_PORTABILITY_subset extension is mandatory.
        requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        
        createInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        
        result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            std::cerr << "failed to create instance Error=" << result << std::endl;
            throw std::runtime_error("failed to create instance!");
        }
        std::cout << "vkCreateInstance successfull" << std::endl;
        
        /***********************************************************************/
        
        // Available extensions
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        std::cout << "InstanceExtension Properties List ::" << std::endl;
        for(const auto& extension : extensions) {
            std::cout << extension.extensionName << " " << extension.specVersion << std::endl;
        }
    }
    
    bool checkValidationLayerSupport() {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

            for(const char* validationLayer : validationLayers) {
                bool layerFound = false;
                for(const VkLayerProperties& layerProperties : availableLayers)
                {
                    if(strcmp(validationLayer, layerProperties.layerName) == 0)
                    {
                        layerFound = true;
                        break;
                    }
                }
                return layerFound;
            }
            return true;
        }
    
private:
    GLFWwindow* window {nullptr};
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    
    // The window surface needs to be created right after the instance creation
    VkSurfaceKHR surface;
    
    // implicitly destroyed when the VkInstance is destroyed, so we won't need to do anything new in the cleanup function.
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    
    // Logical Device
    VkDevice device;
    
    // The queues are automatically created along with the logical device
    // Device queues are implicitly cleaned up when the device is destroyed
    VkQueue graphicsQueue;
    
    // Creating the presentation queue
    VkQueue presentQueue;
    
    // Vulkan does not have the concept of a "default framebuffer", hence it requires an infrastructure
    // that will own the buffers we will render to before we visualize them on the screen.
    // This infrastructure is known as the swap chain and must be created explicitly in Vulkan.
    VkSwapchainKHR swapChain;
    
    // handles of the VkImage in Swap chain
    // Textures and framebuffers in Vulkan are represented by VkImage objects with a certain pixel format,
    // however the layout of the pixels in memory can change based on what you're trying to do with an image.
    std::vector<VkImage> swapChainImages;
    
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    
    // To use any VkImage, including those in the swap chain, in the render pipeline we have to create a VkImageView object.
    // An image view is quite literally a view into an image.
    std::vector<VkImageView> swapChainImageViews;
    
    // we have to create a framebuffer for all of the images in the swap chain and use the one that corresponds to the retrieved image at drawing time.
    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Framebuffers
    std::vector<VkFramebuffer> swapChainFramebuffers;
    
    VkRenderPass renderPass; // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    
    bool framebufferResized = false;
};

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->setFramebufferResized();
}

int main() {
    HelloTriangleApplication app;
    
    try {
        app.run();
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
