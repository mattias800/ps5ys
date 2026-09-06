/* Optional wrapper around the existing indexed-draw oracle. Not linked into prosper. */
#include <dlfcn.h>
#include <renderdoc_app.h>

static RENDERDOC_API_1_6_0* capture_api;

static void capture_begin(const char* path) {
    pRENDERDOC_GetAPI get_api = (pRENDERDOC_GetAPI)dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI");
    require(get_api && get_api(eRENDERDOC_API_Version_1_6_0, (void**)&capture_api),
            "RenderDoc API unavailable; launch with renderdoccmd capture");
    capture_api->SetCaptureFilePathTemplate(path);
    capture_api->SetCaptureOptionU32(eRENDERDOC_Option_RefAllResources, 1);
    capture_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(g_instance), NULL);
    require(capture_api->IsFrameCapturing(), "RenderDoc did not arm the control capture");
}

static void capture_end(void) {
    require(capture_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(g_instance), NULL),
            "RenderDoc did not finish the control capture");
    require(capture_api->GetNumCaptures() == 1, "expected exactly one capture");
    uint32_t size = 0;
    require(capture_api->GetCapture(0, NULL, &size, NULL) && size > 0,
            "capture path unavailable");
    char* path = malloc(size + 1);
    require(path != NULL, "capture path allocation failed");
    require(capture_api->GetCapture(0, path, &size, NULL), "capture path query failed");
    printf("RENDERDOC_CONTROL_CAPTURE=%s\n", path);
    free(path);
}
