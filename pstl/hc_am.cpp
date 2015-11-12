#include <map>

#include "hc_am.hpp"
#include "hsa.h"

namespace am {
struct memory_range {
    void *                  _base_pointer;
    size_t                  _size;
    hsa_agent_t             _hsa_agent;
    hsa_region_t            _hsa_region;


    memory_range(void *base_pointer, size_t size, hsa_agent_t hsa_agent, hsa_region_t hsa_region) :
       _base_pointer(base_pointer), _size(size), _hsa_agent(hsa_agent), _hsa_region(hsa_region) {}; 
    memory_range() : _base_pointer(NULL), _size(0) {};
};

struct context {
    std::map<void *, am::memory_range> memory_tracker;
};


static am::context g_context;
}

//#define TRACE
#ifdef TRACE
#define tprintf(...) fprintf(stderr,__VA_ARGS__)
#else 
#define tprintf(...) 
#endif




//=========================================================================================================
// API Definitions.
//=========================================================================================================
//
//

namespace hc {

// Allocate accelerator memory, return NULL if memory could not be allocated:
auto_voidp am_alloc(size_t size, unsigned flags, hc::accelerator_view av) 
{
    assert(flags == AM_EXPLICIT_SYNC); // TODO - support other flags.

    void *ptr = NULL;

#ifdef HCC_VERSION_08
    if (av.is_hsa_accelerator()) {
#else
    //TODO-kalmar - remove get_hsa_interop, this was old name for this function.
    if (av.get_hsa_interop()) {
#endif
        hsa_agent_t *hsa_agent = static_cast<hsa_agent_t*> (av.get_hsa_agent());
        hsa_region_t *am_region = static_cast<hsa_region_t*>(av.get_hsa_am_region());

        //TODO - how does AMP return errors?


        hsa_status_t s1 = HSA_STATUS_SUCCESS;
        hsa_status_t s2 = HSA_STATUS_SUCCESS;

        s1 = hsa_memory_allocate(*am_region, size, &ptr);
        s2 = hsa_memory_assign_agent(ptr, *hsa_agent, HSA_ACCESS_PERMISSION_RW);



        if ((s1 != HSA_STATUS_SUCCESS) || (s2 != HSA_STATUS_SUCCESS)) {
            ptr = NULL;
        }
        am::memory_range r(ptr, size, *hsa_agent, *am_region);
        am::g_context.memory_tracker[ptr] = r;
        tprintf ("hc_am: tracking %p sz=%zu\n", ptr, size);

    } else if (av.get_accelerator().get_is_emulated()) {
        // TODO - handle host memory allocation here?
    }

    return ptr;
};


am_status_t am_free(void* ptr) 
{
    if (ptr != NULL) {
        hsa_memory_free(ptr);

        size_t erased = am::g_context.memory_tracker.erase(ptr);

        //TODO
        if (erased == 0) {
            tprintf ("hc_am: error - am_free can't find pointer=%p\n", ptr);
        } else {
            tprintf ("hc_am: freeing %p\n", ptr);
        }
        
    }
    return AM_SUCCESS;
}


// Perform a copy src->dst.  Dst is left in current location and is 
// not assigned to a new accelerator cache.
am_status_t am_copy(void*  dst, const void*  src, size_t size)
{
    auto destMR = am::g_context.memory_tracker.find(dst);
    hsa_status_t err;

    if (destMR != am::g_context.memory_tracker.end()) {
        // Known pointer - use copy kernel?
        tprintf ("hc_am: copy_to tracked dst:  %p sz=%zu\n", dst, size);
        //
        err = hsa_memory_copy(dst, src, size);
    } else {
        // not found - must be host memory.
        tprintf ("hc_am: copy_to untracked  dst: %p sz=%zu\n", dst, size);
        err = hsa_memory_copy(dst, src, size);
    }

    return AM_SUCCESS;// TODO.
}


// TODO - change to use accelerator rather than accelerator_view.
am_status_t am_copy(void*  dst, const void*  src, size_t size, hc::accelerator_view dst_av)
{
    am_status_t am_status = AM_ERROR_MISC;

    // TODO - need to check for CPU accelerator not get_hsa_interop.
#ifdef HCC_VERSION_08
    if (dst_av.is_hsa_accelerator()) {
#else
    //TODO-kalmar - remove get_hsa_interop, this was old name for this function.
    if (dst_av.get_hsa_interop()) {
#endif
        hsa_agent_t *hsa_agent = static_cast <hsa_agent_t*> (dst_av.get_hsa_agent());

        hsa_memory_assign_agent(dst,  *hsa_agent, HSA_ACCESS_PERMISSION_RW);
        hsa_status_t err = hsa_memory_copy(dst, src, size);

        if (err == HSA_STATUS_SUCCESS) {
            am_status = AM_SUCCESS;
        } else {
            am_status = AM_ERROR_MISC;
        }
    }
    return am_status;
}

} // end namespace hc.
