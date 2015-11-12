#pragma once

#include <hc.hpp>



// Provide automatic type conversion for void*.
class auto_voidp {
    void *_ptr;
    public:
        auto_voidp (void *ptr) : _ptr (ptr) {}
        template<class T> operator T *() { return (T *) _ptr; }
};

typedef int am_status_t;
#define AM_SUCCESS                           0
#define AM_ERROR_MISC                       -6 /** Misellaneous error */


/** Disable automatic sync when a block is filled from host memory to an accelerator cache. 
    This can be useful in cases where the memory is written by the accelerator before being read.
    Care should be taken with this flag as it effects a block of memory
*/
#define AM_DISABLE_AUTO_SYNC_IN     0x0001

#define AM_ENABLE_AUTO_SYNC_IN      0x0002     

/** Disable automatic synchronization when a block is evicted from the accelerator cache to the host.
    This is useful in cases where the data is not needed on the host. */
#define AM_DISABLE_AUTO_SYNC_OUT    0x0004 

#define AM_ENABLE_AUTO_SYNC_OUT     0x0008

/** Combine both DISABLE_SYNC flags. Application will explicitly manage synchronization for this
memory region using @ref am_copy or @ref am_update.
*/
#define AM_EXPLICIT_SYNC (AM_DISABLE_AUTO_SYNC_IN | AM_DISABLE_AUTO_SYNC_OUT)

namespace hc {

auto_voidp am_alloc(size_t size, unsigned flags, hc::accelerator_view acc) ;
am_status_t am_free(void*  ptr);
am_status_t am_copy(void*  dst, const void*  src, size_t size);
am_status_t am_copy(void*  dst, const void*  src, size_t size, hc::accelerator_view dst_acc);


}; // namespace hc


