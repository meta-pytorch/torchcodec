# set_nvdec_cache_capacity

torchcodec.decoders.set_nvdec_cache_capacity(*capacity: [int](https://docs.python.org/3/library/functions.html#int)*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torchcodec/decoders/_decoder_utils.html#set_nvdec_cache_capacity)

Set the maximum number of NVDEC decoders that can be cached (per GPU).

The NVDEC decoder cache stores hardware decoders for reuse, avoiding the
overhead of creating and destructing new decoders for subsequent video
decoding operations on the same GPU. This function sets the capacity of the
cache, i.e. the maximum number of decoders that can be cached per device.
The default capacity is 20 decoders per device. If the cache contains more
decoders than the target `capacity`, excess decoders will be evicted
using a least-recently-used policy.

Generally, a decoder can be re-used from the cache if it matches the same
codec and frame dimensions.

See also [`get_nvdec_cache_capacity()`](torchcodec.decoders.get_nvdec_cache_capacity.html#torchcodec.decoders.get_nvdec_cache_capacity).

Parameters:

**capacity** ([*int*](https://docs.python.org/3/library/functions.html#int)) - The maximum number of NVDEC decoders that can be cached
per GPU device. Must be non-negative. Setting to 0 disables caching.