.. _decoders:

===================
torchcodec.decoders
===================

.. currentmodule:: torchcodec.decoders



Video Decoding
--------------

For a video decoder tutorial, see: :ref:`sphx_glr_generated_examples_decoding_basic_example.py`.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    VideoDecoder
    WavDecoder

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    VideoStreamMetadata

**CUDA decoding utils:**

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    set_cuda_backend
    set_nvdec_cache_capacity
    get_nvdec_cache_capacity


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    CpuFallbackStatus

Audio Decoding
--------------

For an audio decoder tutorial, see: :ref:`sphx_glr_generated_examples_decoding_audio_decoding.py`.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AudioDecoder
    WavDecoder

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    AudioStreamMetadata


Image Decoding
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    decode_jpeg
    decode_png
    decode_webp
    decode_gif
    decode_avif

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: enum.rst

    ImageReadMode
