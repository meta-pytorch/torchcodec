.. _decoders_blocks:

===========================
torchcodec.decoders._blocks
===========================

.. currentmodule:: torchcodec.decoders._blocks

.. TODO_API_BREAKDOWN DOC Remove all 'warning this is private blahblahblah'

.. warning::

   **The Blocks APIs are under active construction.** They are private and
   unreleased. Signatures and semantics may change without notice.

For a tutorial, see:
:ref:`sphx_glr_generated_examples_decoding_blocks.py`.

Demuxing
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Demuxer
    VideoStream
    AudioStream

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    get_container_metadata

Decoding
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    VideoPacketDecoder
    AudioPacketDecoder

Conversion
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    ColorConverter
    AudioConverter

Data types
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Packet
    RawFrame

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    FrameIndex
    RawAudioSamples

Metadata
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    DemuxerMetadata
    ContainerMetadata
    VideoStreamHeaderMetadata
    AudioStreamHeaderMetadata
    StreamMetadata
