"""
Test concurrent encoding and decoding of fragmented MP4.

This script encodes a video with fragmented MP4 options while repeatedly
attempting to decode the file from another process to test read-while-write.
"""

import multiprocessing
import os
import time

import torch


def get_test_frames():
    """Generate test frames - colored gradient frames."""
    num_frames = 900  # More frames = longer encoding time
    height, width = 1920, 1080  # Full HD for slower encoding
    frames = torch.zeros((num_frames, 3, height, width), dtype=torch.uint8)

    for i in range(num_frames):
        # Create a gradient that changes per frame
        r = int(255 * i / num_frames)
        g = int(255 * (1 - i / num_frames))
        b = 128
        frames[i, 0, :, :] = r  # R channel
        frames[i, 1, :, :] = g  # G channel
        frames[i, 2, :, :] = b  # B channel

    return frames


def writer_process(path: str, ready_event, done_event):
    """Encode frames to a fragmented MP4 file."""
    from torchcodec.encoders import VideoEncoder

    print(f"[WRITER] Starting encoder, output: {path}")
    frames = get_test_frames()
    print(f"[WRITER] Generated {len(frames)} frames of shape {frames.shape}")

    encoder = VideoEncoder(frames=frames, frame_rate=30.0)

    # Signal that we're about to start encoding
    ready_event.set()

    start_time = time.time()
    encoder.to_file(
        dest=path,
        preset="slow",  # Slower preset = more time to test concurrent reads
        extra_options={
            "movflags": "+frag_keyframe+empty_moov",
            "frag_duration": "100000",  # Fragment every 100ms
        },
    )
    elapsed = time.time() - start_time

    print(f"[WRITER] Encoding complete in {elapsed:.2f}s")
    done_event.set()


def reader_process(path: str, ready_event, done_event):
    """Repeatedly attempt to decode the file while it's being written."""
    from torchcodec.decoders import VideoDecoder

    # Wait for writer to be ready
    ready_event.wait()
    print("[READER] Writer is ready, starting decode attempts")

    attempt = 0
    last_file_size = 0
    last_frame_count = 0

    while not done_event.is_set() or attempt < 3:  # A few extra attempts after done
        attempt += 1
        time.sleep(0.1)  # Check every 100ms

        # Check file existence and size
        if not os.path.exists(path):
            print(f"[READER] Attempt {attempt}: File does not exist yet")
            continue

        file_size = os.path.getsize(path)
        size_delta = file_size - last_file_size
        last_file_size = file_size

        try:
            # Try to open with approximate seek mode (more tolerant of incomplete files)
            decoder = VideoDecoder(path, seek_mode="approximate")
            num_frames = len(decoder)

            # Try to decode frames
            decoded_count = 0
            for i in range(num_frames):
                try:
                    decoder.get_frame_at(i)
                    decoded_count += 1
                except RuntimeError:
                    # Stop at first decode error
                    break

            frame_delta = decoded_count - last_frame_count
            last_frame_count = decoded_count

            print(
                f"[READER] Attempt {attempt}: "
                f"file_size={file_size:,} bytes (+{size_delta:,}), "
                f"reported_frames={num_frames}, "
                f"decoded_frames={decoded_count} (+{frame_delta})"
            )

        except RuntimeError as e:
            error_msg = str(e)[:80]  # Truncate long error messages
            print(
                f"[READER] Attempt {attempt}: "
                f"file_size={file_size:,} bytes (+{size_delta:,}), "
                f"error={error_msg}"
            )
        except Exception as e:
            print(
                f"[READER] Attempt {attempt}: Unexpected error: {type(e).__name__}: {e}"
            )

    print(
        f"[READER] Done after {attempt} attempts, final decoded frames: {last_frame_count}"
    )


def main():
    output_path = "/tmp/concurrent_test.mp4"

    # Clean up from previous runs
    if os.path.exists(output_path):
        os.remove(output_path)

    # Create synchronization events
    ready_event = multiprocessing.Event()
    done_event = multiprocessing.Event()

    # Start processes
    writer = multiprocessing.Process(
        target=writer_process, args=(output_path, ready_event, done_event)
    )
    reader = multiprocessing.Process(
        target=reader_process, args=(output_path, ready_event, done_event)
    )

    print("Starting concurrent encode/decode test...")
    print(f"Output file: {output_path}")
    print("-" * 60)

    writer.start()
    reader.start()

    writer.join()
    reader.join()

    print("-" * 60)

    # Final verification
    if os.path.exists(output_path):
        final_size = os.path.getsize(output_path)
        print(f"Final file size: {final_size:,} bytes")

        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(output_path)
        print(f"Final frame count: {len(decoder)}")
    else:
        print("ERROR: Output file was not created")


if __name__ == "__main__":
    main()
