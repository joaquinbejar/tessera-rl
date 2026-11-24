// SPDX-License-Identifier: MIT OR Apache-2.0

//! GIF export of rendered frames (ADR-0083 item 5). Each frame is quantized
//! to its own 256-colour palette (`NeuQuant`); the output is a looping GIF.
//! Video bytes are not part of any golden: only frames are.

use std::io::Write;

use thiserror::Error;

/// A video export failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VideoError {
    /// A frame whose length is not `width * height * 3`.
    #[error("frame {index} has {got} bytes, expected {expected}")]
    FrameLength {
        /// The frame's position.
        index: usize,
        /// The expected length.
        expected: usize,
        /// The given length.
        got: usize,
    },
    /// No frames, a zero side, or a frame rate outside `1..=100`.
    #[error("{0}")]
    Argument(&'static str),
    /// The GIF encoder failed (an I/O error of the writer included).
    #[error("GIF encoding failed: {0}")]
    Encode(#[from] gif::EncodingError),
}

/// Writes `frames` (each `[height, width, 3]` RGB8) to `out` as a looping
/// GIF at `fps` frames per second (1 to 100; GIF delays are in hundredths
/// of a second, so the rate is rounded to `100 / round(100 / fps)`).
///
/// # Errors
///
/// [`VideoError`] for bad arguments, a frame of the wrong length, or an
/// encoder or writer failure.
pub fn encode_gif<W: Write>(
    out: W,
    width: u16,
    height: u16,
    fps: u16,
    frames: &[&[u8]],
) -> Result<W, VideoError> {
    if frames.is_empty() {
        return Err(VideoError::Argument("no frames to encode"));
    }
    if width == 0 || height == 0 {
        return Err(VideoError::Argument("width and height must be at least 1"));
    }
    if !(1..=100).contains(&fps) {
        return Err(VideoError::Argument("fps must be in 1..=100"));
    }
    let expected = usize::from(width) * usize::from(height) * 3;
    for (index, f) in frames.iter().enumerate() {
        if f.len() != expected {
            return Err(VideoError::FrameLength {
                index,
                expected,
                got: f.len(),
            });
        }
    }
    let delay = (100 + fps / 2) / fps;
    let mut encoder = gif::Encoder::new(out, width, height, &[])?;
    encoder.set_repeat(gif::Repeat::Infinite)?;
    for f in frames {
        let mut frame = gif::Frame::from_rgb_speed(width, height, f, 10);
        frame.delay = delay;
        encoder.write_frame(&frame)?;
    }
    Ok(encoder.into_inner()?)
}

#[cfg(test)]
mod tests {
    use super::{VideoError, encode_gif};

    #[test]
    fn test_encode_gif_two_frames_valid_header_and_trailer() {
        let red = [255, 0, 0].repeat(4);
        let blue = [0, 0, 255].repeat(4);
        let out = encode_gif(Vec::new(), 2, 2, 25, &[&red, &blue]).unwrap_or_default();
        assert!(out.starts_with(b"GIF89a"));
        assert_eq!(out.last(), Some(&0x3b));
    }

    #[test]
    fn test_encode_gif_bad_inputs_refused() {
        let px = [0; 12];
        assert!(matches!(
            encode_gif(Vec::new(), 2, 2, 25, &[]),
            Err(VideoError::Argument(_))
        ));
        assert!(matches!(
            encode_gif(Vec::new(), 2, 2, 0, &[&px]),
            Err(VideoError::Argument(_))
        ));
        assert!(matches!(
            encode_gif(Vec::new(), 2, 2, 101, &[&px]),
            Err(VideoError::Argument(_))
        ));
        assert!(matches!(
            encode_gif(Vec::new(), 2, 2, 25, &[&px[..11]]),
            Err(VideoError::FrameLength {
                index: 0,
                expected: 12,
                got: 11
            })
        ));
    }
}
