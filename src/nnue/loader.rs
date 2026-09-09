// Ported from Stockfish sf_16 (68e1e9b3811e16cad014b590d7443b9063b3eb52),
// see Stockfish/src/nnue/nnue_common.h (read_little_endian, read_leb_128)
// and Stockfish/src/nnue/evaluate_nnue.cpp (read_header, read_parameters).

use std::io::{self, Read};

/// Version of the evaluation file.
/// Stockfish/src/nnue/nnue_common.h:51
pub const VERSION: u32 = 0x7AF32F20;

/// Stockfish/src/nnue/nnue_common.h:60-61
const LEB128_MAGIC_STRING: &[u8] = b"COMPRESSED_LEB128";

/// A thin wrapper so we can read sequentially from an in-memory buffer the
/// same way Stockfish reads from an std::istream, and detect EOF exactly.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    pub fn at_eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_exact_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("expected {n} bytes, only {} remain", self.remaining()),
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// read_little_endian<u32> -- Stockfish/src/nnue/nnue_common.h:92-111
    pub fn read_u32_le(&mut self) -> io::Result<u32> {
        let b = self.read_exact_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// read_little_endian<i16> for a single value.
    pub fn read_i16_le(&mut self) -> io::Result<i16> {
        let b = self.read_exact_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    /// read_little_endian<i32> for a single value.
    pub fn read_i32_le(&mut self) -> io::Result<i32> {
        let b = self.read_exact_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// read_little_endian<i8> for a single value (a plain byte reinterpreted signed).
    pub fn read_i8_le(&mut self) -> io::Result<i8> {
        let b = self.read_exact_bytes(1)?;
        Ok(b[0] as i8)
    }

    pub fn read_string(&mut self, len: usize) -> io::Result<String> {
        let b = self.read_exact_bytes(len)?;
        // Stockfish reads raw bytes into a std::string without validating
        // UTF-8; use lossy conversion purely for display purposes.
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    /// read_leb_128<i16>(stream, out, count) -- Stockfish/src/nnue/nnue_common.h:165-194
    ///
    /// Important: LEB128 compression is applied per-tensor at write time.
    /// Every call to this function first checks for the "COMPRESSED_LEB128"
    /// magic string immediately before the tensor, per Stockfish's own
    /// per-tensor framing -- we must not assume a fixed compressed/
    /// uncompressed layout for the whole feature transformer.
    pub fn read_leb128_i16(&mut self, out: &mut [i16]) -> io::Result<()> {
        read_leb128_generic(self, out, 16)
    }

    pub fn read_leb128_i32(&mut self, out: &mut [i32]) -> io::Result<()> {
        read_leb128_generic_i32(self, out)
    }
}

/// Generic LEB128 reader for a signed integer type of `bits` width (16 here),
/// writing results into an i16 output slice. Mirrors the templated C++
/// function bit-for-bit, including the sign-extension logic on the final
/// byte of each value and the 4096-byte read-ahead buffering (which does not
/// affect results, only I/O granularity, so we simplify it to direct byte
/// reads while preserving exact arithmetic).
///
/// FIX: removed the `if shift >= bits { break; }` early exit that was here
/// previously. Stockfish's original loop (nnue_common.h:165-194) has NO such
/// exit -- it is a do/while that only terminates when it reads a byte whose
/// continuation bit (0x80) is clear:
///
///   do {
///       byte = ...;
///       result |= (byte & 0x7f) << shift;
///       shift += 7;
///   } while ((byte & 0x80) != 0);
///
/// The old early-exit could fire on a value whose 3rd byte still had its
/// continuation bit set (fully possible for the i32 psqt-weight path, whose
/// `bits` is 32), causing the loop to `break` WITHOUT ever writing `*slot` --
/// silently leaving that entry at its zeroed `new_zeroed()` default instead
/// of the value actually stored in the file. It also left `bytes_left`/the
/// reader's stream position off by however many trailing bytes of that
/// value were never consumed, desynchronizing every subsequent read in the
/// tensor. Matching the unconditional continuation-bit-only exit fixes both.
fn read_leb128_generic(r: &mut Reader, out: &mut [i16], bits: u32) -> io::Result<()> {
    let magic = r.read_exact_bytes(LEB128_MAGIC_STRING.len())?;
    if magic != LEB128_MAGIC_STRING {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing COMPRESSED_LEB128 magic string before tensor",
        ));
    }
    let mut bytes_left = r.read_u32_le()? as i64;

    for slot in out.iter_mut() {
        let mut result: i32 = 0;
        let mut shift: u32 = 0;
        loop {
            let byte_slice = r.read_exact_bytes(1)?;
            let byte = byte_slice[0];
            bytes_left -= 1;
            result |= ((byte & 0x7f) as i32) << shift;
            shift += 7;
            if (byte & 0x80) == 0 {
                let value: i32 = if bits <= shift || (byte & 0x40) == 0 {
                    result
                } else {
                    result | !((1i32 << shift) - 1)
                };
                *slot = value as i16;
                break;
            }
            // No `shift >= bits` early exit here -- matches Stockfish's
            // unconditional "keep reading until continuation bit clears".
        }
    }

    debug_assert_eq!(bytes_left, 0, "LEB128 tensor did not consume exactly bytes_left bytes");
    Ok(())
}

fn read_leb128_generic_i32(r: &mut Reader, out: &mut [i32]) -> io::Result<()> {
    let magic = r.read_exact_bytes(LEB128_MAGIC_STRING.len())?;
    if magic != LEB128_MAGIC_STRING {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing COMPRESSED_LEB128 magic string before tensor",
        ));
    }
    let mut bytes_left = r.read_u32_le()? as i64;
    let bits: u32 = 32;

    for slot in out.iter_mut() {
        let mut result: i64 = 0;
        let mut shift: u32 = 0;
        loop {
            let byte_slice = r.read_exact_bytes(1)?;
            let byte = byte_slice[0];
            bytes_left -= 1;
            result |= ((byte & 0x7f) as i64) << shift;
            shift += 7;
            if (byte & 0x80) == 0 {
                let value: i64 = if bits <= shift || (byte & 0x40) == 0 {
                    result
                } else {
                    result | !((1i64 << shift) - 1)
                };
                *slot = value as i32;
                break;
            }
            // No `shift >= bits` early exit here -- matches Stockfish's
            // unconditional "keep reading until continuation bit clears".
        }
    }

    debug_assert_eq!(bytes_left, 0, "LEB128 tensor did not consume exactly bytes_left bytes");
    Ok(())
}

/// Read an entire .nnue file into memory.
pub fn read_file(path: &str) -> io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Network header: version, hash, description.
/// Stockfish/src/nnue/evaluate_nnue.cpp:94-105
pub struct Header {
    pub hash_value: u32,
    pub description: String,
}

pub fn read_header(r: &mut Reader) -> io::Result<Header> {
    let version = r.read_u32_le()?;
    let hash_value = r.read_u32_le()?;
    let size = r.read_u32_le()?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad NNUE version: got {version:#010x}, expected {VERSION:#010x}"),
        ));
    }
    let description = r.read_string(size as usize)?;
    Ok(Header {
        hash_value,
        description,
    })
}

/// Detail::read_parameters<T> -- Stockfish/src/nnue/evaluate_nnue.cpp:66-73
/// Reads a per-component u32 header and checks it against the expected hash.
pub fn read_component_header(r: &mut Reader, expected_hash: u32) -> io::Result<()> {
    let header = r.read_u32_le()?;
    if header != expected_hash {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "component hash mismatch: file has {header:#010x}, port computes {expected_hash:#010x} \
                 -- this means our Rust port's declared dimensions/architecture for this \
                 component do not match the network file, and must be fixed before trusting \
                 any further output"
            ),
        ));
    }
    Ok(())
}
