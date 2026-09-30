#![no_std]

use axcompute::tensor::{DataType, Tensor};

pub const MAX_TENSORS_PER_FILE: usize = 128;

/// Descriptor of a single tensor inside the Safetensors container
#[derive(Clone, Copy, Debug)]
pub struct SafeTensorEntry<'a> {
    pub name: &'a str,
    pub dtype: DataType,
    pub shape: [usize; 4],
    pub ndims: usize,
    pub offset_begin: usize,
    pub offset_end: usize,
}

/// Zero-Copy Safetensors Model Weight Archive
pub struct SafeTensors<'a> {
    base_paddr: u64,
    raw_payload_offset: usize,
    entries: [Option<SafeTensorEntry<'a>>; MAX_TENSORS_PER_FILE],
    count: usize,
}

impl<'a> SafeTensors<'a> {
    /// Parse a Safetensors file header directly from contiguous physical/virtual memory
    pub fn parse(buffer: &'a [u8], base_paddr: u64) -> Result<Self, &'static str> {
        if buffer.len() < 8 {
            return Err("Buffer too small for Safetensors header length");
        }

        let header_len_bytes: [u8; 8] = buffer[0..8]
            .try_into()
            .map_err(|_| "Header slice error")?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        if 8 + header_len > buffer.len() {
            return Err("Invalid Safetensors header length exceeds buffer");
        }

        let header_json = core::str::from_utf8(&buffer[8..8 + header_len])
            .map_err(|_| "Invalid UTF-8 in Safetensors JSON header")?;

        let mut instance = Self {
            base_paddr,
            raw_payload_offset: 8 + header_len,
            entries: [None; MAX_TENSORS_PER_FILE],
            count: 0,
        };

        instance.parse_json_metadata(header_json)?;

        Ok(instance)
    }

    /// Lightweight, zero-alloc JSON scanner for Safetensors tensor metadata
    fn parse_json_metadata(&mut self, json: &'a str) -> Result<(), &'static str> {
        let bytes = json.as_bytes();
        let len = bytes.len();
        let mut idx = 0;

        while idx < len {
            // Find key opening quote
            while idx < len && bytes[idx] != b'"' {
                idx += 1;
            }
            if idx >= len {
                break;
            }
            idx += 1; // skip opening quote
            let key_start = idx;
            while idx < len && bytes[idx] != b'"' {
                idx += 1;
            }
            let key_end = idx;
            idx += 1; // skip closing quote

            let key = &json[key_start..key_end];
            if key == "__metadata__" {
                continue;
            }

            // Find colon
            while idx < len && bytes[idx] != b':' {
                idx += 1;
            }
            if idx >= len {
                break;
            }
            idx += 1;

            // Find opening brace '{'
            while idx < len && bytes[idx] != b'{' && bytes[idx] != b',' {
                idx += 1;
            }
            if idx >= len || bytes[idx] != b'{' {
                continue;
            }
            idx += 1;

            // Parse tensor object content until matching '}'
            let mut dtype = DataType::Float16;
            let mut shape = [1usize; 4];
            let mut ndims = 0;
            let mut offset_begin = 0;
            let mut offset_end = 0;

            while idx < len && bytes[idx] != b'}' {
                // Find inner field key
                while idx < len && bytes[idx] != b'"' && bytes[idx] != b'}' {
                    idx += 1;
                }
                if idx >= len || bytes[idx] == b'}' {
                    break;
                }
                idx += 1;
                let field_start = idx;
                while idx < len && bytes[idx] != b'"' {
                    idx += 1;
                }
                let field_end = idx;
                idx += 1;
                let field = &json[field_start..field_end];

                // skip to colon
                while idx < len && bytes[idx] != b':' {
                    idx += 1;
                }
                idx += 1;

                match field {
                    "dtype" => {
                        while idx < len && bytes[idx] != b'"' {
                            idx += 1;
                        }
                        idx += 1;
                        let d_start = idx;
                        while idx < len && bytes[idx] != b'"' {
                            idx += 1;
                        }
                        let d_str = &json[d_start..idx];
                        idx += 1;

                        dtype = match d_str {
                            "F16" => DataType::Float16,
                            "BF16" => DataType::Bfloat16,
                            "F32" => DataType::Float32,
                            "I8" => DataType::Int8,
                            _ => DataType::Float16,
                        };
                    }
                    "shape" => {
                        while idx < len && bytes[idx] != b'[' {
                            idx += 1;
                        }
                        idx += 1;
                        ndims = 0;
                        while idx < len && bytes[idx] != b']' {
                            while idx < len && (bytes[idx] == b' ' || bytes[idx] == b',') {
                                idx += 1;
                            }
                            if bytes[idx] == b']' {
                                break;
                            }
                            let mut num = 0usize;
                            while idx < len && bytes[idx].is_ascii_digit() {
                                num = num * 10 + (bytes[idx] - b'0') as usize;
                                idx += 1;
                            }
                            if ndims < 4 {
                                shape[ndims] = num;
                                ndims += 1;
                            }
                        }
                        if idx < len && bytes[idx] == b']' {
                            idx += 1;
                        }
                    }
                    "data_offsets" => {
                        while idx < len && bytes[idx] != b'[' {
                            idx += 1;
                        }
                        idx += 1;
                        // begin offset
                        while idx < len && (bytes[idx] == b' ' || bytes[idx] == b',') {
                            idx += 1;
                        }
                        let mut num1 = 0usize;
                        while idx < len && bytes[idx].is_ascii_digit() {
                            num1 = num1 * 10 + (bytes[idx] - b'0') as usize;
                            idx += 1;
                        }
                        offset_begin = num1;

                        // end offset
                        while idx < len && (bytes[idx] == b' ' || bytes[idx] == b',') {
                            idx += 1;
                        }
                        let mut num2 = 0usize;
                        while idx < len && bytes[idx].is_ascii_digit() {
                            num2 = num2 * 10 + (bytes[idx] - b'0') as usize;
                            idx += 1;
                        }
                        offset_end = num2;

                        while idx < len && bytes[idx] != b']' {
                            idx += 1;
                        }
                        if idx < len && bytes[idx] == b']' {
                            idx += 1;
                        }
                    }
                    _ => {
                        // Skip unhandled fields until comma or closing brace
                        while idx < len && bytes[idx] != b',' && bytes[idx] != b'}' {
                            idx += 1;
                        }
                    }
                }
            }

            if idx < len && bytes[idx] == b'}' {
                idx += 1;
            }

            if self.count < MAX_TENSORS_PER_FILE {
                self.entries[self.count] = Some(SafeTensorEntry {
                    name: key,
                    dtype,
                    shape,
                    ndims: if ndims == 0 { 1 } else { ndims },
                    offset_begin,
                    offset_end,
                });
                self.count += 1;
            }
        }

        Ok(())
    }

    /// Retrieve metadata entry by tensor name
    pub fn get_tensor_entry(&self, name: &str) -> Option<&SafeTensorEntry<'a>> {
        for i in 0..self.count {
            if let Some(ref entry) = self.entries[i] {
                if entry.name == name {
                    return Some(entry);
                }
            }
        }
        None
    }

    /// Directly wrap the in-memory tensor payload into a zero-copy axcompute::Tensor
    pub fn get_tensor(
        &self,
        name: &str,
        raw_buffer: &'a [u8],
    ) -> Result<Tensor, &'static str> {
        let entry = self
            .get_tensor_entry(name)
            .ok_or("Tensor not found in archive")?;

        let vaddr_start = raw_buffer.as_ptr() as usize + self.raw_payload_offset + entry.offset_begin;
        let paddr_start = self.base_paddr + (self.raw_payload_offset as u64) + (entry.offset_begin as u64);

        Tensor::from_raw_parts(
            paddr_start,
            vaddr_start,
            &entry.shape[..entry.ndims],
            entry.dtype,
        )
    }

    /// Total number of tensors in this Safetensors file
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}
