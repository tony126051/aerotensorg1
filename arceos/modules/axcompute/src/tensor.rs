#![no_std]

use core::slice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Float16,
    Bfloat16,
    Float32,
    Int8,
}

impl DataType {
    pub const fn size_in_bytes(&self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Zero-Copy Tensor Descriptor mapped directly into UMA Physical Memory
#[derive(Clone, Debug)]
pub struct Tensor {
    dims: [usize; 4],
    ndims: usize,
    dtype: DataType,
    paddr: u64,
    vaddr: usize,
    num_elements: usize,
}

impl Tensor {
    /// Construct a Tensor wrapping existing physically contiguous memory
    pub fn from_raw_parts(
        paddr: u64,
        vaddr: usize,
        dims: &[usize],
        dtype: DataType,
    ) -> Result<Self, &'static str> {
        if dims.is_empty() || dims.len() > 4 {
            return Err("Tensor dims must be between 1 and 4");
        }

        let mut dims_arr = [1usize; 4];
        let mut total_elems = 1;
        for (i, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err("Dimension size must be greater than zero");
            }
            dims_arr[i] = d;
            total_elems *= d;
        }

        Ok(Self {
            dims: dims_arr,
            ndims: dims.len(),
            dtype,
            paddr,
            vaddr,
            num_elements: total_elems,
        })
    }

    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.dims[..self.ndims]
    }

    #[inline(always)]
    pub fn dim(&self, index: usize) -> usize {
        if index < self.ndims {
            self.dims[index]
        } else {
            1
        }
    }

    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    #[inline(always)]
    pub fn size_in_bytes(&self) -> usize {
        self.num_elements * self.dtype.size_in_bytes()
    }

    #[inline(always)]
    pub fn paddr(&self) -> u64 {
        self.paddr
    }

    #[inline(always)]
    pub fn vaddr(&self) -> usize {
        self.vaddr
    }

    #[inline(always)]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Read-only slice view of the tensor elements
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        slice::from_raw_parts(self.vaddr as *const T, self.num_elements)
    }

    /// Mutable slice view of the tensor elements
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.vaddr as *mut T, self.num_elements)
    }
}
