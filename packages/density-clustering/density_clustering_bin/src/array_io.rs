// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

use std::io::{Read, Write};

use density_clustering::Array2D;

unsafe fn unsafe_read_array<T: num_traits::Zero + Copy>(
    path: &str,
    width: i32,
    height: i32,
) -> Result<Array2D<T>, std::io::Error> {
    let mut f = std::fs::File::open(path)?;
    let mut result = Array2D::<T>::zeros(width, height);
    unsafe {
        let ptr = result.as_mut_slice().as_mut_ptr();
        let byte_size = (width * height) as usize * std::mem::size_of::<T>();
        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, byte_size);
        f.read_exact(slice)?;
    }
    Ok(result)
}

unsafe fn unsafe_write_array<T: num_traits::Zero + Copy>(
    path: &str,
    array: &Array2D<T>,
) -> Result<(), std::io::Error> {
    let mut f = std::fs::File::create(path)?;
    unsafe {
        let ptr = array.as_slice().as_ptr();
        let byte_size = (array.width() * array.height()) as usize * std::mem::size_of::<T>();
        let slice = std::slice::from_raw_parts(ptr as *mut u8, byte_size);
        f.write_all(slice)?;
    }
    Ok(())
}

pub fn read_f32_array(path: &str, width: i32, height: i32) -> Result<Array2D<f32>, std::io::Error> {
    unsafe { unsafe_read_array::<f32>(path, width, height) }
}

pub fn write_f32_array(path: &str, array: &Array2D<f32>) -> Result<(), std::io::Error> {
    unsafe { unsafe_write_array::<f32>(path, array) }
}

pub fn read_i32_array(path: &str, width: i32, height: i32) -> Result<Array2D<i32>, std::io::Error> {
    unsafe { unsafe_read_array::<i32>(path, width, height) }
}

pub fn write_i32_array(path: &str, array: &Array2D<i32>) -> Result<(), std::io::Error> {
    unsafe { unsafe_write_array::<i32>(path, array) }
}
