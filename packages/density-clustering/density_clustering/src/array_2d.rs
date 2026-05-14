// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct Array2D<T> {
    width: i32,
    height: i32,
    data: Vec<T>,
}

impl<T> Array2D<T> {
    pub fn new(width: i32, height: i32, data: Vec<T>) -> Self {
        Array2D {
            width,
            height,
            data,
        }
    }

    pub fn new_with_constant(width: i32, height: i32, value: T) -> Self
    where
        T: Copy,
    {
        Array2D {
            width,
            height,
            data: vec![value; (width * height) as usize],
        }
    }

    pub fn zeros(width: i32, height: i32) -> Self
    where
        T: num_traits::Zero + Copy,
    {
        Array2D::new(width, height, vec![T::zero(); (width * height) as usize])
    }

    pub fn ones(width: i32, height: i32) -> Self
    where
        T: num_traits::One + Copy,
    {
        Array2D::new(width, height, vec![T::one(); (width * height) as usize])
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn iter_coords(&self) -> Array2DCoordIter {
        Array2DCoordIter {
            x: 0,
            y: 0,
            w: self.width,
            h: self.height,
        }
    }
}

pub struct Array2DCoordIter {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl Iterator for Array2DCoordIter {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.y < self.h {
            let r = (self.x, self.y);
            self.x += 1;
            if self.x == self.w {
                self.x = 0;
                self.y += 1;
            }
            Some(r)
        } else {
            None
        }
    }
}

impl<T> fmt::Debug for Array2D<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Array2D({}, {}) [", self.width, self.height)?;
        for row in 0..self.height {
            writeln!(
                f,
                "  {:?}",
                &self.data[(row * self.width) as usize..(row * self.width + self.width) as usize]
            )?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<T> Index<(i32, i32)> for Array2D<T> {
    type Output = T;
    fn index(&self, location: (i32, i32)) -> &T {
        &self.data[(location.1 * self.width + location.0) as usize]
    }
}

impl<T> IndexMut<(i32, i32)> for Array2D<T> {
    fn index_mut(&mut self, location: (i32, i32)) -> &mut T {
        &mut self.data[(location.1 * self.width + location.0) as usize]
    }
}
