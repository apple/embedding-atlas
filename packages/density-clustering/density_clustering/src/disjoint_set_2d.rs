// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

use crate::array_2d::Array2D;

pub struct DisjointSet2D {
    width: i32,
    height: i32,
    parent: Array2D<(i32, i32)>,
}

impl DisjointSet2D {
    pub fn new(width: i32, height: i32) -> Self {
        let mut parent = Array2D::new(width, height, vec![(0, 0); (width * height) as usize]);
        for (x, y) in parent.iter_coords() {
            parent[(x, y)] = (x, y);
        }
        DisjointSet2D {
            width,
            height,
            parent,
        }
    }

    pub fn find_parent(&mut self, location: (i32, i32)) -> (i32, i32) {
        if self.parent[location] == location {
            location
        } else {
            let p = self.find_parent(self.parent[location]);
            self.parent[location] = p;
            p
        }
    }

    pub fn union(&mut self, location1: (i32, i32), location2: (i32, i32)) {
        let p1 = self.find_parent(location1);
        let p2 = self.find_parent(location2);
        self.parent[p1] = p2;
    }

    pub fn num_unique_sets(&self) -> i32 {
        let mut count = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.parent[(x, y)] == (x, y) {
                    count += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disjoint_set_2d() {
        let mut ds = DisjointSet2D::new(3, 3);
        assert!(ds.find_parent((1, 1)) == (1, 1));
        ds.union((1, 1), (2, 2));
        assert!(ds.num_unique_sets() == 8);
        ds.union((1, 0), (0, 1));
        assert!(ds.num_unique_sets() == 7);
        ds.union((1, 0), (1, 1));
        assert!(ds.num_unique_sets() == 6);
        assert!(ds.find_parent((0, 1)) == ds.find_parent((2, 2)));
        assert!(ds.num_unique_sets() == 6);
    }
}
