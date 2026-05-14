// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

pub struct DisjointSet {
    parent: Vec<i32>,
}

impl DisjointSet {
    pub fn new(length: i32) -> DisjointSet {
        DisjointSet {
            parent: (0..length).collect(),
        }
    }

    pub fn find_parent(&mut self, index: i32) -> i32 {
        if self.parent[index as usize] == index {
            index
        } else {
            let p = self.find_parent(self.parent[index as usize]);
            self.parent[index as usize] = p;
            p
        }
    }

    pub fn union(&mut self, index1: i32, index2: i32) {
        let p1 = self.find_parent(index1);
        let p2 = self.find_parent(index2);
        self.parent[p1 as usize] = p2;
    }

    pub fn num_unique_sets(&self) -> i32 {
        let mut count = 0;
        for i in 0..self.parent.len() {
            if self.parent[i] as usize == i {
                count += 1;
            }
        }
        count
    }
}
