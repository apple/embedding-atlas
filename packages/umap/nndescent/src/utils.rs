/// Utility functions: visited bitmap, norm, etc.

/// Bit-level visited table for tracking which nodes have been explored.
pub struct VisitedTable {
    table: Vec<u8>,
}

impl VisitedTable {
    pub fn new(n_points: usize) -> Self {
        VisitedTable {
            table: vec![0u8; (n_points >> 3) + 1],
        }
    }

    pub fn has_been_visited(&self, candidate: i32) -> bool {
        let loc = (candidate >> 3) as usize;
        let mask = 1u8 << (candidate & 7);
        self.table[loc] & mask != 0
    }

    pub fn mark_visited(&mut self, candidate: i32) {
        let loc = (candidate >> 3) as usize;
        let mask = 1u8 << (candidate & 7);
        self.table[loc] |= mask;
    }

    /// Check if visited and mark as visited in one operation.
    /// Returns true if ALREADY visited (before marking).
    pub fn check_and_mark_visited(&mut self, candidate: i32) -> bool {
        let loc = (candidate >> 3) as usize;
        let mask = 1u8 << (candidate & 7);
        let was_visited = self.table[loc] & mask != 0;
        self.table[loc] |= mask;
        was_visited
    }

    pub fn reset(&mut self) {
        self.table.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_table() {
        let mut table = VisitedTable::new(100);
        assert!(!table.has_been_visited(5));
        table.mark_visited(5);
        assert!(table.has_been_visited(5));
        assert!(!table.has_been_visited(6));

        // check_and_mark_visited
        assert!(!table.check_and_mark_visited(10));
        assert!(table.check_and_mark_visited(10));

        // reset
        table.reset();
        assert!(!table.has_been_visited(5));
        assert!(!table.has_been_visited(10));
    }
}
