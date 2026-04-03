/// Max-heap data structure for k-NN tracking.
/// Stores parallel arrays of indices, distances, and flags per point.
/// The heap root (index 0) always contains the MAXIMUM distance.
use ndarray::Array2;
use rayon::prelude::*;

/// A neighbor heap storing k-nearest neighbors for n points.
pub struct NeighborHeap {
    pub indices: Array2<i32>,
    pub distances: Array2<f32>,
    pub flags: Array2<u8>,
}

/// Create a new heap initialized with -1 indices, inf distances, and 0 flags.
pub fn make_heap(n_points: usize, size: usize) -> NeighborHeap {
    NeighborHeap {
        indices: Array2::from_elem((n_points, size), -1i32),
        distances: Array2::from_elem((n_points, size), f32::INFINITY),
        flags: Array2::zeros((n_points, size)),
    }
}

/// Push a value into a max-heap without checking for duplicates.
/// Returns 1 if the value was inserted, 0 if rejected (distance too large).
pub fn simple_heap_push(priorities: &mut [f32], indices: &mut [i32], p: f32, n: i32) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();
    priorities[0] = p;
    indices[0] = n;

    // Sift down to restore max-heap property
    let mut i = 0;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap;
        if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                i_swap = ic1;
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                i_swap = ic1;
            } else {
                break;
            }
        } else {
            if p < priorities[ic2] {
                i_swap = ic2;
            } else {
                break;
            }
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    1
}

/// Push a value into a max-heap, checking for duplicates first.
/// Returns 1 if inserted, 0 if rejected or duplicate.
pub fn checked_heap_push(priorities: &mut [f32], indices: &mut [i32], p: f32, n: i32) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    // Check for duplicate
    for i in 0..size {
        if n == indices[i] {
            return 0;
        }
    }

    priorities[0] = p;
    indices[0] = n;

    let mut i = 0;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap;
        if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                i_swap = ic1;
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                i_swap = ic1;
            } else {
                break;
            }
        } else {
            if p < priorities[ic2] {
                i_swap = ic2;
            } else {
                break;
            }
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    1
}

/// Push a value with a flag into a max-heap, checking for duplicates.
/// Returns 1 if inserted, 0 if rejected or duplicate.
pub fn checked_flagged_heap_push(
    priorities: &mut [f32],
    indices: &mut [i32],
    flags: &mut [u8],
    p: f32,
    n: i32,
    f: u8,
) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    // Check for duplicate
    for i in 0..size {
        if n == indices[i] {
            return 0;
        }
    }

    priorities[0] = p;
    indices[0] = n;
    flags[0] = f;

    let mut i = 0;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap;
        if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                i_swap = ic1;
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                i_swap = ic1;
            } else {
                break;
            }
        } else {
            if p < priorities[ic2] {
                i_swap = ic2;
            } else {
                break;
            }
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        flags[i] = flags[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    flags[i] = f;
    1
}

/// Sift down element at position `elt` to restore max-heap property.
fn siftdown(heap1: &mut [f32], heap2: &mut [i32], mut elt: usize) {
    while elt * 2 + 1 < heap1.len() {
        let left = elt * 2 + 1;
        let right = left + 1;
        let mut swap = elt;

        if heap1[swap] < heap1[left] {
            swap = left;
        }
        if right < heap1.len() && heap1[swap] < heap1[right] {
            swap = right;
        }
        if swap == elt {
            break;
        }

        heap1.swap(elt, swap);
        heap2.swap(elt, swap);
        elt = swap;
    }
}

/// Convert max-heap to ascending sorted order (in-place).
/// This is the second half of heapsort.
pub fn deheap_sort(indices: &mut Array2<i32>, distances: &mut Array2<f32>) {
    let n_rows = indices.nrows();
    let n_cols = indices.ncols();

    // Process each row in parallel
    let idx_ptr = indices.as_mut_ptr();
    let dist_ptr = distances.as_mut_ptr();

    // SAFETY: We cast array pointers to usize to move them into the parallel closure
    // (raw pointers are !Send). Each parallel iteration `i` exclusively accesses
    // row `i` (elements [i*n_cols .. (i+1)*n_cols]), so no two threads touch the
    // same memory. The arrays remain live and pinned for the duration of the par_iter.
    let idx_raw = idx_ptr as usize;
    let dist_raw = dist_ptr as usize;

    (0..n_rows).into_par_iter().for_each(|i| {
        let idx_slice = unsafe {
            std::slice::from_raw_parts_mut((idx_raw as *mut i32).add(i * n_cols), n_cols)
        };
        let dist_slice = unsafe {
            std::slice::from_raw_parts_mut((dist_raw as *mut f32).add(i * n_cols), n_cols)
        };

        for j in (1..n_cols).rev() {
            idx_slice.swap(0, j);
            dist_slice.swap(0, j);
            siftdown(&mut dist_slice[..j], &mut idx_slice[..j], 0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_heap() {
        let heap = make_heap(5, 3);
        assert_eq!(heap.indices.nrows(), 5);
        assert_eq!(heap.indices.ncols(), 3);
        assert_eq!(heap.indices[[0, 0]], -1);
        assert_eq!(heap.distances[[0, 0]], f32::INFINITY);
    }

    #[test]
    fn test_simple_heap_push() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1i32; 3];

        assert_eq!(simple_heap_push(&mut priorities, &mut indices, 5.0, 1), 1);
        assert_eq!(simple_heap_push(&mut priorities, &mut indices, 3.0, 2), 1);
        assert_eq!(simple_heap_push(&mut priorities, &mut indices, 1.0, 3), 1);
        // Heap is full with max=5.0, pushing 7.0 should fail
        assert_eq!(simple_heap_push(&mut priorities, &mut indices, 7.0, 4), 0);
        // Pushing 2.0 should succeed (replaces the max)
        assert_eq!(simple_heap_push(&mut priorities, &mut indices, 2.0, 5), 1);
    }

    #[test]
    fn test_checked_flagged_heap_push_dedup() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1i32; 3];
        let mut flags = vec![0u8; 3];

        checked_flagged_heap_push(&mut priorities, &mut indices, &mut flags, 5.0, 1, 1);
        checked_flagged_heap_push(&mut priorities, &mut indices, &mut flags, 3.0, 2, 1);
        // Duplicate index 1 should be rejected
        let result =
            checked_flagged_heap_push(&mut priorities, &mut indices, &mut flags, 1.0, 1, 1);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_deheap_sort() {
        let mut heap = make_heap(1, 5);
        // Push values into the heap for row 0
        let row = 0;
        let d = heap.distances.as_slice_mut().unwrap();
        let idx = heap.indices.as_slice_mut().unwrap();
        simple_heap_push(&mut d[..5], &mut idx[..5], 5.0, 50);
        simple_heap_push(&mut d[..5], &mut idx[..5], 3.0, 30);
        simple_heap_push(&mut d[..5], &mut idx[..5], 1.0, 10);
        simple_heap_push(&mut d[..5], &mut idx[..5], 4.0, 40);
        simple_heap_push(&mut d[..5], &mut idx[..5], 2.0, 20);

        deheap_sort(&mut heap.indices, &mut heap.distances);

        // Should be sorted ascending
        let sorted_dists: Vec<f32> = (0..5).map(|j| heap.distances[[row, j]]).collect();
        let sorted_inds: Vec<i32> = (0..5).map(|j| heap.indices[[row, j]]).collect();
        assert_eq!(sorted_dists, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(sorted_inds, vec![10, 20, 30, 40, 50]);
    }
}
