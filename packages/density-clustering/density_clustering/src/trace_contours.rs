// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

use std::collections::{BinaryHeap, HashMap};

use crate::array_2d::Array2D;

pub fn trace_outer_contour<T: Eq + Copy>(start: (i32, i32), array: &Array2D<T>) -> Vec<(i32, i32)> {
    let width = array.width();
    let height = array.height();

    let inner = array[start];

    let search_directions = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    let read_offsets = [(0, -1), (0, 0), (-1, 0), (-1, -1)];
    let mut result: Vec<(i32, i32)> = Vec::new();
    result.push(start);

    let mut location = (start.0, start.1);
    let mut start_direction = 0;
    loop {
        let (lx, ly) = location;
        for j in 0..4 {
            let d = (start_direction + j) & 3;
            let (vx, vy) = read_offsets[d];
            let (rx, ry) = (lx + vx, ly + vy);
            if rx < 0 || rx >= width || ry < 0 || ry >= height {
                continue;
            }
            if array[(rx, ry)] == inner {
                let (dx, dy) = search_directions[d];
                location = (lx + dx, ly + dy);
                start_direction = (d + 3) & 3;
                break;
            }
        }
        if location == start {
            break;
        }
        result.push(location);
    }
    result
}

fn fill_contour<T: Eq + Copy>(array: &mut Array2D<T>, contour: &[(i32, i32)], value: T) {
    // Contour is expected to be what returned from trace_outer_contour.
    let mut edges = HashMap::<i32, BinaryHeap<std::cmp::Reverse<i32>>>::new();

    for i in 0..contour.len() {
        let p1 = contour[i];
        let p2 = contour[(i + 1) % contour.len()];
        if p1.1 != p2.1 {
            assert!(p1.0 == p2.0);
            let x = p1.0;
            for y in p1.1.min(p2.1)..p1.1.max(p2.1) {
                edges.entry(y).or_default().push(std::cmp::Reverse(x));
            }
        }
    }

    for (y, mut heap) in edges {
        let mut inside = false;
        let mut current_x = 0;
        while let Some(rx) = heap.pop() {
            let x = rx.0;
            inside = !inside;
            if inside {
                current_x = x;
            } else {
                for i in current_x..x {
                    array[(i, y)] = value;
                }
            }
        }
    }
}

pub fn trace_all_outer_contours(array: &Array2D<i32>) -> HashMap<i32, Vec<Vec<(i32, i32)>>> {
    let width = array.width();
    let height = array.height();
    let mut result: HashMap<i32, Vec<Vec<(i32, i32)>>> = HashMap::new();
    let mut mask = Array2D::<bool>::new_with_constant(width, height, true);

    for (x, y) in array.iter_coords() {
        let v = array[(x, y)];
        if v >= 0 && mask[(x, y)] {
            let contour = trace_outer_contour((x, y), array);
            fill_contour(&mut mask, &contour, false);
            result.entry(v).or_default().push(contour);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_contours() {
        let array = Array2D::<i32>::new(
            4,
            4,
            vec![
                0, 0, 0, 0, //
                0, 1, 1, 1, //
                0, 1, 0, 1, //
                0, 0, 0, 0, //
            ],
        );
        assert_eq!(
            trace_outer_contour((1, 1), &array),
            [
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (4, 2),
                (4, 3),
                (3, 3),
                (3, 2),
                (2, 2),
                (2, 3),
                (1, 3),
                (1, 2)
            ]
        );
    }

    #[test]
    fn test_trace_contours_diagonal() {
        let array = Array2D::<i32>::new(
            4,
            4,
            vec![
                0, 0, 0, 0, //
                0, 1, 0, 0, //
                0, 0, 1, 0, //
                0, 0, 0, 0, //
            ],
        );
        assert_eq!(
            trace_outer_contour((1, 1), &array),
            [
                (1, 1),
                (2, 1),
                (2, 2),
                (3, 2),
                (3, 3),
                (2, 3),
                (2, 2),
                (1, 2)
            ]
        );
    }

    #[test]
    fn test_fill_contour() {
        let mut array = Array2D::<i32>::new(
            4,
            4,
            vec![
                0, 0, 0, 0, //
                0, 1, 1, 1, //
                0, 1, 0, 1, //
                0, 0, 0, 0, //
            ],
        );
        let contour = trace_outer_contour((1, 1), &array);
        fill_contour(&mut array, &contour, 2);
    }
}
