// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

fn polygon_bounding_rect(polygons: &[Vec<(f64, f64)>]) -> (f64, f64, f64, f64) {
    let mut min_x: f64 = f64::INFINITY;
    let mut min_y: f64 = f64::INFINITY;
    let mut max_x: f64 = -f64::INFINITY;
    let mut max_y: f64 = -f64::INFINITY;
    for polygon in polygons.iter() {
        for &(x, y) in polygon.iter() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    (min_x, min_y, max_x, max_y)
}

fn polygon_area(polygon: &[(f64, f64)]) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..polygon.len() {
        let (x1, y1) = polygon[i];
        let (x2, y2) = polygon[(i + 1) % polygon.len()];
        sum += x1 * y2 - x2 * y1;
    }
    sum.abs() / 2.0
}

fn polygon_rect_overlapping_area(polygon: &[(f64, f64)], rect: &(f64, f64, f64, f64)) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..polygon.len() {
        let (mut x1, mut y1) = polygon[i];
        let (mut x2, mut y2) = polygon[(i + 1) % polygon.len()];
        x1 = x1.clamp(rect.0, rect.2);
        x2 = x2.clamp(rect.0, rect.2);
        y1 = y1.clamp(rect.1, rect.3);
        y2 = y2.clamp(rect.1, rect.3);
        sum += x1 * y2 - x2 * y1;
    }
    sum.abs() / 2.0
}

fn rects_from_multi_polygon_recursion2(
    shape: &[Vec<(f64, f64)>],
    _total_area: f64,
    rect: (f64, f64, f64, f64),
    level: u8,
    output: &mut Vec<(f64, f64, f64, f64)>,
) -> bool {
    let intersection_area: f64 = shape
        .iter()
        .map(|s| polygon_rect_overlapping_area(s, &rect))
        .sum();
    let rect_area = (rect.2 - rect.0) * (rect.3 - rect.1);
    if intersection_area > rect_area * 0.98 {
        output.push(rect);
        true
    } else if intersection_area <= rect_area * 0.02 {
        false
    } else if level <= 11 {
        let [r1, r2] = if level.is_multiple_of(2) {
            let mid = (rect.0 + rect.2) / 2.0;
            [(rect.0, rect.1, mid, rect.3), (mid, rect.1, rect.2, rect.3)]
        } else {
            let mid = (rect.1 + rect.3) / 2.0;
            [(rect.0, rect.1, rect.2, mid), (rect.0, mid, rect.2, rect.3)]
        };
        let cnt = output.len();
        let r1 = rects_from_multi_polygon_recursion2(shape, _total_area, r1, level + 1, output);
        let r2 = rects_from_multi_polygon_recursion2(shape, _total_area, r2, level + 1, output);
        if r1 && r2 {
            output.truncate(cnt);
            output.push(rect);
            true
        } else {
            false
        }
    } else {
        true
    }
}

fn union_two_rects(
    r1: &(f64, f64, f64, f64),
    r2: &(f64, f64, f64, f64),
) -> Option<(f64, f64, f64, f64)> {
    if r1.0 == r2.0 && r1.2 == r2.2 {
        if r1.3 == r2.1 {
            return Some((r1.0, r1.1, r1.2, r2.3));
        } else if r2.3 == r1.1 {
            return Some((r1.0, r2.1, r1.2, r1.3));
        }
    } else if r1.1 == r2.1 && r1.3 == r2.3 {
        if r1.2 == r2.0 {
            return Some((r1.0, r1.1, r2.2, r1.3));
        } else if r2.2 == r1.0 {
            return Some((r2.0, r1.1, r1.2, r1.3));
        }
    }
    None
}

fn rects_from_multi_polygon2(shape: &[Vec<(f64, f64)>]) -> Vec<(f64, f64, f64, f64)> {
    let mut rects: Vec<(f64, f64, f64, f64)> = Vec::new();

    let rect = polygon_bounding_rect(shape);

    let total_area: f64 = shape.iter().map(|s| polygon_area(s)).sum();
    rects_from_multi_polygon_recursion2(shape, total_area, rect, 0, &mut rects);

    // Attempt to union adjacent rects.
    loop {
        let mut did_union = false;

        'outer: for i1 in 0..rects.len() {
            for i2 in i1 + 1..rects.len() {
                let r1 = rects[i1];
                let r2 = rects[i2];
                if let Some(r) = union_two_rects(&r1, &r2) {
                    rects.remove(i2); // i2 must be removed first because it's latter in the array.
                    rects.remove(i1);
                    rects.push(r);
                    did_union = true;
                    break 'outer;
                }
            }
        }

        if !did_union {
            break;
        }
    }

    rects
}

pub fn fit_rects_from_polygons(polygons: &[Vec<(f64, f64)>]) -> Vec<(f64, f64, f64, f64)> {
    rects_from_multi_polygon2(polygons)
}
