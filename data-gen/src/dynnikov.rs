use crate::utilities::pos;
use crate::utilities::neg;

// these next two functions are derived from Thiffeault ch8
// it's a kind of involved piecewise function
// so apologies if it isn't nice to read

/// Updates a dynnikov coordinate based on a generator.
fn pve_sigma_action(coord: &Vec<i64>, sigma: i64) -> Vec<i64> {
    let n = (coord.len() as i64+4)/2;
    let i = sigma as usize;

    // denotes some key places in the coordinate vector
    // lot stands for last of type
    let lot = (n-3) as usize;
    let a_start: usize = 0;
    let b_start: usize = lot + 1;

    let mut new_coord = coord.to_vec();

    // deal with edge cases
    if sigma == 1 {
        let a1 = coord[0];
        let b1 = coord[b_start];

        new_coord[a_start] = -b1 + pos(a1 + pos(b1));
        new_coord[b_start] = a1 + pos(b1);
    } else if sigma == n-1 {
        let a_last = coord[a_start + lot];
        let b_last = coord[b_start + lot];

        new_coord[a_start + lot] = -b_last + neg(a_last + neg(b_last));
        new_coord[b_start + lot] = a_last + neg(b_last);
    } else {
        // the normal case
        let aisub1 = coord[a_start + i - 2];
        let bisub1 = coord[b_start + i - 2];
        let ai = coord[a_start + i - 1];
        let bi = coord[b_start + i - 1];

        // technically this is c_{i-1}, oh well
        let c = aisub1 - ai - pos(bi) + neg(bisub1);

        new_coord[a_start + i - 2] = aisub1 - pos(bisub1) - pos(pos(bi) + c);
        new_coord[b_start + i - 2] = bi + neg(c);
        new_coord[a_start + i - 1] = ai - neg(bi) - neg(neg(bisub1) - c);
        new_coord[b_start + i - 1] = bisub1 - neg(c);
    }

    return new_coord;
}

/// Updates a dynnikov coordinate based on an inverse generator.
fn nve_sigma_action(coord: &Vec<i64>, sigma: i64) -> Vec<i64> {
    let n = (coord.len() as i64+4)/2;
    let i = sigma as usize;

    // denotes some key places in the coordinate vector
    // lot stands for last of type
    let lot = (n-3) as usize;
    let a_start: usize = 0;
    let b_start: usize = lot + 1;

    let mut new_coord = coord.to_vec();

    // deal with edge cases
    if sigma == 1 {
        let a1 = coord[0];
        let b1 = coord[b_start];

        new_coord[a_start] = b1 - pos(pos(b1) - a1);
        new_coord[b_start] = pos(b1) - a1;
    } else if sigma == n-1 {
        let a_last = coord[a_start + lot];
        let b_last = coord[b_start + lot];

        new_coord[a_start + lot] = b_last - neg(neg(b_last) - a_last);
        new_coord[b_start + lot] = neg(b_last) - a_last;
    } else {
        // the normal case
        let aisub1 = coord[a_start + i - 2];
        let bisub1 = coord[b_start + i - 2];
        let ai = coord[a_start + i - 1];
        let bi = coord[b_start + i -1];

        // technically this is d_{i-1}, oh well
        let d = aisub1 - ai + pos(bi) - neg(bisub1);

        new_coord[a_start + i - 2] = aisub1 + pos(bisub1) + pos(pos(bi) - d);
        new_coord[b_start + i - 2] = bi - pos(d);
        new_coord[a_start + i - 1] = ai + neg(bi) + neg(neg(bisub1) + d);
        new_coord[b_start + i - 1] = bisub1 + pos(d);
    }

    return new_coord;
}

/// Computes the action of a braid word on a dynnikov coordinate.
/// Braids act left to right.
/// 4 is the action s_4, -4 is the action s_4^{-1}
pub fn word_action(coord: &Vec<i64>, word: &Vec<i64>) -> Vec<i64> {
    let mut new_coord = coord.to_vec();

    for sigma in word.iter() {
        if *sigma == 0 {
            // 0 is the identity, do nothing
            continue;
        } else if *sigma > 0 {
            // +ve version of generators
            new_coord = pve_sigma_action(&new_coord, *sigma);
        } else {
            // -ve version of generators
            new_coord = nve_sigma_action(&new_coord, -*sigma);
        }
    }

    return new_coord;
}