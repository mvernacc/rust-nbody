/// Gravitational constant [units: N m^2 kg^-1].
const G: f64 = 6.6743015e-11;

/// Calculate the gravitational force of body 2 on body 1.
///
/// The position vectors must be given in the same reference frame, and the
/// force vector is returned in that frame.
///
/// Arguments:
/// * `mass_1`: Mass of body 1 [units: kg].
/// * `mass_2`: Mass of body 2 [units: kg].
/// * `r_1`: Position vector of body 1 [units: m].
/// * `r_2`: Position vector of body 2 [units: m].
pub fn calc_grav_force_two_bodies(
    mass_1: f64,
    mass_2: f64,
    r_1: [f64; 3],
    r_2: [f64; 3],
) -> [f64; 3] {
    // Position vector from body 1 center to body 2 center [units: m].
    let r_1to2: Vec<f64> = r_1.iter().zip(r_2.iter()).map(|(x1, x2)| x1 - x2).collect();
    // Distance between body 1 and 2, to the third power [units: m^3]
    let distance_cubed = r_1to2
        .iter()
        .fold(0.0, |accum, item| accum + item * item)
        .sqrt()
        .powi(3);
    let gmm_over_dist_cubed = G * mass_1 * mass_2 / distance_cubed;
    // Gravitational force of body 2 on body 1 [units: N].
    [
        -gmm_over_dist_cubed * r_1to2[0],
        -gmm_over_dist_cubed * r_1to2[1],
        -gmm_over_dist_cubed * r_1to2[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn earth_surface_weight() {
        // Setup
        let mass_earth = 5.9722e24; // [kg]
        let radius_earth = 6371e3; // [m]
        let one_kg = 1.0;
        let g = 9.807; // [m s^-2]

        // Action
        let force = calc_grav_force_two_bodies(
            one_kg,
            mass_earth,
            [0.0, 0.0, radius_earth],
            [0.0, 0.0, 0.0],
        );

        // Verification
        // Force on body one should be [0, 0, -one_kg * g]
        assert_eq!(force[0], 0.0);
        assert_eq!(force[1], 0.0);
        assert_relative_eq!(force[2], -one_kg * g, max_relative = 1e-2);
    }
}
