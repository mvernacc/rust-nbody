/// Gravitational constant [units: N m^2 kg^-1].
const G: f64 = 6.6743015e-11;

/// Calculate the gravitational force of body 2 on body 1.
pub fn calc_grav_force_two_bodies(mass_1: f64, mass_2: f64, r_1: [f64; 3], r_2: [f64; 3]) -> [f64; 3] {
    // Position vector from body 1 center to body 2 center [units: m].
    let r_1to2 = [r_1[0] - r_2[0], r_1[1] - r_2[1], r_1[2] - r_2[2]];
    // Distance between body 1 and 2, to the third power [units: m^3]
    let distance_cubed =
        (r_1to2[0] * r_1to2[0] + r_1to2[1] * r_1to2[1] + r_1to2[2] * r_1to2[2]).powf(3.0 / 2.0);
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
