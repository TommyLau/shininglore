use std::ops;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(v: [f32; 3]) -> Self {
        Vec3 {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }

    pub fn multiply(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }

    // Vector 3 Cross
    pub fn cross(self, other: Self) -> Self {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        // Add check to avoid NaN error!
        if mag > f32::EPSILON {
            self.multiply(1.0f32 / mag)
        }
    }

    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn into(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl ops::Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl ops::AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(mut self, other: f32) -> Self::Output {
        self.multiply(other);
        self
    }
}
