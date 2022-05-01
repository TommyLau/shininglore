use std::convert::TryInto;
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
        let mut mag = self.magnitude();
        if mag == 0.0 {
            // Set normal to ShiningLore Model +Z Up (0, 1, 1) if normal was Zero
            self.z = 1.0;
            return;
        }
        while mag < f32::EPSILON {
            // Add check to avoid NaN error!
            self.multiply(10.0);
            mag = self.magnitude();
        }
        self.multiply(1.0f32 / mag)
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

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
    }

    pub fn multiply(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self.w *= s;
    }

    pub fn as_array(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    pub fn from_array([x, y, z, w]: [f32; 4]) -> Self {
        Self { x, y, z, w }
    }
}

impl ops::Add for Vec4 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl ops::Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(mut self, rhs: f32) -> Self::Output {
        self.multiply(rhs);
        self
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    pub x: Vec4,
    pub y: Vec4,
    pub z: Vec4,
    pub w: Vec4,
}

impl Mat4 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        [
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3
        ]: [f32; 16],
    ) -> Mat4 {
        Mat4 {
            x: Vec4::new(c0r0, c0r1, c0r2, c0r3),
            y: Vec4::new(c1r0, c1r1, c1r2, c1r3),
            z: Vec4::new(c2r0, c2r1, c2r2, c2r3),
            w: Vec4::new(c3r0, c3r1, c3r2, c3r3),
        }
    }

    pub fn from_array([x, y, z, w]: [[f32; 4]; 4]) -> Mat4 {
        Mat4 {
            x: Vec4::from_array(x),
            y: Vec4::from_array(y),
            z: Vec4::from_array(z),
            w: Vec4::from_array(w),
        }
    }

    pub fn as_array(&self) -> [f32; 16] {
        matrix_to_array(
            [
                self.x.as_array(),
                self.y.as_array(),
                self.z.as_array(),
                self.w.as_array(),
            ]
            .concat(),
        )
    }
}

impl ops::Mul<Mat4> for Mat4 {
    type Output = Mat4;
    fn mul(self, rhs: Mat4) -> Self::Output {
        let a = self.x;
        let b = self.y;
        let c = self.z;
        let d = self.w;
        Mat4 {
            x: a * rhs.x.x + b * rhs.x.y + c * rhs.x.z + d * rhs.x.w,
            y: a * rhs.y.x + b * rhs.y.y + c * rhs.y.z + d * rhs.y.w,
            z: a * rhs.z.x + b * rhs.z.y + c * rhs.z.z + d * rhs.z.w,
            w: a * rhs.w.x + b * rhs.w.y + c * rhs.w.z + d * rhs.w.w,
        }
    }
}

fn matrix_to_array<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}
