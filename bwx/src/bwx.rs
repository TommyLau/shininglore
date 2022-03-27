use std::io::{Cursor, Read};
use std::path::Path;
use std::{mem, fs};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use tracing::{debug, error, info, trace, warn};
use serde::Serialize;
use cgmath::*;
use gltf::Gltf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
#[derive(Debug, Clone, Serialize)]
pub enum SlType {
    UChar(u8),
    Char(i8),
    Word(i16),
    Int(i32),
    Float(f32),
    String(String),
    Data(Vec<u8>),
    Array(Vec<SlType>),
    DArray(String, Vec<SlType>),
    None,
}

impl Default for SlType {
    fn default() -> Self {
        SlType::None
    }
}

impl SlType {
    pub fn int(&self) -> Result<i32> {
        match self {
            SlType::UChar(v) => Ok(*v as i32),
            SlType::Char(v) => Ok(*v as i32),
            SlType::Word(v) => Ok(*v as i32),
            SlType::Int(v) => Ok(*v),
            _ => Err("Cannot get integer from SlType".into()),
        }
    }

    pub fn float(&self) -> Result<f32> {
        if let SlType::Float(v) = self {
            Ok(*v)
        } else {
            Err("Cannot get float from SlType".into())
        }
    }

    pub fn string(&self) -> Result<String>
    {
        match self {
            SlType::String(v) | SlType::DArray(v, _) => { Ok(v.into()) }
            _ => Err("Cannot get string from SlType".into())
        }
    }

    pub fn data(&self) -> Result<&Vec<u8>> {
        if let SlType::Data(v) = self {
            Ok(v)
        } else {
            debug!("Fuck: {:#?}", self);
            Err("Cannot get data from SlType".into())
        }
    }

    pub fn array(&self) -> Result<&Vec<SlType>> {
        if let SlType::Array(v) = self {
            Ok(v)
        } else {
            Err("Cannot get array from SlType".into())
        }
    }

    pub fn d_array(&self) -> Result<&Vec<SlType>> {
        if let SlType::DArray(_, v) = self {
            Ok(v)
        } else {
            Err("Cannot get d-array from SlType".into())
        }
    }
}

// Block 'HEAD'
pub struct Head {
    pub name: String,
    pub description: String,
    pub magic: i32,
    pub version: i16,
    pub other: String,
}

pub struct SubMaterial {
    pub highlight: f32,
    pub filename: String,
    pub used: bool,
}

// Block 'MTRL'
pub struct Material {
    pub sub_materials: Vec<SubMaterial>,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coord: [f32; 2],
}

#[derive(Debug, Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    content: Cursor<Vec<u8>>,
    pub data: SlType,
    version: i32,
    vertices: Vec<Vertex>,
}

pub fn print_matrix<T>(m: &Matrix4<T>)
    where
        T: std::fmt::Display,
{
    println!("{0:<30}{1:<30}{2:<30}{3:<30}", m.x.x, m.y.x, m.z.x, m.w.x);
    println!("{0:<30}{1:<30}{2:<30}{3:<30}", m.x.y, m.y.y, m.z.y, m.w.y);
    println!("{0:<30}{1:<30}{2:<30}{3:<30}", m.x.z, m.y.z, m.z.z, m.w.z);
    println!("{0:<30}{1:<30}{2:<30}{3:<30}", m.x.w, m.y.w, m.z.w, m.w.w);
}

impl BWX {
    /// Returns a BWX with the given file name
    ///
    // /// TODO: Merge load_from_file() to new()
    // /// # Arguments
    // ///
    // /// * `path` - A string slice that holds the file name of a BNX / PNX file
    // ///
    /// # Examples
    ///
    /// ```
    /// use bwx::BWX;
    /// let bwx = BWX::new();
    /// ```
    pub fn new() -> Self {
        BWX {
            ..Default::default()
        }
    }

    /// Load BWX file from file
    #[tracing::instrument(skip(self, filename))]
    //pub fn load_from_file(&mut self, filename: &str) -> Result<()> {
    pub fn load_from_file<T>(&mut self, filename: T) -> Result<()>
        where T: AsRef<Path>
    {
        /*
        // Now I got the idea of how to inverse the matrix
        // From:
        // https://github.com/toji/gl-matrix/issues/408
        let mmm = gltf::scene::Transform::Matrix {
            matrix: [
                [0.04151877760887146, 0.020675182342529297, -6.616836412121074e-9, 0.0, ],
                [6.616837744388704e-9, 1.55635282439448e-9, 0.04638181999325752, 0.0, ],
                [-0.020675182342529297, 0.04151877760887146, 1.5563523803052703e-9, 0.0, ],
                [0.0, 0.0, 0.0, 1.0]
            ]
        };
        let m: [[f64; 4]; 4] = [
            [0.6007744, -1.0449057e-6, 9.0477204e-7, 0.0, ],
            [1.0107502e-6, 0.5811367, 9.953766e-9, 0.0, ],
            [-1.6252185e-6, -1.7125686e-8, 1.0, 0.0, ],
            [-1.730051e-9, -0.7697969, -0.019488811, 1.0, ]
        ];
        let m2: [[f32; 4]; 4] = [
            [0.6007744, -1.0449057e-6, 9.0477204e-7, 0.0, ],
            [1.0107502e-6, 0.5811367, 9.953766e-9, 0.0, ],
            [-1.6252185e-6, -1.7125686e-8, 1.0, 0.0, ],
            [-1.730051e-9, -0.7697969, -0.019488811, 1.0, ]
        ];

        let translation = Vector3 { x: m[3][0], y: m[3][1], z: m[3][2] };
        debug!("Translation: {:#?}", translation);
        let mut i = Matrix3 {
            x: Vector3 { x: m[0][0], y: m[0][1], z: m[0][2] },
            y: Vector3 { x: m[1][0], y: m[1][1], z: m[1][2] },
            z: Vector3 { x: m[2][0], y: m[2][1], z: m[2][2] },
        };
        let sx = i.x.magnitude();
        let sy = i.y.magnitude();
        let sz = i.determinant().signum() * i.z.magnitude();
        let scale = [sx, sy, sz];
        debug!("Scale: {:#?}", scale);
        /*
        i.x *= 1.0 / sx;
        i.y *= 1.0 / sy;
        i.z *= 1.0 / sz;

         */
        let q = Quaternion::from(i);
        let rotation = [q.v.x, q.v.y, q.v.z, q.s];
        debug!("Rotation: {:#?}", rotation);

        let t = Matrix4::from_translation(translation);
        let r = Matrix4::from(q);
        let s = Matrix4::from_nonuniform_scale(sx, sy, sz);
        //debug!("T: {:#?}", t);
        //debug!("R: {:#?}", r);
        //debug!("S: {:#?}", r);
        let x = t * r * s;

        debug!("Matrix:");
        let mm = Matrix4::from(m);
        print_matrix(&mm);
        debug!("Calculated Matrix:");
        print_matrix(&x);

        // Calculate f32
        let t = Vector3 { x: -1.730051e-9f32, y: -0.7697969f32, z: -0.019488811f32 };
        let q = Quaternion {
            v: Vector3 { x: (q.v.x as f32), y: (q.v.y as f32), z: (q.v.z as f32) },
            s: (q.s as f32),
        };
        let (sx, sy, sz) = (sx as f32, sy as f32, sz as f32);
        let t = Matrix4::from_translation(t);
        let r = Matrix4::from(q);
        let s = Matrix4::from_nonuniform_scale(sx, sy, sz);
        let x = t * r * s;

        debug!("Calculated Matrix in f32:");
        print_matrix(&x);


        // Library calculation
        let m = gltf::scene::Transform::Matrix { matrix: m2 };
        let (translation, rotation, scale) = m.decomposed();
        let mm = gltf::scene::Transform::Decomposed { translation, rotation, scale };
        let mm = Matrix4::from(mm.matrix());
        debug!("Library Matrix in f32:");
        print_matrix(&mm);

        return Ok(());
        let m4 = mmm.matrix();
        let m3 = Matrix3 {
            x: Vector3 { x: m4[0][0], y: m4[0][1], z: m4[0][2] },
            y: Vector3 { x: m4[1][0], y: m4[1][1], z: m4[1][2] },
            z: Vector3 { x: m4[2][0], y: m4[2][1], z: m4[2][2] },
        };
        //debug!("M3: {:#?}", m3);
        let sx = m3.x.magnitude();
        let sy = m3.y.magnitude();
        let sz = m3.z.magnitude();
        let t_s = [sx, sy, sz];
        debug!("T_S: {:#?}", t_s);
        let mut nx = m3.x * 1.0 / sx;
        let ny = m3.y * 1.0 / sy;
        let nz = m3.z * 1.0 / sz;
        let mut nr = Matrix3 { x: nx, y: ny, z: nz };
        debug!("N_R: {:#?}, ------- aaa: ", nr);
        let aaa = nx.cross(ny).dot(nz);
        if aaa < 0.0 {
            nx = nx * -1.0;
            nr = Matrix3 { x: nx, y: ny, z: nz };
            debug!("New N_R: {:#?}, ------- aaa: {}", nr, aaa);
        }


        let mut t_r = Quaternion::from(nr).normalize();
        debug!("T_R: {:#?}", t_r);
        let t_t = Vector3 { x: m4[3][0], y: m4[3][1], z: m4[3][2] };
        debug!("T_T: {:#?}", t_t);
        let t = Matrix4::from_translation(t_t);
        let r = Matrix4::from(t_r);
        let s = Matrix4::from_nonuniform_scale(sx, sy, sz);
        let x = t * r * s;
        debug!("My Calc: {:#?}", x);
        let m = gltf::scene::Transform::Matrix { matrix: m4 };
        let (translation, rotation, scale) = m.decomposed();
        debug!("Decompose rotation: {:#?}", rotation);
        let dd = gltf::scene::Transform::Decomposed { translation, rotation, scale };
        let d = gltf::scene::Transform::Decomposed {
            translation: t_t.into(),
            rotation: [
                0.1619011,
                -0.68832266,
                0.6883227,
                -0.161901,
            ],
            scale: t_s.into(),
        };
        let matrix = d.matrix();
        //        debug!("new matrix: {:#?}", matrix);
        debug!("old matrix: {:#?}", m4);
        let m3 = dd.matrix();
        debug!("round matrix: {:#?}", m3);


        let om = Matrix4 {
            x: Vector4 { x: m4[0][0], y: m4[0][1], z: m4[0][2], w: m4[0][3] },
            y: Vector4 { x: m4[1][0], y: m4[1][1], z: m4[1][2], w: m4[1][3] },
            z: Vector4 { x: m4[2][0], y: m4[2][1], z: m4[2][2], w: m4[2][3] },
            w: Vector4 { x: m4[2][0], y: m4[2][1], z: m4[2][2], w: m4[2][3] },
        };
        let v = Vector4 { x: 3.0, y: 4.0, z: 5.0, w: 1.0 };
        let t1 = om * v;
        debug!("Test Orig: {:#?}", t1);
        let t = Matrix4::from_translation(translation.into());
        let rotation = Quaternion {
            v: Vector3 {
                x: rotation[0],
                y: rotation[1],
                z: rotation[2],
            },
            s: rotation[3],
        };
        let r = Matrix4::from(rotation);
        let s = Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2]);
        let t2 = t * r * s * v;
        debug!("Test Decompose: {:#?}", t2);
        let m3 = Matrix4::from(m3);
        let t3 = m3 * v;
        debug!("T3 round Decompose: {:#?}", t3);

         */


        /*
        let mmm = gltf::scene::Transform::Matrix { matrix: m4 };
        let (translation, rotation, scale) = mmm.decomposed();
        debug!("t: {:#?}", translation);
        debug!("r: {:#?}", rotation);
        debug!("s: {:#?}", scale);
        let decomposed = gltf::scene::Transform::Decomposed { translation, rotation, scale };
        let matrix = decomposed.matrix();
        debug!("new matrix: {:#?}", matrix);

         */


        //return Ok(());


        info!("{}", filename.as_ref().display());

        let data = std::fs::read(filename)?;
        self.content = Cursor::new(data);

        self.check_bwx_header()?;
        self.content.set_position(4);
        self.data = self.go_through(true)?;

        // Test obj code
        use std::io::Write;
        let mut output = vec![];
        writeln!(output, "# ShiningLore Online Development Team (SLODT)")?;
        writeln!(output, "# Tommy Lau <tommy.lhg@gmail.com>")?;
        // Test

        for node in self.data.array().unwrap() {
            let name = node.string()?;
            let children = node.d_array()?;
            match name.as_str() {
                "0" => {
                    // Default block "0" with string "SLBWX"
                }
                "HEAD" => {
                    if children.len() >= 4 {
                        // 0 - HEAD
                        // 1 - head block
                        // 2 - PNX\0
                        // 3 - 0x0500: SL1, 0x0602: SL2
                        // 4 - "BWX PNX KAK"
                        if children[0].string().unwrap().as_str() != "HEAD" {
                            error!("Incorrect HEAD block");
                        }
                        if children[2].int().unwrap() != 0x504e5800 {
                            error!("Header magic != PNX");
                        }
                        self.version = children[3].int()?;
                        match self.version {
                            0x500 => trace!("ShiningLore V1 PNX"),
                            0x602 => trace!("ShiningLore V2 PNX"),
                            _ => error!("Unknown ShiningLore PNX version!!"),
                        }
                    } else {
                        warn!("HEAD block length < 4, no PNX version available!");
                    }
                }
                "MTRL" => {
                    // TODO: Save material information in struct
                    for materials in children {
                        let material = materials.array()?;
                        // material[0] - Material Group "MTRL"
                        // material[1] - Material Group Name
                        // material[2..n] - Material Array for Sub Materials
                        trace!("Material: {}", material[1].string()?);
                        for (i, sub_materials) in material.iter().enumerate().skip(2) {
                            let sub_material = sub_materials.array()?;
                            let highlight = sub_material[5].float()?;
                            // 0 - "SUBMTRL"
                            // 1 - Diffuse ???
                            // 2 - Ambient ???
                            // 3 - Specular ???
                            // 4 - Some float ???
                            // 5 - High light
                            // 6 - Most 0x01
                            // 7 - ???
                            // 8 - Texture Array
                            let filename = if sub_material.len() > 8 {
                                // Some materials have no texture, such as glass
                                let texture = sub_material[8].array()?;
                                // 0 - TEX
                                // 1 - Most 0x00, timer?
                                // 2 - Filename
                                let filename = texture[2].string()?;
                                filename.split('\\').last().unwrap().into()
                            } else { "".to_string() };
                            trace!("\tSub Material {}: Highlight: {}, File: {}", i - 2, highlight, filename);
                        }
                    }
                }
                // TODO: Parse OBJ2 mesh data from SL1
                "OBJ2" => {
                    warn!("OBJ2 parsing needs to be implemented! {}@{}", file!(), line!());
                }
                "OBJECT" => {}
                "CAM" => {}
                "LIGHT" => {}
                "SOUND" => {}
                "BONE" => {}
                "CHART" => {}
                "DXOBJ" | "SPOB" => {
                    // TODO: Store parsed data into struct
                    if children.is_empty() {
                        warn!("No data block found in {}", name);
                        continue;
                    }

                    // Index for OBJ output, vertex index starts from 1 in OBJ
                    let mut idx: u32 = 1;

                    for objects in children {
                        let object = objects.array()?;
                        // 0 - "DXOBJ" / "SPOB"
                        // 1 - Mesh Name
                        // 2 - Unknown integer
                        // 3 - Texture Group Index : -1 means no texture
                        // 4, 5 - Unknown
                        // 6 - 0x4D534858h("MSHX") or 0x4D4E4858h("MNHX")
                        // 7 - Array("DXMESH")
                        // 8 - Array("MATRIX")
                        let name = object[1].string()?;
                        let texture_index = object[3].int()?;
                        writeln!(output, "o {}", name)?;
                        trace!("Object: {}, Index: {}", name, texture_index);
                        //------------------------------
                        // Get only the first matrix
                        let matrix = {
                            let matrix = object[8].array()?[0].array()?[1].data()?;
                            let mut buffer = Cursor::new(matrix);
                            let _timeline = buffer.read_u32::<LittleEndian>()?;
                            Matrix4::new(
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                                buffer.read_f32::<LittleEndian>()?,
                            )
                        };
                        //debug!("{:#?}", matrix);
                        // ========================================
                        // Meshes
                        let meshes = object[7].array()?;
                        for mesh_array in meshes {
                            let mesh = mesh_array.array()?;
                            // 0 - "DXMESH"
                            // 1 - Texture Index in Texture Group
                            // 2 - Array("DXMESHF")
                            // 3 - Index Buffer Size
                            // 4 - Index Buffer
                            let sub_texture_index = mesh[1].int()?;
                            let index_count = mesh[3].int()?;
                            let index_buffer = mesh[4].data()?;
                            if index_buffer.len() != index_count as usize * 2 {
                                error!("Index block size incorrect!");
                            }
                            trace!("\tMesh: [Texture Index: {}, Index Count: {}, Size: {}",
                                sub_texture_index, index_count, index_buffer.len());
                            let blocks = mesh[2].array()?;
                            for vertices in blocks {
                                let vertex = vertices.array()?;
                                // 0 - "DXMESHF"
                                // 1 - VB Timer
                                // 2 - Vertex Type??? - 0x15
                                // 3 - Vertex Count
                                // 4 - Vertex Size - 0x20
                                // 5 - Vertex Buffer
                                let timer = vertex[1].int()?;
                                let vertex_type = vertex[2].int()?;
                                let vertex_count = vertex[3].int()?;
                                let vertex_size = vertex[4].int()?;
                                let vertex_buffer = vertex[5].data()?;
                                trace!("\t\tVertex: [Timer: {}, Type: {}, Count: {}, Size: {}, BufLen: {}",
                                timer, vertex_type, vertex_count, vertex_size, vertex_buffer.len());

                                // Test
                                let mut v_buffer = Cursor::new(vertex_buffer);
                                let mut v = Vec::new();
                                let mut vn = Vec::new();
                                let mut vt = Vec::new();
                                for _ in 0..vertex_count {
                                    let x = v_buffer.read_f32::<LittleEndian>()?;
                                    let y = v_buffer.read_f32::<LittleEndian>()?;
                                    let z = v_buffer.read_f32::<LittleEndian>()?;
                                    let vv = Vector4::new(x, y, z, 1.0);
                                    // Normal code
                                    /*
                                    // let t = matrix * vv;
                                    // writeln!(output, "v {} {} {}", t.x, t.y, t.z)?;
                                     */
                                    // ??? DirectX and OpenGL got different Z-Axis
                                    // ??? Since the game was originally develop for DirectX
                                    // ??? Reverse the Z-Axis to fit OpenGL spec
                                    //
                                    // Add rotation to fit Blender?!
                                    // Method 1, change (x,y,z) -> (x,z,-y)
                                    //writeln!(output, "v {} {} {}", t.x, t.z, -t.y)?;
                                    // Method 2, rotate -90 degrees along x-axis
                                    let rot = Matrix4::from_angle_x(Rad(-90.0f32.to_radians()));
                                    let t = rot * matrix * vv;
                                    writeln!(output, "v {} {} {}", t.x, t.y, t.z)?;
                                    // End Blender rotation
                                    let position = [t.x, t.y, t.z];
                                    v.push(position);
                                    let normal = [
                                        v_buffer.read_f32::<LittleEndian>()?,
                                        v_buffer.read_f32::<LittleEndian>()?,
                                        v_buffer.read_f32::<LittleEndian>()?,
                                    ];
                                    vn.push(normal);
                                    let tex_coord = [
                                        v_buffer.read_f32::<LittleEndian>()?,
                                        v_buffer.read_f32::<LittleEndian>()?,
                                    ];
                                    vt.push(tex_coord);
                                    self.vertices.push(Vertex {
                                        position,
                                        normal,
                                        tex_coord,
                                    });
                                }
                                /*
                                for vv in v {
                                    //writeln!(output, "v {} {} {}", vv[0], vv[1], vv[2])?;
                                    // Implement Matrix transformation
                                    //debug!("Before: [{}, {}, {}]", vv[0],vv[1],vv[2]);
                                    let v = Vector4::new(vv[0], vv[1], vv[2], 1.0);
                                    let t = matrix * v;
                                    //debug!("After: [{}, {}, {}, {}]", t.x,t.y,t.z,t.w);
                                    writeln!(output, "v {} {} {}", t.x, t.y, t.z)?;
                                    // End Matrix transformation
                                }
                                 */
                                /*
                                for vv in vn {
                                    writeln!(output, "vn {} {} {}", vv[0], vv[1], vv[2])?;
                                }
                                for vv in vt {
                                    writeln!(output, "vt {} {}", vv[0], vv[1])?;
                                }
                                 */
                                let mut v_buffer = Cursor::new(index_buffer);
                                for _i in 0..index_count / 3 {
                                    /*
                                    let a = v_buffer.read_u16::<LittleEndian>()?;
                                    let b = v_buffer.read_u16::<LittleEndian>()?;
                                    let c = v_buffer.read_u16::<LittleEndian>()?;
                                    */
                                    let a = v_buffer.read_u16::<LittleEndian>()? as u32 + idx;
                                    let b = v_buffer.read_u16::<LittleEndian>()? as u32 + idx;
                                    let c = v_buffer.read_u16::<LittleEndian>()? as u32 + idx;
                                    //writeln!(output, "f {}/{}/{}", a, b, c)?;
                                    // ??? Change DirectX clock-wise to counter clock-wise
                                    writeln!(output, "f {} {} {}", a, b, c)?;
                                }
                                idx += vertex_count as u32;
                                // End test
                            }
                        }
                        // Matrices
                        let matrices = object[8].array()?;
                        for matrix_array in matrices {
                            let matrix = matrix_array.array()?;
                            // 0 - "MATRIX"
                            // 1..n - Matrix
                            // TODO: Parse matrix later
                            trace!("\tMatrix: [Length: {}]", matrix.len() -1);
                            for m in matrix.iter().skip(1) {
                                let mm = m.data()?;
                                // 0 - Timeline, based on 160, in u32
                                // 1 ~ 16, 4x4 Matrix in f32, column-major order, for eg.
                                // [0.9542446, -0.2165474, -0.103003055, 0.0]
                                // [0.09967622, -0.026197463, 0.9785, 0.0]
                                // [-0.21809866, -0.9594297, -0.0034697813, 0.0]
                                // [3.17442, 16.080942, 53.538746, 1.0]
                                // =>
                                // |  0.9542446,    0.09967622,  -0.21809866,   3.17442   |
                                // | -0.2165474,   -0.026197463, -0.9594297,    16.080942 |
                                // | -0.103003055,  0.9785,      -0.0034697813, 53.538746 |
                                // |  0.0,          0.0,          0.0,          1.0       |
                                // 17 ~ 23, unknown data, for eg.
                                // [1.0, 1.0, 1.0, -0.0013206453, 0.00029969783, 0.00014250366, 0.002762136]
                                // Guessing: [1.0, 1.0, 1.0], scale factor ???
                                // Left another Vec4(-0.0013206453, 0.00029969783, 0.00014250366, 0.002762136), hmm...
                                let mut buffer = Cursor::new(mm);
                                /*
                                let mut f = Vec::new();
                                for i in 1..17 {
                                    //for i in 0..mm.len() / 4 {
                                    // if i == 0 {
                                    //     let a = buffer.read_u32::<littleendian>()?;
                                    //     //debug!("{}", a);
                                    // } else {
                                    let a = buffer.read_f32::<LittleEndian>()?;
                                    //debug!("{}", a);
                                    f.push(a);
                                    //}
                                }

                                 */
                                let _timeline = buffer.read_u32::<LittleEndian>()?;
                                //let mmm = Matrix4::new(
                                let mmm = gltf::scene::Transform::Matrix {
                                    matrix: [[
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                    ], [
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                    ], [
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                    ], [
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                        buffer.read_f32::<LittleEndian>()?,
                                    ]]
                                };
                                /*
                                let mmm: [[f32; 4]; 4] = mmm.matrix();
                                let o_t = cgmath::Vector3 { x: mmm[3][0], y: mmm[3][1], z: mmm[3][2] };
                                let mmm = gltf::scene::Transform::Matrix { matrix: mmm };
                                let o_scale = [buffer.read_f32::<LittleEndian>()?, buffer.read_f32::<LittleEndian>()?, buffer.read_f32::<LittleEndian>()?];
                                let o_rotation = [buffer.read_f32::<LittleEndian>()?, buffer.read_f32::<LittleEndian>()?, buffer.read_f32::<LittleEndian>()?, buffer.read_f32::<LittleEndian>()?];
                                let o_s = cgmath::Vector3 { x: o_scale[0], y: o_scale[1], z: o_scale[2] };
                                let mut o_r = cgmath::Quaternion {
                                    v: cgmath::Vector3 { x: o_rotation[0], y: o_rotation[1], z: o_rotation[2] },
                                    s: o_rotation[3],
                                };
                                o_r = o_r.normalize();
                                debug!("{:#?} - {:#?} - {:#?}", o_t,o_s,o_r);
                                let t = cgmath::Matrix4::from_translation(o_t);
                                let r = cgmath::Matrix4::from(o_r);
                                let s = cgmath::Matrix4::from_nonuniform_scale(o_s.x, o_s.y, o_s.z);
                                // //debug!("T: {:#?}", t);
                                //debug!("R: {:#?}", r);
                                //debug!("S: {:#?}", r);
                                let x = t * r * s;

                                debug!("Calculated Matrix: {:#?}", x);
                                 */
                                //debug!("My calc: {:#?}", x);
                                //debug!("orig_scale: {:#?}", o_scale);
                                //debug!("orig_rotation: {:#?}", o_rotation);
                                debug!("Matrix: {:#?}", mmm);
                                let m4 = mmm.matrix();
                                let m3 = Matrix3 {
                                    x: Vector3 { x: m4[0][0], y: m4[0][1], z: m4[0][2] },
                                    y: Vector3 { x: m4[1][0], y: m4[1][1], z: m4[1][2] },
                                    z: Vector3 { x: m4[2][0], y: m4[2][1], z: m4[2][2] },
                                };
                                //debug!("M3: {:#?}", m3);
                                let sx = m3.x.magnitude();
                                let sy = m3.y.magnitude();
                                let sz = m3.z.magnitude() * m3.determinant().signum();
                                let t_s = [sx, sy, sz];
                                debug!("T_S: {:#?}", t_s);
                                let nx = m3.x * 1.0 / sx;
                                let ny = m3.y * 1.0 / sy;
                                let nz = m3.z * 1.0 / sz;
                                let nr = Matrix3 { x: nx, y: ny, z: nz };
                                //debug!("N_R: {:#?}", nr);
                                let mut t_r = Quaternion::from(nr);
                                debug!("T_R: {:#?}", t_r);
                                let t_t = Vector3 { x: m4[3][0], y: m4[3][1], z: m4[3][2] };
                                debug!("T_T: {:#?}", t_t);
                                let t = Matrix4::from_translation(t_t);
                                let r = Matrix4::from(t_r);
                                let s = Matrix4::from_nonuniform_scale(sx, sy, sz);
                                let x = t * r * s;
                                debug!("My Calc: {:#?}", x);
                                let mmm = gltf::scene::Transform::Matrix { matrix: m4 };
                                let (translation, rotation, scale) = mmm.decomposed();
                                debug!("t: {:#?}", translation);
                                debug!("r: {:#?}", rotation);
                                debug!("s: {:#?}", scale);
                                let decomposed = gltf::scene::Transform::Decomposed { translation, rotation, scale };
                                let matrix = decomposed.matrix();
                                debug!("new matrix: {:#?}", matrix);
                                debug!("origin matrix: {:#?}", m4);


                                // TODO: Update logic here, processing only one matrix right now
                                break;
                            }
                        }
                        // SFX Blocks?
                        if object.len() > 9 {
                            // TODO: Parse SFX
                            let sfx = object[9].array()?;
                            if !sfx.is_empty() {
                                warn!("\tSFX: Unhandled SFX blocks? {}@{}", file!(), line!());
                            }
                        }
                    }
                }
                _ => {
                    error!("Unknown block: {}", name);
                }
            }
        }

        //debug!("{:#?}", self.data);

        // Test obj code

        std::fs::write("test.obj", output)?;
        debug!("{:#?}", self.vertices.len());


        Ok(())
    }

    /// Export OBJ file
    pub fn export_obj(&self) -> Result<()> {
        Ok(())
    }

    /// Check whether the file is a valid BNX / PNX format
    fn check_bwx_header(&mut self) -> Result<()> {
        let header = &self.content.get_ref()[..4];
        if header != "BWXF".as_bytes() {
            return Err("Invalid BWX file.".into());
        }

        Ok(())
    }

    /// Read BNX / PNX special packed integer value (little endian)
    fn read_i32_packed(&mut self) -> Result<i32> {
        let mut result: u32 = 0;
        let mut shift = 0;

        while shift < 35 {
            let t = self.content.read_u8()? as u32;
            result |= (t & 0x7f) << shift;

            if t & 0x80 == 0 {
                break;
            } else {
                shift += 7;
            }
        }

        Ok(result as i32)
    }

    /// Read block size & numbers
    fn read_block_size_number(&mut self) -> Result<(i32, i32)> {
        Ok((self.read_i32_packed()?, self.read_i32_packed()?))
    }

    /// Read string
    fn read_string(&mut self) -> Result<String> {
        //let length = self.content.read_u8()?;
        // Bug fix, found length 0x80 in "OBO020_DEFAULT.PNX", use packed int
        let length = self.read_i32_packed()?;
        let mut buffer = Vec::new();
        buffer.resize(length as usize, 0);
        self.content.read_exact(&mut buffer)?;

        let (cow, _encoding, had_errors) = encoding_rs::EUC_KR.decode(&buffer);
        if had_errors {
            error!("Failed to convert string from Korean to UTF-8!");
            Ok(String::from_utf8_lossy(&buffer).trim_matches('\0').to_string())
        } else {
            Ok(cow.trim_matches('\0').to_string())
        }
    }

    /// Go through the whole BWX file
    #[tracing::instrument(skip(self, root))]
    fn go_through(&mut self, root: bool) -> Result<SlType> {
        // Parse root as Signature D
        let signature = if root { 0x44 } else { self.content.read_u8()? };

        let data = match signature {
            0x41 => { // Signature A
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature A] - Size: {}, Blocks: {}", size, blocks);
                let mut node = vec![];
                while blocks > 0 {
                    let child = self.go_through(false)?;
                    node.push(child);
                    blocks -= 1;
                }
                SlType::Array(node)
            }
            0x42 => { // Signature B
                let size = self.read_i32_packed()?;
                trace!("[Signature B] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size as usize, 0);
                self.content.read_exact(&mut buffer)?;
                SlType::Data(buffer)
            }
            0x43 => { // Signature C
                let value = -self.content.read_i8()?;
                trace!("[Signature C] - Value: {}", value);
                SlType::Char(value)
            }
            0x44 => { // Signature D
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature D] - Size: {}, Blocks: {}", size, blocks);
                let mut node = vec![];
                while blocks > 0 {
                    let name = self.read_string()?;
                    trace!("[Signature D] - Name: {}", name);
                    let child = self.go_through(false)?;
                    match child {
                        SlType::Array(children) => {
                            node.push(SlType::DArray(name, children));
                        }
                        _ => {
                            node.push(SlType::DArray(name, vec![child]));
                        }
                    }
                    blocks -= 1;
                }
                SlType::Array(node)
            }
            0x46 => { // Signature F
                let value = self.content.read_f32::<LittleEndian>()?;
                trace!("[Signature F] - Value: {:.3}", value);
                SlType::Float(value)
            }
            0x48 => { // Signature H
                let value = -self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature H] - Value: {}", value);
                SlType::Word(value)
            }
            0x49 => { // Signature I
                let value = self.content.read_i32::<LittleEndian>()?;
                trace!("[Signature I] - Value: {}", value);
                SlType::Int(value)
            }
            0x53 => { // Signature S
                let value = self.read_string()?;
                trace!("[Signature S] - Value: {}", value);
                SlType::String(value)
            }
            0x57 => { // Signature W
                let value = self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature W] - Value: {}", value);
                SlType::Word(value)
            }
            0x59 => { // Signature Y
                let value = self.content.read_u8()?;
                trace!("[Signature Y] - Value: {}", value);
                SlType::UChar(value)
            }
            s if s < 0x20 => {
                // Independent data
                trace!("[Independent Data] - Value: {}", s);
                SlType::UChar(s)
            }
            s if s >= 0x80 => {
                // Independent data block
                let size = s as usize & 0x7f;
                trace!("[Independent Data Block] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size, 0);
                self.content.read_exact(&mut buffer)?;
                SlType::Data(buffer)
            }
            _ => {
                error!("Unhandled signature = 0x{:02x}, position: {}", signature, self.content.position());
                //debug!("{:#?}", self.node);
                panic!("Unhandled type {}", signature);
            }
        };
        Ok(data)
    }
}

fn align_to_multiple_of_four(n: &mut u32) {
    *n = (*n + 3) & !3;
}

fn to_padded_byte_vector<T>(vec: Vec<T>) -> Vec<u8> {
    let byte_length = vec.len() * mem::size_of::<T>();
    let byte_capacity = vec.capacity() * mem::size_of::<T>();
    let alloc = vec.into_boxed_slice();
    let ptr = Box::<[T]>::into_raw(alloc) as *mut u8;
    let mut new_vec = unsafe { Vec::from_raw_parts(ptr, byte_length, byte_capacity) };
    while new_vec.len() % 4 != 0 {
        new_vec.push(0); // pad to multiple of four bytes
    }
    new_vec
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_bwx_header() {
        let mut bwx = BWX::new();
        bwx.content = Cursor::new(vec![66, 87, 88, 70]);
        assert!(bwx.check_bwx_header().is_ok(), "File header check should pass");
        bwx.content = Cursor::new(vec![11, 22, 33, 44]);
        assert!(bwx.check_bwx_header().is_err(), "File header check should fail");
    }

    #[test]
    fn read_i32_packed() {
        let mut bwx = BWX::new();
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.content = Cursor::new(vec![0xc1, 0xef, 0x5a, 0x0c]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 1488833, "Packed integer value incorrect");
        bwx.content = Cursor::new(vec![0x0c, 0x02]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 12, "Packed integer value incorrect");
    }

    #[test]
    fn read_string() {
        let mut bwx = BWX::new();
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.content = Cursor::new(vec![0x02, 0x30, 0x00, 0x53]);
        assert_eq!(bwx.read_string().unwrap().as_str(), "0", "The string should be '0'");
    }
}

