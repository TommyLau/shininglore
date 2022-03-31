use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::{mem, fs};
use std::borrow::Borrow;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use tracing::{debug, error, info, trace, warn};
use serde::Serialize;
use cgmath::*;
use gltf::{Gltf, json::{self, validation::Checked::Valid}, Node, scene::Transform::Matrix};
use gltf::json::Asset;
use gltf::texture::{MinFilter, MagFilter, WrappingMode};
use image::GenericImageView;
use serde_json::json;

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
    buffer: Vec<u8>,
    buffer_views: Vec<json::buffer::View>,
    accessors: Vec<json::Accessor>,
    meshes: Vec<json::Mesh>,
    nodes: Vec<json::Node>,
    images: Vec<json::Image>,
    textures: Vec<json::Texture>,
    materials: Vec<json::Material>,
    // Store the material group information
    material_index: Vec<Vec<u32>>,
    node_index: Vec<u32>,
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
        let oname = filename.as_ref().to_owned();

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
                    for material_groups in children {
                        let material_group = material_groups.array()?;
                        // material[0] - Material Group "MTRL"
                        // material[1] - Material Group Name
                        // material[2..n] - Material Array for Sub Materials
                        let name = material_group[1].string()?;
                        trace!("Material Group: {}", name);
                        let mut material_array = vec![];
                        for (i, sub_materials) in material_group.iter().enumerate().skip(2) {
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
                            let (filename, tga) = if sub_material.len() > 8 {
                                // Some materials have no texture, such as glass
                                let texture = sub_material[8].array()?;
                                // 0 - "TEX"
                                // 1 - Most 0x00, timer?
                                // 2 - Filename
                                let filename = texture[2].string()?
                                    .split('\\').last().unwrap().to_owned();
                                (
                                    filename.to_lowercase()
                                        .split('.').next().unwrap()
                                        .to_owned() + ".png",
                                    if filename[filename.len() - 3..].to_lowercase().starts_with("dds") {
                                        filename.to_owned() + ".png"
                                    } else {
                                        filename.to_owned()
                                    }
                                )
                            } else { ("".into(), "".into()) };


                            debug!("TGA File: {}", tga);
                            let a = self.images.iter()
                                .position(|x| x.uri.as_ref() == Some(&filename));
                            let texture_index = if a.is_some() {
                                Some(a.unwrap() as u32)
                            } else if tga.len() > 0 {
                                {
                                    // Convert image from TGA to PNG
                                    let img = image::open(tga.clone())?;

                                    // The dimensions and color
                                    debug!("\tdimensions: {:?}, color: {:?}", img.dimensions(), img.color());
                                    // Write the contents of this image to the Writer in PNG format.
                                    if !Path::new(&filename).exists() {
                                        img.save(filename.clone())?;
                                    } else {
                                        debug!("\tImage file {} exists, no convert", filename);
                                    }
                                }
                                let texture_index = self.textures.len() as u32;
                                let image_index = self.images.len() as u32;
                                let image = json::Image {
                                    buffer_view: None,
                                    mime_type: None,
                                    name: None,
                                    uri: Some(filename.clone()),
                                    extensions: None,
                                    extras: Default::default(),
                                };
                                self.images.push(image);
                                let texture = json::Texture {
                                    name: None,
                                    //sampler: Some(json::Index::new(0)),
                                    sampler: None,
                                    source: json::Index::new(image_index),
                                    extensions: None,
                                    extras: Default::default(),
                                };
                                self.textures.push(texture);
                                Some(texture_index)
                            } else {
                                None
                            };


                            // Store the material id in array for mesh to lookup
                            let material_id = self.materials.len() as u32;
                            material_array.push(material_id);

                            let material = json::Material {
                                alpha_cutoff: None,
                                alpha_mode: Default::default(),
                                // FIXME: Enable double side material for disordered meshes !!!
                                double_sided: true,
                                name: Some(name.clone()),
                                pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                                    base_color_factor: Default::default(),
                                    base_color_texture: if let Some(i) = texture_index {
                                        Some(json::texture::Info {
                                            index: json::Index::new(i),
                                            tex_coord: 0, // Only have texture_0 now
                                            extensions: None,
                                            extras: Default::default(),
                                        })
                                    } else { None },
                                    metallic_factor: Default::default(),
                                    roughness_factor: Default::default(),
                                    metallic_roughness_texture: None,
                                    extensions: None,
                                    extras: Default::default(),
                                },
                                normal_texture: None,
                                occlusion_texture: None,
                                emissive_texture: None,
                                emissive_factor: Default::default(),
                                extensions: None,
                                extras: Default::default(),
                            };
                            self.materials.push(material);

                            trace!("\tSub Material {}: Highlight: {}, File: {}, Orig: {}",
                                i - 2, highlight, filename, tga);
                        }
                        // Store current material group in material index
                        self.material_index.push(material_array);
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
                        trace!("Object: {}, Texture Index: {}", name, texture_index);

                        {
                            // Do not process special object starts with EV_ / EP_
                            if name.starts_with("EV_") || name.starts_with("EP_") {
                                continue;
                            }
                        }

                        // Confirmed: MSHX = clockwise?, MNHX = counter-clockwise?
                        let mut direction = vec![];
                        direction.write_i32::<byteorder::BigEndian>(object[6].int()?)?;
                        let direction = std::str::from_utf8(&direction).unwrap();
                        // After checking, no matter MSHX nor MNHX, we have to change the index order
                        // DirectX is rendering in clockwise, OpenGL is rendering in couter-clockwise
                        // So we have to change the order from (a, b, c) -> (a, c, b)
                        debug!("Direction: {}", direction);
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
                        let mut node_index = vec![];
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
                            let index_buffer_length = index_buffer.len();
                            if index_buffer_length != index_count as usize * 2 {
                                error!("Index block size incorrect!");
                            }
                            trace!("\tMesh: [Texture Index: {}, Index Count: {}, Size: {}",
                                sub_texture_index, index_count, index_buffer.len());
                            debug!("Before Padding Length: {}", index_buffer.len());
                            let mut index_buffer = Cursor::new(index_buffer.clone());
                            // Padding to four bytes
                            let padding = ((index_buffer_length + 3) & !3) - index_buffer_length;
                            if padding > 0 {
                                debug!("============Padding: {}", padding);
                                index_buffer.seek(SeekFrom::End(0))?;
                                for _ in 0..padding {
                                    index_buffer.write_u8(0)?;
                                }
                            }
                            let index_buffer = index_buffer.into_inner();
                            debug!("After Padding Length: {}", index_buffer.len());

                            let blocks = mesh[2].array()?;
                            // for vertices in blocks {
                            // FIXME: Only process the first frame of the animation
                            let vertices = &blocks[0];
                            {
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
                                // 111
                                let mut buffer_view_index = self.buffer_views.len();
                                let accessor_index = self.accessors.len() as u32;
                                let mesh_index = self.meshes.len() as u32;

                                // Store the children
                                node_index.push(self.nodes.len() as u32);
                                let node = json::Node {
                                    camera: None,
                                    children: None,
                                    extensions: Default::default(),
                                    extras: Default::default(),
                                    matrix: None,
                                    mesh: Some(json::Index::new(mesh_index)),
                                    name: None,
                                    rotation: None,
                                    scale: None,
                                    translation: None,
                                    skin: None,
                                    weights: None,
                                };
                                self.nodes.push(node);

                                // Mesh - Perimitive
                                let primitive = json::mesh::Primitive {
                                    attributes: {
                                        let mut map = std::collections::HashMap::new();
                                        map.insert(Valid(json::mesh::Semantic::Positions), json::Index::new(accessor_index));
                                        // FIXME: Something wrong with the normals, might be DirectX clock-wise normal calculation?
                                        // Don't know how it will be to render in WebGPU, will see later, keep it as ON for now.
                                        map.insert(Valid(json::mesh::Semantic::Normals), json::Index::new(accessor_index + 1));
                                        map.insert(Valid(json::mesh::Semantic::TexCoords(0)), json::Index::new(accessor_index + 2));
                                        map
                                    },
                                    extensions: Default::default(),
                                    extras: Default::default(),
                                    indices: Some(json::Index::new(accessor_index + 3)),
                                    material: if texture_index < 0 { None } else {
                                        Some(json::Index::new(
                                            self.material_index[texture_index as usize][sub_texture_index as usize]))
                                    },
                                    mode: Valid(json::mesh::Mode::Triangles),
                                    targets: None,
                                };

                                let mesh = json::Mesh {
                                    extensions: Default::default(),
                                    extras: Default::default(),
                                    // name: Some(name.clone().into()),
                                    // As a mesh group, no name is giving to the mesh but the object
                                    name: None,
                                    primitives: vec![primitive],
                                    weights: None,
                                };
                                self.meshes.push(mesh);

                                // Vertex
                                let mut v_min = cgmath::Vector3::new(0.0f32, 0.0, 0.0);
                                let mut v_max = cgmath::Vector3::new(0.0f32, 0.0, 0.0);
                                let mut v_set = false;
                                {
                                    // TODO: Clean up code
                                    // Calculate min / max for position
                                    let mut vb = Cursor::new(vertex_buffer.clone());
                                    for i in 0..vertex_count / 3 {
                                        let v = cgmath::Vector3::new(
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                        );
                                        let n = (
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                        );
                                        let uv = (
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                        );
                                        if v_set {
                                            //debug!("Min Max: {:?} - {:?} - {:?}", v, v_min, v_max);
                                            if v.x > v_max.x { v_max.x = v.x; }
                                            if v.y > v_max.y { v_max.y = v.y; }
                                            if v.z > v_max.z { v_max.z = v.z; }
                                            if v.x < v_min.x { v_min.x = v.x; }
                                            if v.y < v_min.y { v_min.y = v.y; }
                                            if v.z < v_min.z { v_min.z = v.z; }
                                        } else {
                                            v_min = v;
                                            v_max = v;
                                            v_set = true;
                                        }
                                    }
                                    debug!("Min = {:?}, Max = {:?}", v_min, v_max);
                                }
                                let v_min: [f32; 3] = v_min.into();
                                let v_max: [f32; 3] = v_max.into();
                                let mut accessor = json::Accessor {
                                    buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                                    byte_offset: 0,
                                    count: vertex_count as u32,
                                    component_type: Valid(json::accessor::GenericComponentType(
                                        json::accessor::ComponentType::F32
                                    )),
                                    extensions: None,
                                    extras: Default::default(),
                                    type_: Valid(json::accessor::Type::Vec3),
                                    min: Some(json!(v_min)),
                                    max: Some(json!(v_max)),
                                    name: None,
                                    normalized: false,
                                    sparse: None,
                                };
                                self.accessors.push(accessor.clone());
                                // Normal
                                accessor.byte_offset = (3 * mem::size_of::<f32>()) as u32;
                                accessor.min = None;
                                accessor.max = None;
                                self.accessors.push(accessor.clone());
                                // Texture Coordinate
                                accessor.byte_offset = (6 * mem::size_of::<f32>()) as u32;
                                accessor.type_ = Valid(json::accessor::Type::Vec2);
                                self.accessors.push(accessor.clone());

                                let mut buffer_view = json::buffer::View {
                                    buffer: json::Index::new(0),
                                    byte_length: vertex_buffer.len() as u32,
                                    byte_offset: Some(self.buffer.len() as u32),
                                    byte_stride: Some(vertex_size as u32),
                                    name: None,
                                    target: None,
                                    extensions: None,
                                    extras: Default::default(),
                                };
                                self.buffer_views.push(buffer_view.clone());

                                // Index Buffer ------------------
                                buffer_view_index = self.buffer_views.len();
                                // Accessor for index
                                let accessor = json::Accessor {
                                    buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                                    byte_offset: 0,
                                    count: index_count as u32,
                                    component_type: Valid(json::accessor::GenericComponentType(
                                        json::accessor::ComponentType::U16
                                    )),
                                    extensions: None,
                                    extras: Default::default(),
                                    type_: Valid(json::accessor::Type::Scalar),
                                    min: None,
                                    max: None,
                                    name: None,
                                    normalized: false,
                                    sparse: None,
                                };
                                self.accessors.push(accessor);
                                // Index bufferView
                                buffer_view_index = self.buffer_views.len();
                                self.buffer.append(&mut vertex_buffer.clone());
                                // TODO: UV's V is negative value, change to positive and horizontal flip image?
                                {
                                    // MEMO: Guessing, the extra two vertex are MIN / MAX
                                    // Partially confirm with EV_COL3D object, but the coordinate value is incorrect
                                    // So, just leave it alone as what it is?
                                    // Test code to output UV texture coordinate
                                    let mut vb = Cursor::new(vertex_buffer.clone());
                                    for i in vertex_count / 3..vertex_count / 3 + 2 {
                                        // Skip Vertex (96) + Normal (96) = 192 = 128 + 64
                                        let v = (
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                        );
                                        let n = (
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                        );
                                        let uv = (
                                            vb.read_f32::<LittleEndian>()?,
                                            vb.read_f32::<LittleEndian>()?,
                                        );
                                        // debug!("---- Vertex: {:?}", v);
                                        // debug!("---- Normal: {:?}", n);
                                        // debug!("--- UV: {:?}",uv);
                                    }
                                    // End test output text UV
                                }
                                buffer_view.buffer = json::Index::new(0);
                                buffer_view.byte_length = index_buffer.len() as u32;
                                buffer_view.byte_offset = Some(self.buffer.len() as u32);
                                buffer_view.byte_stride = None;
                                self.buffer_views.push(buffer_view);

                                // Seems double sided material could solve the problem
                                // And for HERO PNX, no matter how I change the index order
                                // The mesh data with normals are incorrect, comment out the following code
                                // and use only "DOUBLE SIDED" material? MAYBE...
                                // TODO: Comment out the code or not?!
                                // if direction.starts_with("MSHX") {
                                //     // "MSHX", DirectX, left hand clockwise triangles
                                //     // Have to be changed to right hand counter-clockwise for OpenGL
                                //     // Change (a, b, c) -> <a, c, b>
                                //     debug!("------ changing order ------");
                                //     let mut i_buffer = Cursor::new(index_buffer);
                                //     let mut o_buffer = Cursor::new(vec![]);
                                //     for _i in 0..index_count / 3 {
                                //         let a = i_buffer.read_u16::<LittleEndian>()?;
                                //         let b = i_buffer.read_u16::<LittleEndian>()?;
                                //         let c = i_buffer.read_u16::<LittleEndian>()?;
                                //         o_buffer.write_u16::<LittleEndian>(a)?;
                                //         o_buffer.write_u16::<LittleEndian>(c)?;
                                //         o_buffer.write_u16::<LittleEndian>(b)?;
                                //     }
                                //     let mut buffer = o_buffer.into_inner();
                                //     self.buffer.append(&mut buffer);
                                //     // End order changing
                                // } else {
                                // Original order, store to binary directly
                                self.buffer.append(&mut index_buffer.clone());
                                // }

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
                        let node_matrix = [
                            matrix.x.x, matrix.x.y, matrix.x.z, matrix.x.w,
                            matrix.y.x, matrix.y.y, matrix.y.z, matrix.y.w,
                            matrix.z.x, matrix.z.y, matrix.z.z, matrix.z.w,
                            matrix.w.x, matrix.w.y, matrix.w.z, matrix.w.w,
                        ];

                        // Store the node for Scene
                        let node_count = self.nodes.len() as u32;
                        debug!("mesh_node: {:#?}", node_count);
                        debug!("node_index: {:?}", node_index);
                        self.node_index.push(node_count);
                        let node = json::Node {
                            camera: None,
                            children: Some(node_index.into_iter().map(|x| json::Index::new(x)).collect()),
                            extensions: Default::default(),
                            extras: Default::default(),
                            matrix: Some(node_matrix),
                            mesh: None,
                            name: Some(name.clone().into()),
                            rotation: None,
                            scale: None,
                            translation: None,
                            skin: None,
                            weights: None,
                        };
                        self.nodes.push(node);

                        // Matrices
                        let matrices = object[8].array()?;
                        for matrix_array in matrices {
                            let matrix = matrix_array.array()?;
                            // 0 - "MATRIX"
                            // 1..n - Matrix
                            // TODO: Parse matrix later
                            trace!("\tMatrix: [Length: {}]", matrix.len() - 1);
                            let mut o_buffer = Cursor::new(vec![]);
                            let mut timeline_max = 0.0;
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
                                //debug!("-------matrix len: {}", mm.len());
                                let mut buffer = Cursor::new(mm);
                                let timeline = buffer.read_u32::<LittleEndian>()? as f32 / 3600.0;
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
                                let (translation, rotation, scale) = mmm.decomposed();
                                debug!("T: {:.2}, {:?}, {:?}, {:?}", timeline, translation, rotation, scale);
                                // Write timeline, translation, rotation and scale to buffer
                                // Could use system's array.as_bytes, but cannot ensure when running on big endian system
                                // So use the old school byteorder method
                                o_buffer.write_f32::<LittleEndian>(timeline)?;
                                for v in translation { o_buffer.write_f32::<LittleEndian>(v)?; }
                                for v in rotation { o_buffer.write_f32::<LittleEndian>(v)?; }
                                for v in scale { o_buffer.write_f32::<LittleEndian>(v)?; }

                                if timeline > timeline_max {
                                    timeline_max = timeline;
                                }
                                /*
                                // TODO: Update logic here, processing only one matrix right now
                                break;

                                 */
                            }
                            // Store data in buffer
                            let offset = self.buffer.len();
                            let length = o_buffer.get_ref().len();
                            self.buffer.append(o_buffer.get_mut());

                            // Prepare bufferView
                            let buffer_view_index = self.buffer_views.len();
                            let buffer_view = json::buffer::View {
                                buffer: json::Index::new(0),
                                byte_length: length as u32,
                                byte_offset: Some(offset as u32),
                                byte_stride: Some((mem::size_of::<f32>() * 11) as u32),
                                name: Some(name.clone() + "_Matrix"),
                                target: None,
                                extensions: None,
                                extras: Default::default(),
                            };
                            self.buffer_views.push(buffer_view);

                            // Accessor
                            let accessor_index = self.accessors.len();
                            let accessor = json::Accessor {
                                buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                                byte_offset: 0,
                                count: matrix.len() as u32 - 1,
                                component_type: Valid(json::accessor::GenericComponentType(
                                    json::accessor::ComponentType::F32)),
                                extensions: None,
                                extras: Default::default(),
                                type_: Valid(json::accessor::Type::Scalar),
                                min: Some(json!([0.0f32])),
                                max: Some(json!([timeline_max])),
                                name: Some(name.clone() + "_Timeline"),
                                normalized: false,
                                sparse: None,
                            };
                            self.accessors.push(accessor);
                            debug!("bufferView: {}", buffer_view_index);

                            // TODO: Add accessors for Translation / Rotation / Scale
                            // UPDATE ABOVE! 2022-03-30
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

        // let oname = String::from(filename.as_ref().to_str().unwrap());
        let oname = oname.to_str().unwrap().to_lowercase();
        let oname = oname[..oname.rfind('.').unwrap()].to_owned();
        debug!("{}", oname);
        std::fs::write(oname.clone() + ".obj", output)?;


        let buffer = json::Buffer {
            byte_length: self.buffer.len() as u32,
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
            uri: Some(oname.clone() + ".bin".into()),
        };

        let asset = json::Asset {
            copyright: Some("SLODT All Rights Reserved. (C) 2022".into()),
            extensions: None,
            extras: Default::default(),
            generator: Some("Tommy's BWX Exporter".into()),
            min_version: None,
            version: "2.0".to_string(),
        };

        //let a: Vec<json::Index<Node>> = (0..10u32).map(|x| json::Index::new(x)).colletc();
        // let scene_nodes: Vec<json::Index<json::Node>> = (0..self.nodes.len() as u32)
        //     .map(|x| json::Index::new(x)).collect();
        let scene_nodes = self.node_index.iter()
            .map(|x| json::Index::new(*x)).collect();

        // Disable sampler should display correct texture
        /*
        let sampler = json::texture::Sampler {
            mag_filter: Some(Valid(MagFilter::Nearest)),
            min_filter: Some(Valid(MinFilter::Nearest)),
            name: None,
            wrap_s: Valid(WrappingMode::ClampToEdge),
            wrap_t: Valid(WrappingMode::ClampToEdge),
            extensions: None,
            extras: Default::default(),
        };

         */

        // NOTICE: Texture should be vertical flipped
        let root = json::Root {
            asset,
            scene: Some(json::Index::new(0)),
            scenes: vec![json::Scene {
                extensions: Default::default(),
                extras: Default::default(),
                name: Some("Scene".into()),
                nodes: scene_nodes,
            }],
            nodes: self.nodes.clone(),
            meshes: self.meshes.clone(),
            accessors: self.accessors.clone(),
            buffer_views: self.buffer_views.clone(),
            buffers: vec![buffer],
            //samplers: vec![sampler],
            materials: self.materials.clone(),
            textures: self.textures.clone(),
            images: self.images.clone(),
            ..Default::default()
        };


        //debug!("Buffer Views:\n{:#?}", self.buffer_views);
        //debug!("Accessors:\n{:#?}", self.accessors);
        let j = json::serialize::to_string_pretty(&root).expect("OK");
        //debug!("glTF:\n{}", j);

        std::fs::write(oname.clone() + ".gltf", j.as_bytes());
        std::fs::write(oname + ".bin", self.buffer.clone());

        // debug!("{:#?}", self.material_index);


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

