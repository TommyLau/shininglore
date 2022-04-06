use std::io::{Cursor, Read};
use std::path::Path;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use tracing::{debug, error, info, trace, warn};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
#[derive(Debug, Clone)]
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
#[derive(Debug, Default)]
pub struct Head {
    pub name: String,
    pub description: String,
    pub magic: i32,
    pub version: i32,
    pub other: String,
}

// Block 'MTRL' and 'SUBMTRL'
#[derive(Debug, Default)]
pub struct SubMaterial {
    pub diffuse: i32,
    pub ambient: i32,
    pub specular: i32,
    pub highlight: f32,
    pub filename: Option<String>,
}

#[derive(Debug, Default)]
pub struct Material {
    pub sub_materials: Vec<SubMaterial>,
}

#[derive(Debug, Default)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coord: [f32; 2],
}

// Block 'DXMESHF'
#[derive(Debug, Default)]
pub struct SubMesh {
    timeline: i32,
    vertices: Vec<Vertex>,
}

// Block 'DXMESH'
#[derive(Debug, Default)]
pub struct Mesh {
    sub_material: i32,
    sub_meshes: Vec<SubMesh>,
    index_count: i32,
    indices: Vec<u16>,
}

#[derive(Debug, Default)]
pub struct Matrix {
    timeline: i32,
    matrix: [f32; 16],
}

#[derive(Debug, Default)]
pub struct Object {
    name: String,
    meshes: Vec<Mesh>,
    matrices: Vec<Matrix>,
}

#[derive(Debug, Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    content: Cursor<Vec<u8>>,
    pub data: SlType,
    // HEAD
    pub head: Head,
    // MTRL
    pub materials: Vec<Material>,
    // DXOBJ / SPOB
    pub objects: Vec<Object>,
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
        info!("{}", filename.as_ref().display());
        let oname = filename.as_ref().to_owned();

        let data = std::fs::read(filename.as_ref())?;
        self.content = Cursor::new(data);

        self.check_bwx_header()?;
        self.content.set_position(4);
        self.data = self.go_through(true)?;

        for node in self.data.array().unwrap() {
            let node_name = node.string()?;
            let children = node.d_array()?;
            match node_name.as_str() {
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
                        self.head = Head {
                            name: children[0].string()?,
                            description: children[1].string()?,
                            magic: children[2].int()?,
                            version: children[3].int()?,
                            other: children[4].string()?,
                        };
                        if !self.head.name.starts_with("HEAD") {
                            error!("Incorrect HEAD block");
                        }
                        if self.head.magic != 0x504e5800 {
                            error!("Header magic != PNX");
                        }
                        match self.head.version {
                            0x500 => trace!("ShiningLore V1 PNX"),
                            0x602 => trace!("ShiningLore V2 PNX"),
                            _ => error!("Unknown ShiningLore PNX version!!"),
                        }
                    } else {
                        warn!("HEAD block length < 4, no PNX version available!");
                    }
                }
                "MTRL" => {
                    for material_children in children {
                        let material_array = material_children.array()?;
                        // material[0] - Material Group "MTRL"
                        // material[1] - Material Group Name
                        // material[2..n] - Material Array for Sub Materials
                        let material_name = material_array[1].string()?;
                        trace!("Material Group: {}", material_name);
                        let mut sub_materials = vec![];
                        for sub_material_children in material_array.iter().skip(2) {
                            let sub_material_array = sub_material_children.array()?;
                            // 0 - "SUBMTRL"
                            // 1 - Diffuse ???
                            // 2 - Ambient ???
                            // 3 - Specular ???
                            // 4 - Some float ???
                            // 5 - High light
                            // 6 - Most 0x01
                            // 7 - ???
                            // 8 - Texture Array
                            let sub_material = SubMaterial {
                                diffuse: sub_material_array[1].int()?,
                                ambient: sub_material_array[2].int()?,
                                specular: sub_material_array[3].int()?,
                                highlight: sub_material_array[5].float()?,
                                filename: if sub_material_array.len() > 8 {
                                    // Some materials have no texture, such as glass
                                    let texture_array = sub_material_array[8].array()?;
                                    // 0 - "TEX"
                                    // 1 - Most 0x00, timer?
                                    // 2 - Filename
                                    Some(texture_array[2].string()?
                                        .split('\\').last().unwrap()
                                        .split('.').next().unwrap()
                                        .to_owned() + ".png")
                                } else {
                                    None
                                },
                            };
                            trace!("\tSub Material - Texture: {:?}", sub_material.filename);
                            sub_materials.push(sub_material);
                        }
                        self.materials.push(Material { sub_materials });
                    }
                }
                // TODO: Parse OBJ2 mesh data from SL1
                "OBJ2" => {
                    if children.is_empty() {
                        warn!("No data block found in {}", node_name);
                        continue;
                    }

                    warn!("OBJ2 parsing needs to be implemented! {}@{}", file!(), line!());
                }
                "OBJECT" => {}
                "CAM" => {}
                "LIGHT" => {}
                "SOUND" => {}
                "BONE" => {}
                "CHART" => {}
                "DXOBJ" | "SPOB" => {
                    if children.is_empty() {
                        warn!("No data block found in {}", node_name);
                        continue;
                    }

                    for object_children in children {
                        let object_array = object_children.array()?;
                        // 0 - "DXOBJ" / "SPOB"
                        // 1 - Mesh Name
                        // 2 - Unknown integer
                        // 3 - Texture Group Index : -1 means no texture
                        // 4, 5 - Unknown
                        // 6 - 0x4D534858h("MSHX") or 0x4D4E4858h("MNHX")
                        // 7 - Array("DXMESH")
                        // 8 - Array("MATRIX")
                        let object_name = object_array[1].string()?;
                        let material = object_array[3].int()?;
                        let mut direction = vec![];
                        direction.write_i32::<BigEndian>(object_array[6].int()?)?;
                        let direction = std::str::from_utf8(&direction).unwrap();
                        trace!("Object: {}, Material: {}, Direction: {}", object_name, material, direction);

                        {
                            // Do not process special object starts with EV_ / EP_
                            // FIXME: Enable later when process with collision detection and etc.
                            if object_name.starts_with("EV_") || object_name.starts_with("EP_") {
                                continue;
                            }
                        }

                        // Meshes - DXMESH
                        let mesh_children = object_array[7].array()?;
                        let mut meshes = vec![];
                        for meshes_array in mesh_children {
                            let mesh_array = meshes_array.array()?;
                            // 0 - "DXMESH"
                            // 1 - Sub Material in Materials
                            // 2 - Array("DXMESHF")
                            // 3 - Index Count
                            // 4 - Index Buffer
                            let sub_material = mesh_array[1].int()?;
                            trace!("\tMesh - Sub_Material: {}", sub_material);

                            let sub_mesh_children = mesh_array[2].array()?;
                            let mut sub_meshes = vec![];
                            for sub_mesh_array in sub_mesh_children {
                                let sub_mesh = sub_mesh_array.array()?;
                                // 0 - "DXMESHF"
                                // 1 - VB Timer
                                // 2 - Vertex Type??? - 0x15
                                // 3 - Vertex Count
                                // 4 - Vertex Size - 0x20
                                // 5 - Vertex Buffer
                                let timeline = sub_mesh[1].int()?;
                                let _vertex_type = sub_mesh[2].int()?;
                                let vertex_count = sub_mesh[3].int()?;
                                let _vertex_size = sub_mesh[4].int()?;
                                let vertex_buffer = sub_mesh[5].data()?.clone();
                                trace!("\t\tSub_Mesh - Timeline: {}, Count: {}", timeline, vertex_count );
                                let mut vertex_buffer = Cursor::new(vertex_buffer);

                                // Vertex
                                let mut vertices = vec![];
                                for _ in 0..vertex_count {
                                    let position = [
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                    ];
                                    let normal = [
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                    ];
                                    let tex_coord = [
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                        vertex_buffer.read_f32::<LittleEndian>()?,
                                    ];
                                    vertices.push(Vertex { position, normal, tex_coord });
                                }
                                sub_meshes.push(SubMesh { timeline, vertices });
                            }

                            // Index Count & Buffer
                            let index_count = mesh_array[3].int()?;
                            let index_buffer = mesh_array[4].data()?.clone();
                            if index_buffer.len() as i32 != index_count * 2 {
                                error!("Index block size incorrect!");
                            }
                            let mut index_buffer = Cursor::new(index_buffer);
                            // And for HERO PNX, no matter how I change the index order
                            // The mesh data with normals are incorrect, comment out the following code
                            // and use only "DOUBLE SIDED" material? MAYBE...
                            // TODO: Comment out the code or not?!
                            let mut indices = vec![];
                            if direction.starts_with("MSHX") {
                                for _ in 0..index_count / 3 {
                                    indices.push(index_buffer.read_u16::<LittleEndian>()?);
                                    let b = index_buffer.read_u16::<LittleEndian>()?;
                                    let c = index_buffer.read_u16::<LittleEndian>()?;
                                    indices.push(c);
                                    indices.push(b);
                                }
                            } else {
                                for _ in 0..index_count {
                                    indices.push(index_buffer.read_u16::<LittleEndian>()?);
                                }
                            }

                            meshes.push(Mesh { sub_material, sub_meshes, index_count, indices });
                        }

                        // Matrices - MATRIX
                        let matrices_children = object_array[8].array()?;
                        let mut matrices = vec![];
                        for matrices_array in matrices_children {
                            let matrix_array = matrices_array.array()?;
                            // 0 - "MATRIX"
                            // 1..n - Matrix
                            trace!("\tMatrix - Count: {}", matrix_array.len() - 1);
                            for m in matrix_array.iter().skip(1) {
                                let mut buffer = Cursor::new(m.data()?.clone());
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
                                let timeline = buffer.read_i32::<LittleEndian>()?;
                                let matrix = [
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
                                ];

                                matrices.push(Matrix { timeline, matrix });
                            }
                        }

                        self.objects.push(Object { name: object_name, meshes, matrices });

                        // SFX Blocks?
                        if object_array.len() > 9 {
                            // TODO: Parse SFX
                            let sfx = object_array[9].array()?;
                            if !sfx.is_empty() {
                                warn!("\tSFX: Unhandled SFX blocks? {}@{}", file!(), line!());
                            }
                        }
                    }
                }
                _ => {
                    error!("Unknown block: {}", node_name);
                }
            }
        }

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

